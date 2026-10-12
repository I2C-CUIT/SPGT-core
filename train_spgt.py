"""
训练带聚类特征的最优简化模型
在OptimalSimpleModel基础上加入轨迹聚类特征
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
import pandas as pd
import argparse
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description="Train optimal ICD model with clustering features")
parser.add_argument('--data_dir', type=str, default='icd_prediction_data_with_temporal_k10_2dim',
                    help='Path to prepared ICD data directory')
parser.add_argument('--output_dir', type=str, default='optimal_model_with_clustering_no_leak_output',
                    help='Directory to save model and metrics')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Training epochs')
parser.add_argument('--disable_cluster', action='store_true',
                    help='Disable clustering features by masking all cluster ids')
parser.add_argument('--use_soft_cluster', action='store_true',
                    help='Use temperature-smoothed soft cluster distribution instead of hard one-hot')
parser.add_argument('--cluster_temperature', type=float, default=0.35,
                    help='Temperature for soft cluster distribution (smaller = sharper)')
parser.add_argument('--use_cluster_gate', action='store_true',
                    help='Use adaptive gate to modulate cluster branch contribution')
parser.add_argument('--adm_history', type=int, default=2, choices=[1, 2],
                    help='Number of historical admissions to use as model input (1 or 2)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducible experiments')
parser.add_argument('--disable_demo', action='store_true',
                    help='Disable demographic features by zero-masking demo inputs')
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

# ==================== 数据加载 ====================
data_dir = Path(args.data_dir)

with open(data_dir / "token_vocabulary.json", 'r') as f:
    token_vocab = json.load(f)
    
with open(data_dir / "icd_vocabulary.json", 'r') as f:
    icd_vocab = json.load(f)
    
with open(data_dir / "data_statistics.json", 'r') as f:
    stats = json.load(f)

vocab_size = len(token_vocab)
num_icds = len(icd_vocab)

print(f"Vocabulary size: {vocab_size}")
print(f"Number of ICD codes: {num_icds}")

K = 12  # 使用K=12的聚类结果

# ==================== Dataset ====================
class ICDPredictionDatasetWithClustering(Dataset):
    def __init__(self, samples):
        """
        samples: 从pkl文件加载的数据列表
        每个sample包含:
        - adm1_token_ids, adm1_cluster
        - adm2_token_ids, adm2_cluster
        - target_icd_labels (Adm3目标)
        - demo_features
        注意：为避免目标泄露，模型输入不使用Adm3 token/cluster
        """
        self.samples = samples
        self.disable_cluster = args.disable_cluster
        self.use_soft_cluster = args.use_soft_cluster
        self.cluster_temperature = max(args.cluster_temperature, 1e-6)
        self.adm_history = args.adm_history
        self.disable_demo = args.disable_demo
    
    def __len__(self):
        return len(self.samples)

    def _get_cluster_distances(self, sample, adm_key):
        """
        Read distances from one admission to all KMeans centroids.
        Compatible field names:
        - adm1_cluster_distances / adm2_cluster_distances
        - cluster_dist_adm1 / cluster_dist_adm2
        - cluster_distances_adm1 / cluster_distances_adm2

        Returns a FloatTensor with shape [K], or None when distance fields are absent.
        """
        candidate_keys = [
            f"{adm_key}_cluster_distances",
            f"cluster_dist_{adm_key}",
            f"cluster_distances_{adm_key}",
        ]

        for key in candidate_keys:
            if key not in sample or sample[key] is None:
                continue

            value = sample[key]

            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return None

            try:
                value = list(value)
            except TypeError:
                return None

            if len(value) != K:
                return None

            try:
                return torch.FloatTensor([float(v) for v in value])
            except (TypeError, ValueError):
                return None

        return None
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Token sequences for each admission
        sequences = []
        cluster_ids = []
        
        adm_keys = ['adm1'] if self.adm_history == 1 else ['adm1', 'adm2']

        for adm_key in adm_keys:
            token_key = f'{adm_key}_token_ids'
            cluster_key = f'{adm_key}_cluster'
            
            if token_key in sample:
                sequences.append(torch.LongTensor(sample[token_key]))
                # 聚类ID（如果存在）
                if (not self.disable_cluster) and cluster_key in sample:
                    cluster_ids.append(sample[cluster_key])
                else:
                    cluster_ids.append(-1)  # 缺失值
            else:
                # 填充空序列
                sequences.append(torch.zeros(1, dtype=torch.long))
                cluster_ids.append(-1)

        # If only Adm1 is used, keep admission slots aligned to 3 admissions.
        if self.adm_history == 1:
            sequences.append(torch.zeros(1, dtype=torch.long))
            cluster_ids.append(-1)

        # 第三次住院（目标住院）不作为输入，强制置空
        sequences.append(torch.zeros(1, dtype=torch.long))
        cluster_ids.append(-1)
        
        # Demographic features
        demo = sample.get('demo_features', {})
        if self.disable_demo:
            demo_vec = torch.zeros(3, dtype=torch.float32)
        else:
            demo_vec = torch.FloatTensor([
                demo.get('age_group', 5) / 10.0,
                demo.get('gender_encoded', 1) / 2.0,
                demo.get('race_encoded', 2) / 5.0
            ])
        
        # Cluster features:
        # - hard one-hot by default
        # - distance-based temperature softmax when distance fields are available:
        #   P_j = softmax(-d_j / T), where d_j is the distance to centroid j.
        # - fallback to the old one-hot temperature smoothing for old pkl files.
        cluster_features = []

        def placeholder_cluster_vec():
            if self.use_soft_cluster:
                return torch.ones(K) / K
            return torch.zeros(K)

        # Historical admissions only. Adm3 is the prediction target and must not use
        # its cluster id or distances as model input.
        for adm_idx, adm_key in enumerate(adm_keys):
            cid = cluster_ids[adm_idx]

            one_hot = torch.zeros(K)
            if (not self.disable_cluster) and cid >= 0 and cid < K:
                one_hot[cid] = 1.0

            if self.disable_cluster:
                cluster_features.append(placeholder_cluster_vec())
            elif self.use_soft_cluster:
                distances = self._get_cluster_distances(sample, adm_key)

                if distances is not None:
                    logits = -distances / self.cluster_temperature
                    probs = torch.softmax(logits, dim=0)
                    cluster_features.append(probs)
                else:
                    # Backward-compatible fallback: old pkl files only contain hard labels.
                    logits = one_hot / self.cluster_temperature
                    probs = torch.softmax(logits, dim=0)
                    cluster_features.append(probs)
            else:
                cluster_features.append(one_hot)

        # If only Adm1 is used, keep the Adm2 slot for the fixed 3*K input shape.
        if self.adm_history == 1:
            cluster_features.append(placeholder_cluster_vec())

        # Target admission slot: do not use Adm3 cluster or distances.
        cluster_features.append(placeholder_cluster_vec())

        cluster_vec = torch.cat(cluster_features)  # 3*K维
        
        # ICD labels
        labels = torch.FloatTensor(sample['target_icd_labels'])
        
        return sequences, demo_vec, cluster_vec, labels

# 自定义collate函数
def collate_fn(batch):
    sequences_batch = []
    demo_batch = []
    cluster_batch = []
    labels_batch = []
    
    max_lens = [0, 0, 0]
    for sequences, demo, cluster_vec, labels in batch:
        for i, seq in enumerate(sequences):
            max_lens[i] = max(max_lens[i], len(seq))
    
    for sequences, demo, cluster_vec, labels in batch:
        padded_seqs = []
        for i, seq in enumerate(sequences):
            if len(seq) < max_lens[i]:
                padded = torch.cat([seq, torch.zeros(max_lens[i] - len(seq), dtype=torch.long)])
            else:
                padded = seq
            padded_seqs.append(padded)
        
        sequences_batch.append(padded_seqs)
        demo_batch.append(demo)
        cluster_batch.append(cluster_vec)
        labels_batch.append(labels)
    
    return sequences_batch, torch.stack(demo_batch), torch.stack(cluster_batch), torch.stack(labels_batch)

# ==================== 模型定义 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class OptimalModelWithClustering(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=2,
                 num_icds=100, K=12, use_cluster_gate=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        self.use_cluster_gate = use_cluster_gate
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Admission-level Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.admission_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Demographic encoder
        self.demo_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Cluster feature encoder (可选：对one-hot做一个小的变换)
        self.cluster_encoder = nn.Sequential(
            nn.Linear(3 * K, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.trajectory_proj = nn.Sequential(
            nn.Linear(3 * embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Adaptive gate on cluster branch, conditioned on trajectory + demographics.
        if self.use_cluster_gate:
            self.cluster_gate = nn.Sequential(
                nn.Linear(320, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.Sigmoid()
            )
        else:
            self.cluster_gate = None

        self.cluster_residual = nn.Parameter(torch.tensor(0.01))
        
        # Classifier
        # 输入维度: 3次住院(3*512) + 人口学(64) + 聚类特征(64) = 1664
        self.classifier = nn.Sequential(
            nn.Linear(3 * embed_dim + 64 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_icds)
        )
    
    def forward(self, sequences, demographics, cluster_features, return_features=False):
        batch_size = len(sequences)
        
        # Process each admission
        admission_reprs = []
        for adm_idx in range(3):
            seqs = torch.stack([sequences[b][adm_idx] for b in range(batch_size)]).to(device)
            
            # 检查是否全为空序列
            if seqs.size(1) == 0 or (seqs == 0).all():
                # 空序列，用零向量
                adm_repr = torch.zeros(batch_size, self.embed_dim).to(device)
            else:
                # Embedding
                embedded = self.token_embedding(seqs)
                embedded = self.pos_encoding(embedded)
                
                # Transformer
                mask = (seqs == 0)
                # 避免全为True的mask
                if mask.all():
                    adm_repr = torch.zeros(batch_size, self.embed_dim).to(device)
                else:
                    encoded = self.admission_encoder(embedded, src_key_padding_mask=mask)
                    # Pooling (只对非padding的token求平均)
                    valid_mask = ~mask
                    if valid_mask.any():
                        encoded_masked = encoded * valid_mask.unsqueeze(-1)
                        adm_repr = encoded_masked.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    else:
                        adm_repr = encoded.mean(dim=1)
            
            admission_reprs.append(adm_repr)
        
        # Concatenate admission representations
        trajectory_repr = torch.cat(admission_reprs, dim=1)
        
        # Encode demographics
        demo_repr = self.demo_encoder(demographics)
        
        # Encode cluster features
        cluster_repr = self.cluster_encoder(cluster_features)
        traj_repr_proj = self.trajectory_proj(trajectory_repr)
        
        if self.cluster_gate is not None:
            cluster_diff = torch.abs(traj_repr_proj - cluster_repr)
            cluster_interaction = traj_repr_proj * cluster_repr
            gate_input = torch.cat([traj_repr_proj, demo_repr, cluster_repr, cluster_diff, cluster_interaction], dim=1)
            gate = self.cluster_gate(gate_input)
            cluster_repr = traj_repr_proj + self.cluster_residual * gate * cluster_repr
        else:
            cluster_repr = traj_repr_proj + self.cluster_residual * cluster_repr

        # Concatenate all features
        combined = torch.cat([trajectory_repr, demo_repr, cluster_repr], dim=1)
        if combined.ndim != 2 or combined.shape[1] != 1664:
            print(f"trajectory_repr shape: {trajectory_repr.shape}")
            print(f"demo_repr shape: {demo_repr.shape}")
            print(f"cluster_repr shape: {cluster_repr.shape}")
            raise ValueError(f"Expected combined feature shape [batch, 1664], got {combined.shape}")
        
        # Classify
        logits = self.classifier(combined)

        if return_features:
            return logits, combined

        return logits

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for sequences, demographics, cluster_features, labels in tqdm(dataloader, desc="Training"):
        demographics = demographics.to(device)
        cluster_features = cluster_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(sequences, demographics, cluster_features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for sequences, demographics, cluster_features, labels in tqdm(dataloader, desc="Evaluating"):
            demographics = demographics.to(device)
            cluster_features = cluster_features.to(device)
            
            logits = model(sequences, demographics, cluster_features)
            probs = torch.sigmoid(logits)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
    
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    # Micro-AUC
    micro_auc = roc_auc_score(all_labels.ravel(), all_preds.ravel())
    
    # Macro-AUC
    macro_aucs = []
    for i in range(num_icds):
        if all_labels[:, i].sum() > 0:
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                macro_aucs.append(auc)
            except:
                pass
    macro_auc = np.mean(macro_aucs) if macro_aucs else 0.0
    
    # Precision@k and Recall@k
    def precision_at_k(y_true, y_pred, k=10):
        top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
        precisions = []
        for i in range(len(y_true)):
            true_labels = set(np.where(y_true[i] == 1)[0])
            pred_labels = set(top_k_indices[i])
            if len(pred_labels) > 0:
                precisions.append(len(true_labels & pred_labels) / k)
        return np.mean(precisions)
    
    def recall_at_k(y_true, y_pred, k=20):
        top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
        recalls = []
        for i in range(len(y_true)):
            true_labels = set(np.where(y_true[i] == 1)[0])
            pred_labels = set(top_k_indices[i])
            if len(true_labels) > 0:
                recalls.append(len(true_labels & pred_labels) / len(true_labels))
        return np.mean(recalls)
    
    p_at_10 = precision_at_k(all_labels, all_preds, k=10)
    r_at_20 = recall_at_k(all_labels, all_preds, k=20)
    
    return {
        'micro_auc': micro_auc,
        'macro_auc': macro_auc,
        'precision_at_10': p_at_10,
        'recall_at_20': r_at_20
    }

# ==================== 主训练流程 ====================
print("\n" + "="*50)
print("加载数据...")

# 加载pkl数据
with open(data_dir / "train_data_traj3.pkl", 'rb') as f:
    data = pickle.load(f)

train_samples = data['train']
val_samples = data['val']
test_samples = data['test']

train_dataset = ICDPredictionDatasetWithClustering(train_samples)
val_dataset = ICDPredictionDatasetWithClustering(val_samples)
test_dataset = ICDPredictionDatasetWithClustering(test_samples)

# 检查第一个样本
sample = train_samples[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"✓ 使用轨迹聚类特征 (K={K})")
print("✓ 无泄露设置：只使用Adm1/Adm2输入，Adm3输入置空")
print(f"✓ Admission history: {args.adm_history}")
print(f"✓ Cluster feature enabled: {not args.disable_cluster}")
print(f"✓ Soft cluster encoding: {args.use_soft_cluster} (temp={args.cluster_temperature})")
print(f"✓ Cluster gate enabled: {args.use_cluster_gate}")
print(f"✓ Demographic feature enabled: {not args.disable_demo}")
print(f"✓ Random seed: {args.seed}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

print(f"训练集: {len(train_dataset)} 样本")
print(f"验证集: {len(val_dataset)} 样本")
print(f"测试集: {len(test_dataset)} 样本")

# 创建模型
print("\n" + "="*50)
print("创建模型...")
model = OptimalModelWithClustering(
    vocab_size=vocab_size,
    embed_dim=512,
    num_heads=8,
    num_layers=2,
    num_icds=num_icds,
    K=K,
    use_cluster_gate=args.use_cluster_gate
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# 训练设置
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练
print("\n" + "="*50)
print("开始训练...")
num_epochs = args.num_epochs
best_val_auc = 0
output_dir = args.output_dir
best_model_path = f"{output_dir}/best_model.pt"
Path(output_dir).mkdir(exist_ok=True)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_metrics = evaluate(model, val_loader)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Micro-AUC: {val_metrics['micro_auc']:.4f}")
    print(f"Val Macro-AUC: {val_metrics['macro_auc']:.4f}")
    print(f"Val P@10: {val_metrics['precision_at_10']:.4f}")
    print(f"Val R@20: {val_metrics['recall_at_20']:.4f}")
    
    if val_metrics['micro_auc'] > best_val_auc:
        best_val_auc = val_metrics['micro_auc']
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ 保存最佳模型 (Val AUC: {best_val_auc:.4f})")

# 测试最佳模型
print("\n" + "="*50)
print("加载最佳模型进行测试...")
model.load_state_dict(torch.load(best_model_path))
test_metrics = evaluate(model, test_loader)

print("\n" + "="*50)
print("【带聚类的最优简化模型（无泄露）- 测试结果】")
print("="*50)
print(f"Micro-AUC: {test_metrics['micro_auc']:.4f}")
print(f"Macro-AUC: {test_metrics['macro_auc']:.4f}")
print(f"Precision@10: {test_metrics['precision_at_10']:.4f}")
print(f"Recall@20: {test_metrics['recall_at_20']:.4f}")
print(f"总参数量: {total_params:,}")
print("="*50)

# 保存结果
results = {
    'test_metrics': test_metrics,
    'total_params': total_params,
    'model_config': {
        'embed_dim': 512,
        'num_layers': 2,
        'K': K,
        'with_clustering': not args.disable_cluster,
        'use_soft_cluster': args.use_soft_cluster,
        'cluster_temperature': args.cluster_temperature,
        'use_cluster_gate': args.use_cluster_gate,
        'adm_history': args.adm_history,
        'seed': args.seed,
        'with_demo': not args.disable_demo,
        'use_adm3_input': False,
        'leakage_blocked': True
    }
}

with open(f"{output_dir}/results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到 {output_dir}/results.json")