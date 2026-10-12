# SPGT for Next-Admission ICD Prediction

This repository contains the core implementation of the Soft-Prior Gated
Transformer (SPGT) for multi-label ICD prediction at the next hospital
admission.

SPGT combines representations of historical admissions with an
admission-level K-Means prior. The soft prior is derived from distances to
cluster centroids and is fused with the sequence representation through an
adaptive gate.

## Requirements

- Python 3.9 or later
- PyTorch
- NumPy
- pandas
- scikit-learn
- tqdm

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data

The training script expects a prepared data directory supplied with
`--data_dir`. Clinical data are not included in this repository. Users must
obtain access to the source clinical database and prepare the required files
in accordance with its data-use agreement.

## Training

Run the full SPGT configuration with soft cluster assignment and adaptive
cluster gating:

```bash
python train_spgt.py \
  --data_dir PATH_TO_PREPARED_DATA \
  --output_dir outputs/spgt \
  --use_soft_cluster \
  --use_cluster_gate \
  --adm_history 2 \
  --seed 42
```

Useful ablation options include `--disable_cluster` and `--disable_demo`.
Run `python train_spgt.py --help` for all available arguments.

## Privacy

No patient-level data, trained checkpoints, extracted representations, or
prediction files are distributed with this code.
