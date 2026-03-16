# (TraffCal)

Transformer-based traffic forecasting experiments and utilities.

This repository combines:
- A `Trafficformer` model implementation and training scripts
- Dataset assets for METR-LA and a `.pt`-based traffic dataset
- Supporting preprocessing and visualization artifacts

## Repository Layout

- `Trafficformer/` — model code, train/predict scripts, and saved weights
- `Trafficformer/data/metr-la/` — METR-LA files (`METR-LA.h5`, adjacency)
- `newdatset/` — tensor dataset files (`x.pt`, `y.pt`) and CSV exports
- `dataset.py` — helper script to download METR-LA via `kagglehub`

## Environment Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas tensorflow scikit-learn h5py torch kagglehub
```

## Data Preparation

Download/copy METR-LA files into `Trafficformer/data/metr-la/`:

```bash
python dataset.py
```

The `.pt` dataset is already included under `newdatset/`.

## Training

Run commands from `Trafficformer/`:

```bash
cd Trafficformer

# 1) METR-LA (mean across sensors)
python train_metrla.py \
	--h5_path data/metr-la/METR-LA.h5 \
	--n_days 20 \
	--mode mean \
	--epochs 20 \
	--batch_size 128 \
	--save_name metrla_mean

# 2) PT dataset
python train_ptdataset.py \
	--x_path ../newdatset/x.pt \
	--y_path ../newdatset/y.pt \
	--aggregate nodewise \
	--feature_index 0 \
	--train_size 23970 \
	--val_size 5141 \
	--test_size 5141 \
	--epochs 20 \
	--batch_size 128 \
	--save_name ptdataset_nodewise_23970_5141_5141
```

You can also run the bundled helper script:

```bash
bash run_two_datasets.sh
```

## Legacy `.traff` Training + Prediction

For the original single-series workflow:

```bash
cd Trafficformer
python train.py --filename example.traff --n_days 20 --save_name example
python predict.py --filename predict.traff --save_name example
```

## Outputs

- Model artifacts are saved to `Trafficformer/model/`
- Visualization outputs are in `Trafficformer/visualization/`

## Notes

- The folder name `newdatset/` is kept as-is to match the current repository contents.
- `run_two_datasets.sh` references `../newdataset/`; update it to `../newdatset/` if needed in your local copy.
- Some files are large (for example `.pt` and `.h5`); Git LFS is recommended for long-term storage.
