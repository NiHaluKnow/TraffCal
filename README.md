# TraffCal (Traffic + Weather Forecasting)

This project compares multiple forecasting models on:

- **METR-LA base dataset** (traffic only)
- **METR-LA + weather integrated dataset**

The repository includes:

- Transformer baseline (`All_model/transformer.py`)
- Additional models (`Others Models/`): CNN, GNN, Hybrid CNN-LSTM, LSTM-BiLSTM, RF-XGBoost
- Data pipeline for weather integration
- Saved MAE result files and presentation plots

## Project Layout

- `All_model/` -> Transformer architecture
- `predict+train/` -> Transformer training scripts
- `DataPipelining/` -> Weather integration + tensor generation pipeline
- `Others Models/` -> Alternative model implementations
- `data/metr-la/` -> Original METR-LA files
- `newdatset/` -> Generated integrated tensors/metadata (large files are gitignored)
- `results /metr-la/` -> METR-LA experiment outputs
- `results /weather_integrated/` -> Weather-integrated experiment outputs
- `results /presentation_plots/` -> Comparison charts for slides

## Environment

Python 3.13 (venv used in this workspace):

```bash
source /home/nihal/Desktop/ML_50/.venv/bin/activate
```

Suggested dependencies:

```bash
pip install tensorflow torch pandas numpy scikit-learn matplotlib h5py scipy xgboost
```

## Run Transformer Baselines

From `Trafficking/`:

### 1) METR-LA base

```bash
/home/nihal/Desktop/ML_50/.venv/bin/python "predict+train/train_metrla.py" \
  --h5_path "data/metr-la/METR-LA.h5" \
  --mode mean \
  --epochs 30 \
  --batch_size 64 \
  --save_name metrla_full_run
```

Expected split (mean mode):

- Train/Val/Test = **23970/5141/5141**

### 2) Weather-integrated full dataset

```bash
/home/nihal/Desktop/ML_50/.venv/bin/python "predict+train/train_ptdataset.py" \
  --x_path "newdatset/x.pt" \
  --y_path "newdatset/y.pt" \
  --aggregate mean \
  --epochs 40 \
  --batch_size 64 \
  --save_name weather_integrated_full_run_v2
```

## Latest MAE Summary

From `results /presentation_plots/comparison_table.csv`:

| Model | METR-LA MAE | Weather MAE |
|---|---:|---:|
| Transformer | 1.201600 | 0.057000 |
| GNN | 0.837943 | 0.040604 |
| RF-XGBoost | 1.141856 | 0.062581 |
| LSTM-BiLSTM | 2.266045 | 0.118180 |
| Hybrid CNN-LSTM | 2.656890 | 0.086431 |
| CNN | 2.747160 | 0.135330 |

## Result Files

### METR-LA

- `results /metr-la/transformer_error.txt`
- `results /metr-la/cnn_error.txt`
- `results /metr-la/gnn_error.txt`
- `results /metr-la/hybrid_cnn_lstm_error.txt`
- `results /metr-la/lstm_bilstm_error.txt`
- `results /metr-la/rf_xgboost_error.txt`
- `results /metr-la/others_models_metrla_summary.json`

### Weather Integrated

- `results /weather_integrated/transformer_error.txt`
- `results /weather_integrated/cnn_error.txt`
- `results /weather_integrated/gnn_error.txt`
- `results /weather_integrated/hybrid_cnn_lstm_error.txt`
- `results /weather_integrated/lstm_bilstm_error.txt`
- `results /weather_integrated/rf_xgboost_error.txt`
- `results /weather_integrated/others_models_mae_summary.json`

## Presentation Plots

Generated charts are in `results /presentation_plots/`:

- `metrla_model_comparison.png`
- `weather_model_comparison.png`
- `cross_dataset_model_comparison.png`
- `line_cross_dataset_comparison.png`
- `line_improvement_percentage.png`
- `line_metrla_models.png`
- `line_weather_models.png`
- `comparison_table.csv`

## Notes

- Large generated artifacts are intentionally gitignored:
  - `newdatset/x.pt`
  - `newdatset/y.pt`
  - `newdatset/merged_scaled.csv`
  - `newdatset/merged_unscaled.csv`
- Keep quoted paths when commands include the folder name `results ` (with trailing space).
