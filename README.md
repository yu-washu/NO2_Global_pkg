# NO2_Global_pkg

A ML/DL pipeline for global surface NO2 estimation using a LightGBM/convolutional neural network (CNN/ResNet). The pipeline covers the full workflow from raw data processing to global map estimation and uncertainty quantification.

---

## Repository Structure

```
NO2_Global_pkg/
├── Data_Processing/          # Input data preparation pipeline
├── Training_pkg/             # Model architecture and training
├── Evaluation_pkg/           # Cross-validation and performance evaluation
├── Estimation_pkg/           # Global map estimation
├── Uncertainty_pkg/          # Uncertainty estimation from CV results
├── Mahalanobis_Uncertainty/  # Mahalanobis distance-based uncertainty mapping
├── Mask_func_pkg/            # Land/region masking utilities
├── Model_Evaluation_pkg/     # Offline model evaluation tools
├── visualization_pkg/        # Plotting functions
├── lassonet/                 # LassoNet for variable selection
├── main.py                   # Main entry point
└── config.toml               # All configuration settings
```

---

## Folder Descriptions

### `Data_Processing/`
Prepares all inputs and labels required for model training. Contains sub-packages for each data source and pipeline shell scripts for batch processing across years.

| Sub-folder | Description |
|---|---|
| `Derive_Label/` | Processes ground-based NO2 observations into training labels |
| `Derive_Geographical_Input/` | Derives geographic features (lat/lon coordinates) |
| `Derive_Geophysical_NO2/` | Derives GeoNO2 (satellite-based geophysical NO2) from TROPOMI/OMI via tessellation |
| `Get_CEDS_Anthro_Emission_Input/` | Processes CEDS anthropogenic NO2 emission inventories |
| `Get_GFED_Emission_Input/` | Processes GFED fire emission data |
| `Get_ISA_Input/` | Processes Impervious Surface Area (ISA) data |
| `Get_LandCover_Input/` | Processes land cover classification data |
| `Get_Meteorology_Variables_Input/` | Processes meteorological variables (wind, temperature, humidity, etc.) |
| `Get_NDVI_Input/` | Processes NDVI (vegetation index) data |
| `Get_OpenStreetMap_Input/` | Processes road network density/distance features from OpenStreetMap |
| `Get_PopulationMap_Input/` | Processes population density maps |
| `Regrid_GCHP/` | Regrids GCHP model output to the target resolution |
| `derive_TrainingDatasets/` | Assembles all processed inputs into the final training NetCDF dataset |

Pipeline automation scripts:
- `pipeline_2006_2022_auto.sh` — full pipeline for 2006–2022
- `pipeline_omi_2005_auto.sh` — OMI-based pipeline for 2005
- `pipeline_v513_2023_auto.sh` — pipeline for 2023 (v5.13 satellite product)
- `submit_parallel_2007_2022.sh` — parallel HPC job submission

---

### `Training_pkg/`
Defines the CNN/ResNet model architecture and training logic.

- `Net_Construction.py` — network architecture definition
- `Model_Func.py` — model construction helpers
- `Loss_Func.py` — custom loss functions
- `ConvNet_Data_Func.py` — data loading for convolutional inputs
- `data_func.py`, `iostream.py`, `utils.py` — data I/O and utilities
- `Statistic_Func.py` — statistical helpers used during training

---

### `Evaluation_pkg/`
Implements cross-validation strategies and model performance diagnostics.

| File | Description |
|---|---|
| `Spatial_CrossValidation.py` | Standard spatial cross-validation (AVD, sample-based) |
| `BLOO_CrossValidation.py` | Buffer Leave-One-Out (BLOO) spatial CV — excludes nearby sites within a buffer |
| `BLCO_CrossValidation.py` | Buffer Leave-Cluster-Out (BLCO) spatial CV — cluster-based hold-out |
| `Hyperparameter_Search_Validation.py` | Hyperparameter search with optional W&B sweep |
| `Sensitivity_Spatial_CrossValidation.py` | Variable inclusion/exclusion sensitivity tests |
| `SHAPvalue_analysis.py` | SHAP feature importance analysis |

---

### `Estimation_pkg/`
Applies a trained model to generate global NO2 concentration maps.

- `Estimation.py` — main estimation function; runs model inference month by month across all grid cells
- `predict_func.py` — model prediction utilities (ResNet and LightGBM)
- `training_func.py` — re-training utility for estimation-stage fine-tuning
- `Quality_Control.py` — post-estimation quality control (e.g., population-weighted mean calculation)
- `data_func.py`, `iostream.py`, `utils.py` — I/O and helper functions

---

### `Uncertainty_pkg/`
Derives spatially explicit uncertainty maps from cross-validation residuals.

- `uncertainty_estimation.py` — computes LOWESS-smoothed uncertainty as a function of model residuals; produces monthly and seasonal uncertainty maps
- `data_func.py`, `iostream.py`, `utils.py` — supporting functions

---

### `Mahalanobis_Uncertainty/`
Provides a complementary uncertainty estimate based on Mahalanobis distance — measuring how far each prediction point is from the training data distribution in feature space.

**Workflow:**
1. **Collect BLISCO test data** (`derive_resampled_trainingdata_BLISCO_data.py`) — assembles resampled training/testing site inputs from BLISCO cross-validation runs
2. **Compute Mahalanobis distance → rRMSE relationship** (`calculate_the_mahalanobis_distance_binned_rRMSE.py` + `mahalabobis_distance_uncertainty_test.ipynb`) — bins Mahalanobis distances and fits a distance-to-error curve
3. **Map uncertainty globally** (`derive_map_mahalanobis_uncertainty.py`) — computes Mahalanobis distance at every grid cell and maps it to an uncertainty estimate

Sub-packages: `data_func/`, `map_uncertainty_func/`, `visualization_pkg/`

---

### `Mask_func_pkg/`
Utilities for applying geographic and land-use masks (e.g., ocean masking, region selection) to estimation outputs.

---

### `Model_Evaluation_pkg/`
Offline tools for evaluating model inputs and outputs outside of the main training loop. Contains `input_eval/` and `output_eval/` sub-directories.

---

### `visualization_pkg/`
Plotting functions used throughout the pipeline.

| File | Description |
|---|---|
| `Assemble_Func.py` | Top-level plotting orchestration (loss curves, estimation maps, uncertainty maps) |
| `Estimation_plot.py` | Global/regional NO2 estimation map figures |
| `Evaluation_plot.py` | Regression scatter plots (obs vs. predicted) |
| `Training_plot.py` | Training/validation loss and accuracy curves |
| `Uncertainty_plot.py` | Uncertainty map figures |
| `LassoNet_plot.py` | LassoNet stability selection plots |
| `VIF_plot.py` | Variance Inflation Factor plots for collinearity analysis |
| `Addtional_Plot_Func.py` | Miscellaneous supplementary plots |

---

### `lassonet/`
LassoNet implementation for input variable stability selection. Used to identify the most important NO2 predictor variables before full model training.

- `StabilitySelection_AssembleFunc.py` — wraps LassoNet stability search
- Entry point in root: `lasso_stability_test.py`, `LassoNet.py`

---

## Entry Point

All pipeline stages are controlled through `main.py` with boolean switches in `config.toml`:

| Switch | Stage |
|---|---|
| `Hyperparameters_Search_Validation_Switch` | Hyperparameter search (optionally with W&B sweep) |
| `LassoNet_Stability_Selection_Switch` | Variable selection via LassoNet |
| `Spatial_CrossValidation_Switch` | Spatial cross-validation |
| `BLOO_CrossValidation_Switch` | BLOO cross-validation |
| `BLCO_CrossValidation_Switch` | BLCO cross-validation |
| `Estimation_Switch` | Global map estimation |
| `Uncertainty_Switch` | Uncertainty map derivation |
| `Sensitivity_Test_Switch` | Sensitivity tests (variable inclusion/exclusion) |

All input/output paths and model hyperparameters are configured in `config.toml`.

---

## HPC Job Scripts

| File | Description |
|---|---|
| `run_gpu.slurm` / `run_gpu.bsub` | GPU job submission (SLURM / LSF) |
| `run_cpu.bsub` | CPU-only job submission |
| `loop_years_script.sh` | Loop estimation over multiple years |
| `loop_variables_script.sh` | Loop sensitivity tests over variable groups |
| `loop_radius_script.sh` | Loop BLOO/BLCO over buffer radii |
