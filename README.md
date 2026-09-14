# Formulations Database & ML Optimization

A machine learning system for pharmaceutical formulation optimization using XGBoost ensembles, uncertainty estimation, active learning, and Bayesian optimization. The repository combines database management, feature engineering, predictive modeling, and experiment selection into a workflow designed to reduce wet-lab trial-and-error.

## Overview

This project implements an AI-guided experiment suggestion system for drug formulation development. It:

* **Trains XGBoost ensemble models** to predict encapsulation efficiency (EE)
* **Generates candidate formulations** using PCA-based exploration of lipid property space
* **Quantifies predictive uncertainty** through disagreement between ensemble models
* **Measures formulation novelty** relative to existing experiments
* **Ranks candidates** using an acquisition function combining predicted performance, uncertainty, and novelty
* **Selects diverse next experiments** to avoid testing highly similar formulations
* **Optimizes theoretical formulations** using the trained surrogate model and Optuna

The overall workflow is designed to reduce the number of wet-lab experiments needed to identify high-performing formulations.

## Workflow

The project separates model training, formulation optimization, and sequential experiment selection:

```text
Existing formulation data
        ↓
    Train XGBoost
      ensemble
        ↓
Generate candidate formulations
        ↓
Predict performance
 + uncertainty + novelty
        ↓
Acquisition function
        ↓
Select diverse next
     experiments
        ↓
      Wet lab
        ↓
Add experimental results
        ↓
      Retrain
        ↺
```

The trained surrogate can also be used independently to optimize a theoretical formulation using Optuna.

## Project Structure

```text
├── data/                          # Raw data storage
│   ├── excel/                     # Excel source data
│   └── papers/                    # Reference documents
│
├── db/                            # Database files
│   ├── archive/                   # Historical data
│   ├── master/                    # Production databases
│   └── work/                      # Working databases
│
├── ml/                            # Machine learning pipeline
│   ├── classes/
│   │   ├── experiment_config.py   # Configuration dataclass
│   │   └── pca_model.py           # PCA model wrapper
│   ├── PCA analysis/              # Exploratory PCA analysis scripts
│   ├── train_xgboost.py           # XGBoost hyperparameter tuning & ensemble training
│   ├── optimize_formulation.py    # Optuna-based formulation optimization
│   ├── suggest_next_experiments.py # Acquisition & next-experiment selection
│   ├── make_dataset.py            # Dataset creation from databases
│   ├── feature_engineering.py     # Feature transformation
│   ├── formulation_utils.py       # Formulation generation & utilities
│   ├── ml_utils.py                # Reusable ML utilities
│   ├── get_api_profile.py         # API (drug) property extraction
│   ├── get_lipid_profile.py       # Lipid property extraction
│   ├── lipid_utils.py             # Lipid data utilities
│   ├── formulation_run_db.py      # Run and experiment logging
│   ├── load_data.py               # Data loading utilities
│   ├── plot_best_formulations.py  # Visualization
│   ├── X.csv                      # Feature matrix cache
│   └── y.csv                      # Target (EE) cache
│
├── models/                        # Serialized trained models
│   ├── xgb_model_0.pkl            # XGBoost ensemble member
│   ├── xgb_model_1.pkl            # XGBoost ensemble member
│   ├── ...                        # Additional ensemble members
│   ├── api_pca_model.joblib       # PCA model for API space
│   └── lipid_pca_model.joblib     # PCA model for lipid space
│
├── scripts/                       # Database & ETL scripts
│   ├── create_master.py           # Initialize master database
│   ├── copy_master_to_work.py     # Workflow: master → work
│   ├── promote_work_to_master.py  # Workflow: work → master
│   ├── read_db.py                 # Database inspection
│   ├── import_excel_to_db.py      # Excel → database import
│   ├── api db/                    # API property database scripts
│   │   ├── fetch_smiles.py        # Download SMILES from PubChem
│   │   ├── create_api_properties_db.py
│   │   ├── calculate_api_descriptors.py
│   │   ├── import_backup_properties.py
│   │   └── utils.py
│   └── lipid db/                  # Lipid database scripts
│       ├── create_lipid_properties_db.py
│       ├── calculate_lipid_descriptors.py
│       └── fetch_smiles.py        # Download SMILES from PubChem
│       └── import_backup_properties.py
│       └── import_manual_formulation_descriptors.py
│   ├── db utils/                  # Property database script utils for APIs and lipids
│   │   ├── smiles_utils.py        
│   │   ├── descriptor_pipeline.py
│   │   ├── pubchem_synonyms.py
│   │   ├── rdkit_descriptors.py
│   │   └── database_utils.py
│
├── tests/                         # Test suite
│   ├── conftest.py                # Pytest configuration
│   ├── test_make_dataset.py       # Dataset creation tests
│   ├── test_ml_utils.py           # ML utility tests
│   ├── test_optimize_formulation.py # Formulation optimization tests
│   ├── test_suggest_next_experiments.py # Experiment suggestion tests
│   └── test_train_xgboost.py      # Model training tests
│
└── requirements.txt               # Python dependencies
```

## Installation

### Prerequisites

* Python 3.13+
* Conda or pip for package management

### Setup

1. **Clone/setup the repository:**

   ```bash
   cd "c:\Users\masun1863\Python projects\Formulations db"
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**

   ```bash
   python -m pytest tests/ -v
   ```

## Quick Start

### 1. Create Databases

Initialize the master database from raw data:

```bash
python scripts/create_master.py
```

### 2. Train Models

Train an XGBoost ensemble with Optuna hyperparameter tuning:

```bash
python ml/train_xgboost.py
```

The training pipeline:

* Creates the modelling dataset
* Splits data according to the configured split mode
* Tunes XGBoost hyperparameters using Optuna and GroupKFold cross-validation
* Trains an ensemble of five XGBoost models
* Evaluates the ensemble on the held-out test set
* Saves the trained models to `models/`
* Records the training run and performance metrics in the run database

### 3. Optimize a Formulation

Use the trained surrogate model to search for a formulation predicted to have high encapsulation efficiency:

```bash
python ml/optimize_formulation.py
```

This uses Optuna to optimize:

* Number of lipids
* Lipid composition
* Lipid fractions
* API-to-lipid ratio

The formulation optimizer is useful for asking:

> **"Given the current model, what formulation does it predict will perform best?"**

This is separate from the sequential next-experiment selection workflow.

### 4. Generate Next-Experiment Suggestions

Suggest the next formulations to test experimentally:

```bash
python ml/suggest_next_experiments.py
```

This will:

* Load the trained XGBoost ensemble
* Generate candidate formulations using PCA-based lipid space exploration
* Predict EE and ensemble uncertainty for each candidate
* Compute novelty relative to existing experiments
* Calculate an acquisition score
* Select diverse high-scoring candidates

By default, the system generates approximately 5000 candidate formulations and selects 5 suggestions.

This corresponds to the sequential question:

> **"Which formulations should we test next?"**

After experimental results are added to the database, the model can be retrained and the process repeated.

### 5. Use a Specific API

Change the drug target by modifying `api_name` in `ml/classes/experiment_config.py`.

Default:

```python
api_name: str = "Micrococcin P1"
```

## Key Components

### ExperimentConfig

`ExperimentConfig` is the central configuration dataclass used by the formulation and experiment-selection workflows.

It manages:

* Model ensemble
* Feature columns from the dataset
* API profile and molecular properties
* Data split mode
* Lipid selection mode (`PCA` or `RANDOM`)
* PCA dimensionality
* Acquisition function parameters
* Candidate generation settings
* Formulation optimization settings
* Number of experiments to suggest
* API-to-lipid ratio range

### Acquisition Modes

The acquisition function combines normalized predicted performance, ensemble uncertainty, and formulation novelty:

```text
score =
    normalized predicted EE
    + β × normalized uncertainty
    + γ × normalized novelty
```

Available modes:

* `exploitation` (β = 0.2, γ = 0.1): Primarily prioritize predicted EE
* `balanced` (β = 0.8, γ = 0.4): Balance predicted EE, uncertainty, and novelty
* `exploration` (β = 1.5, γ = 1.0): Strongly prioritize uncertainty and novelty

The ensemble standard deviation provides a practical measure of model disagreement. It should be interpreted as an ensemble-based uncertainty estimate rather than a calibrated Bayesian posterior uncertainty.

### Dataset Creation

`make_dataset()` combines:

* Formulation data

  * API-to-lipid ratio
  * Lipid fractions
* API properties

  * Molecular weight
  * Molecular descriptors
* Lipid properties

  * Molecular descriptors
  * SMILES-derived properties
* Target values

  * Encapsulation efficiency (EE)

Output:

* Feature matrix `X`
* Target vector `y`
* API groups used for grouped cross-validation and data splitting

### Model Training

The modelling pipeline includes:

* **Baseline:** Random Forest (`train_baseline_RF.py`)
* **Production model:** XGBoost ensemble (`train_xgboost.py`)
* **Hyperparameter optimization:** Optuna
* **Cross-validation:** GroupKFold to reduce data leakage between related API groups
* **Ensemble:** Five XGBoost models trained with the same optimized hyperparameters but different random seeds

The ensemble is used both for prediction and for estimating uncertainty through model disagreement.

### Formulation Optimization

`optimize_formulation.py` uses the trained XGBoost ensemble as a surrogate model and Optuna to search formulation space.

Candidate formulations vary in:

* Number of lipids: 1–3
* Lipid identity
* Lipid fractions
* API-to-lipid ratio

The formulation objective maximizes predicted EE while applying penalties to formulations with extreme lipid fractions.

This optimization is **model-guided optimization using the current surrogate**. It does not itself perform the sequential wet-lab Bayesian optimization loop.

### Next-Experiment Selection

`suggest_next_experiments.py` implements the sequential experiment-selection component:

1. **Candidate generation**
   Generate candidate formulations in lipid property space using PCA-guided exploration.

2. **Prediction**
   Predict EE using the XGBoost ensemble.

3. **Uncertainty estimation**
   Calculate the mean and standard deviation of ensemble predictions.

4. **Novelty computation**
   Calculate the distance between each candidate and the nearest existing formulation after feature standardization.

5. **Acquisition scoring**
   Combine predicted EE, uncertainty, and novelty.

6. **Diverse selection**
   Select high-scoring candidates while encouraging diversity between the suggested experiments.

This provides an AI-guided Bayesian optimization workflow in which the XGBoost ensemble acts as the surrogate model and ensemble disagreement contributes to the acquisition function.

## PCA-Based Lipid Exploration

PCA is used to represent lipids in a lower-dimensional property space.

The lipid PCA model:

* Standardizes lipid descriptors
* Projects lipid properties into principal components
* Uses the selected principal components to guide candidate lipid selection

The default configuration uses three principal components.

Two lipid-selection modes are available:

* `PCA`: Property-space-guided exploration
* `RANDOM`: Random lipid selection without PCA guidance

PCA therefore guides **which lipid combinations are explored**, while the acquisition function determines which complete formulations are prioritized for experimentation.

## Testing

Run all tests:

```bash
python -m pytest tests/ -v
```

Run a specific test file:

```bash
python -m pytest tests/test_suggest_next_experiments.py -v
```

Current tests cover:

* `test_make_dataset.py`

  * Dataset creation and validation
* `test_ml_utils.py`

  * Ensemble prediction
  * Empty-model handling
* `test_optimize_formulation.py`

  * Formulation objective
  * Lipid selection
  * Formulation penalties
* `test_suggest_next_experiments.py`

  * Uncertainty estimation
  * Novelty calculation
  * Acquisition scoring
  * Candidate selection
* `test_train_xgboost.py`

  * Training objective and model-training logic

## Database Workflow

```text
Raw Data (Excel)
    ↓
create_master.py
    ↓
Master DB (Production)
    ↓
copy_master_to_work.py
    ↓
Work DB (Development)
    ↓
Add / modify experimental data
    ↓
promote_work_to_master.py
    ↓
Master DB (Updated)
```

The work database is used during model development and formulation optimization, while the master database provides the production data source.

## Configuration & Customization

### Change Target API

Edit `ml/classes/experiment_config.py`:

```python
api_name: str = "Your_Drug_Name"
```

The default is:

```python
api_name: str = "Micrococcin P1"
```

### Change Data Split

The training pipeline supports three split modes:

```python
split_mode: str = "within_api"
```

Available modes:

* `within_api`: Split formulations within each API into training and test data
* `api`: Hold out entire APIs for testing
* `random`: Randomly split the complete dataset

### Adjust Acquisition Function

Change the acquisition mode in `experiment_config.py`:

```python
acquisition_mode: str = "balanced"
```

Available modes:

```text
exploitation
balanced
exploration
```

Alternatively, `beta` and `gamma` can be set explicitly.

### Tune Candidate Generation and Optimization

```python
n_candidates: int = 5000
n_formulation_trials: int = 1000
n_suggestions: int = 5
n_pca_components: int = 3
```

* `n_candidates`: Number of candidate formulations generated per suggestion round
* `n_formulation_trials`: Number of Optuna trials for theoretical formulation optimization
* `n_suggestions`: Number of experiments selected for the next batch
* `n_pca_components`: Number of PCA components used for lipid-space exploration

The API-to-lipid ratio can also be configured through:

```python
api_ratio_min: float = 1e-3
api_ratio_max: float = 1.0
```

## Dependencies

**Core:**

* pandas
* numpy
* scikit-learn
* xgboost
* sqlalchemy

**Database:**

* psycopg2
* alembic

**Optimization:**

* optuna

**Visualization:**

* matplotlib
* pillow

**Data I/O:**

* openpyxl
* PyPDF2

See `requirements.txt` for pinned versions.

## Performance Notes

* **PCA Lipid Space:** Represents lipid properties using a reduced number of principal components
* **Candidate Generation:** Approximately 5000 formulations are generated per suggestion round by default
* **Ensemble Predictions:** Five XGBoost models provide predictions and ensemble-based uncertainty estimates
* **Novelty Scaling:** Features are standardized using the existing experimental data before distance calculations
* **Diverse Selection:** Suggested formulations are selected using a combination of acquisition score and diversity

## Troubleshooting

### Import Errors

Ensure you're running commands from the project root directory.

The `conftest.py` file adds `ml/` to `sys.path` for imports during testing.

### Missing Models

Run:

```bash
python ml/train_xgboost.py
```

This generates the trained XGBoost ensemble in:

```text
models/xgb_model_0.pkl
models/xgb_model_1.pkl
...
models/xgb_model_4.pkl
```

### Database Connection Issues

Check that PostgreSQL is running if using the remote database, and verify the relevant connection settings.

### Missing PCA Model

Ensure that `models/lipid_pca_model.joblib` exists before using PCA-based formulation generation.
