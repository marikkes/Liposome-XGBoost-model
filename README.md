# Formulations Database & ML Optimization

A machine learning system for optimizing pharmaceutical formulations using ensemble methods, active learning, and Bayesian optimization. This repository combines database management, feature engineering, and predictive modeling to suggest next experiments in a formulation optimization workflow.

## Overview

This project implements an intelligent experiment suggestion system for drug formulation development. It:
- **Trains ensemble models** (XGBoost) to predict encapsulation efficiency (EE)
- **Generates candidate formulations** using PCA-based lipid space exploration
- **Quantifies uncertainty** through ensemble predictions
- **Measures novelty** to balance exploitation vs. exploration
- **Suggests optimal next experiments** using an acquisition function that combines predicted performance, uncertainty, and novelty

The system is designed to minimize the number of wet-lab experiments needed to find high-performing formulations.

## Project Structure

```
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
│   ├── PCA analysis/              # Exploratory analysis scripts
│   ├── train_xgboost.py           # XGBoost hyperparameter tuning
│   ├── train_baseline_RF.py       # Random Forest baseline
│   ├── suggest_next_experiments.py # Active learning & experiment suggestion
│   ├── make_dataset.py            # Dataset creation from databases
│   ├── feature_engineering.py     # Feature transformation
│   ├── formulation_utils.py       # Formulation generation & utilities
│   ├── get_api_profile.py         # API (drug) property extraction
│   ├── get_lipid_profile.py       # Lipid property extraction
│   ├── lipid_utils.py             # Lipid data utilities
│   ├── formulation_run_db.py      # Experiment logging
│   ├── load_data.py               # Data loading utilities
│   ├── plot_best_formulations.py  # Visualization
│   ├── X.csv                      # Feature matrix cache
│   └── y.csv                      # Target (EE) cache
│
├── models/                        # Serialized trained models
│   ├── api_pca_model.joblib      # PCA model for API space
│   └── lipid_pca_model.joblib    # PCA model for lipid space
│
├── scripts/                       # Database & ETL scripts
│   ├── create_master.py          # Initialize master database
│   ├── copy_master_to_work.py    # Workflow: master → work
│   ├── promote_work_to_master.py # Workflow: work → master
│   ├── read_db.py                # Database inspection
│   ├── import_excel_to_db.py     # Excel → database import
│   ├── api db/                   # API property database scripts
│   │   ├── fetch_smiles.py       # Download SMILES from PubChem
│   │   ├── create_api_properties_db.py
│   │   ├── calculate_api_descriptors.py
│   │   ├── import_backup_properties.py
│   │   └── utils.py
│   └── lipid db/                 # Lipid database scripts
│       ├── create_lipid_db.py
│       └── import_lipid_properties.py
│
├── tests/                        # Test suite
│   ├── conftest.py              # Pytest configuration
│   ├── test_make_dataset.py
│   ├── test_suggest_next_experiments.py
│   └── test_train_xgboost.py
│
└── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites
- Python 3.13+
- Conda or pip for package management

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

### 3. Generate Experiment Suggestions

Suggest the next 5 most promising formulations:
```bash
python ml/suggest_next_experiments.py
```

This will:
- Load trained ensemble models
- Generate 5000 candidate formulations using PCA-based lipid space
- Predict performance and uncertainty for each candidate
- Compute novelty (distance from existing experiments)
- Score candidates using an acquisition function
- Select 5 diverse top candidates

### 4. Use a Specific API

Change the drug target by modifying the `api_name` in `ml/classes/experiment_config.py` (default: "Micrococcin P1").

## Key Components

### ExperimentConfig
Central configuration dataclass that manages:
- Model ensemble
- Feature columns from dataset
- API profile (molecular properties)
- Lipid selection mode (PCA or RANDOM)
- Acquisition function parameters (beta, gamma)
- Candidate generation settings

**Acquisition Modes:**
- `exploitation` (β=0.2, γ=0.1): Prioritize predicted EE
- `balanced` (β=0.8, γ=0.4): Balance EE, uncertainty, novelty
- `exploration` (β=1.5, γ=1.0): Maximize exploration

### Dataset Creation
`make_dataset()` combines:
- Formulation data (API ratio, lipid fractions)
- API properties (molecular weight, descriptors)
- Lipid properties (SMILES, molecular weight)
- Target values (encapsulation efficiency)

Output: Feature matrix X and target vector y

### Model Training
- **Baseline**: Random Forest (`train_baseline_rf.py`)
- **Production**: XGBoost ensemble with cross-validation (`train_xgboost.py`)
- **Hyperparameter Optimization**: Optuna with GroupKFold (prevents data leakage)

### Experiment Suggestion
1. **Candidate Generation**: Creates diverse formulations in PCA-projected lipid space
2. **Uncertainty Quantification**: Ensemble predictions yield mean and std
3. **Novelty Computation**: Distance to nearest existing experiment
4. **Acquisition Scoring**: Combines mean, uncertainty, novelty with learnable weights
5. **Diverse Selection**: Max-min distance sampling to avoid redundant suggestions

## Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test file:
```bash
python -m pytest tests/test_suggest_next_experiments.py -v
```

**Test Coverage:**
- `test_suggest_next_experiments.py`: Uncertainty quantification, novelty, acquisition function
- `test_train_xgboost.py`: Ensemble prediction, penalty application, objective function
- `test_make_dataset.py`: Dataset creation and validation

## Database Workflow

```
Raw Data (Excel)
    ↓
create_master.py
    ↓
Master DB (Production)
    ↓
copy_master_to_work.py
    ↓
Work DB (Development)
    ↓ (After experiments)
promote_work_to_master.py
    ↓
Master DB (Updated)
```

## Configuration & Customization

### Change Target API
Edit `ml/classes/experiment_config.py`:
```python
api_name: str = "Your_Drug_Name"  # Default: "Micrococcin P1"
```

### Adjust Acquisition Function
In `experiment_config.py`, modify acquisition mode or beta/gamma values:
```python
acquisition_mode: str = "exploration"  # or "balanced" / "exploitation"
```

### Tune Candidate Generation
```python
n_candidates: int = 5000              # Number of candidates to generate
n_formulation_trials: int = 500       # Optuna trials for formulation optimization
n_suggestions: int = 5                # Top experiments to suggest
n_pca_components: int = 3             # PCA dimensionality
```

## Dependencies

**Core:**
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (ML utilities, PCA, cross-validation)
- xgboost (gradient boosting models)
- sqlalchemy (database ORM)

**Database:**
- psycopg2 (PostgreSQL adapter)
- alembic (database migrations)

**Optimization:**
- optuna (hyperparameter tuning)

**Visualization:**
- matplotlib, pillow (plotting)

**Data I/O:**
- openpyxl (Excel files)
- PyPDF2 (PDF processing)

See `requirements.txt` for pinned versions.

## Performance Notes

- **PCA Lipid Space**: Reduces lipid dimensionality from N lipids → 3 components
- **Candidate Generation**: ~5000 formulations evaluated per suggestion round
- **Ensemble Predictions**: 5 XGBoost models for uncertainty quantification
- **Novelty Scaling**: StandardScaler fit on existing data before distance computation

## Troubleshooting

**Import Errors:**
- Ensure you're running from the project root directory
- The `conftest.py` file adds `ml/` to sys.path for imports

**Database Connection Issues:**
- Check PostgreSQL is running (if using remote database)
- Verify connection strings in config

**Missing Models:**
- Run `train_xgboost.py` to generate trained models in `models/`

