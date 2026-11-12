# SmartPricer

SmartPricer is a machine-learning application that estimates the market price of a
smartphone from specifications that an ordinary phone owner can find quickly.
The application returns a point estimate and a likely price range instead of
pretending that a prediction is exact.

The project covers the complete workflow:

1. Raw smartphone data cleaning.
2. Feature extraction and standardisation.
3. Exploratory data analysis (EDA).
4. Model comparison and feature-ablation experiments.
5. Training and serialising the production model.
6. Serving predictions through a FastAPI API.
7. Providing a responsive browser interface.
8. Packaging the application in Docker.

## What the user does

The web page asks for four required values:

- Brand
- RAM
- Storage
- Whether the phone supports 5G

The user can optionally provide six more values to narrow the estimate:

- Screen size
- Battery capacity
- Main rear-camera megapixels
- NFC support
- Fast-charging wattage
- Android/iOS major version

The result is displayed in Pakistani rupees (PKR), with the model's underlying
prediction made in Indian rupees (INR). The conversion rate can be changed with
the `INR_TO_PKR` environment variable.

## Repository structure

```text
.
├── app/
│   ├── main.py                 FastAPI application and validation
│   ├── index.html              Browser UI and API client
│   ├── requirements.txt        Runtime Python dependencies
│   ├── model/
│   │   ├── model.pkl           Legacy model artifact
│   │   └── ct.pkl              Legacy transformer artifact
│   └── data/data.csv           Legacy application data
├── data/
│   ├── Smart-phone.xlsx        Original raw dataset
│   ├── Clean1_Smart_phone.csv  Output after structural cleaning
│   └── Clean2_Smart_Phones.csv Output after feature engineering
├── model/
│   └── model.joblib            Current trained model bundle
├── notebook/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_eda.ipynb
├── train.py                    Training, validation, and model export
├── Dockerfile                  Container image definition
└── .dockerignore               Files excluded from the image build context
```

The `app/model` files and `app/data/data.csv` belong to an older application
version. The current API loads `model/model.joblib` from the repository root.

## Data preparation and analysis

### Phase 1: raw data cleaning

[`notebook/01_data_cleaning.ipynb`](notebook/01_data_cleaning.ipynb) loads
`data/Smart-phone.xlsx` and produces `data/Clean1_Smart_phone.csv`.
The notebook addresses issues found in the source dataset, including:

- Extracting and standardising the brand/company field.
- Removing corrupt, misaligned, and non-smartphone rows.
- Correcting values shifted into the wrong columns.
- Fixing camera, display, operating-system, and memory-card inconsistencies.
- Performing sanity checks before export.

### Phase 2: feature engineering

[`notebook/02_feature_engineering.ipynb`](notebook/02_feature_engineering.ipynb)
converts the cleaned specifications into model-friendly columns and exports
`data/Clean2_Smart_Phones.csv`. This includes:

- Boolean flags for 5G, NFC, IR blaster, and memory-card support.
- Processor-name cleanup and manufacturer standardisation.
- Numeric processor cores and processor speed.
- Numeric RAM and storage in GB, including conversion of 1 TB to 1024 GB.
- Main rear-camera megapixels, front-camera megapixels, and camera counts.
- Screen size, refresh rate, battery, fast charging, and OS version.
- Card-slot type and dedicated card capacity.
- Removal of redundant raw columns and leakage-prone fields.

The resulting engineered file contains 981 rows and 28 columns in the checked-in
dataset. Prices in the source range from budget phones to luxury/outlier listings,
so the training script applies an explicit upper price boundary.

### Phase 3: exploratory data analysis

[`notebook/03_eda.ipynb`](notebook/03_eda.ipynb) investigates the prepared data
before modelling. It examines:

- Brand share and brand-versus-price behaviour.
- Price distribution and rating quality.
- Connectivity features such as 5G, NFC, and IR blaster.
- Processor manufacturers, names, speeds, and core-count segments.
- RAM, storage, camera, screen-size, operating-system, and card-slot patterns.
- Correlations and missing values.
- Baseline and boosting-model comparisons.
- Sanity-check predictions and deployment-oriented reference values.

The notebooks are analysis and reproducibility artifacts. The production training
path is implemented in `train.py`, so the deployed model does not depend on
running a notebook at request time.

## Model training

Run:

```bash
python train.py --data data/Clean2_Smart_Phones.csv
```

`train.py` performs the following work:

1. Removes generated index columns and normalises brand names.
2. Groups brands with fewer than five examples into `Other`.
3. Converts boolean-like and numeric columns safely.
4. Removes invalid prices and prices above `200,000` INR, which are treated as
   one-off luxury or mispriced rows for this use case.
5. Runs a feature-ablation experiment using repeated 5-fold cross-validation.
6. Fits the final model with preprocessing inside a scikit-learn pipeline.
7. Estimates 10th-90th percentile error bands using out-of-fold predictions.
8. Saves the trained bundle to `model/model.joblib`.

### Ablation experiment

The ablation table compares:

- The original full feature set, including rating.
- The full feature set without rating.
- Ten easy-to-answer fields.
- Four essential fields.
- Brand, RAM, and storage only.
- The final dropout-trained model with four answered fields.
- The same final model with all ten fields answered.

The output reports:

- `R2`: explained variance, retained for comparison with conventional
  regression reports.
- `MAE`: mean absolute error in INR.
- `MdAPE%`: median absolute percentage error, representing typical relative
  error more fairly than raw error for prices at different scales.
- `within20%`: percentage of phones predicted within 20% of the listed price.

The table is printed and saved as `model/ablation_results.csv`. This file is
generated output and may not exist until training is run.

### Final model design

The production model uses:

- `HistGradientBoostingRegressor`.
- `OrdinalEncoder` for brand, with safe handling of unknown brands.
- A `ColumnTransformer` and estimator inside one scikit-learn pipeline.
- `log1p`/`expm1` target transformation so the model learns relative price
  differences instead of being dominated by the most expensive phones.
- Feature-dropout augmentation: optional fields are randomly hidden during
  training so the same model can handle both the basic four-question flow and
  the detailed flow.

All preprocessing is fitted as part of the training pipeline. This avoids
fitting imputers or encoders on held-out folds and leaking information into
validation.

## Running locally

From the repository root on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r app\requirements.txt
```

The checked-in model can be served immediately:

```powershell
$env:INR_TO_PKR = "3.3"
python -m uvicorn app.main:app --reload
```

Open <http://localhost:8000>.

To retrain first:

```powershell
python train.py --data data\Clean2_Smart_Phones.csv
python -m uvicorn app.main:app --reload
```

Use `--skip-ablation` when only the final model is needed:

```powershell
python train.py --data data\Clean2_Smart_Phones.csv --skip-ablation
```

## API

The FastAPI application is implemented in [`app/main.py`](app/main.py).

### `GET /`

Returns the browser application from `app/index.html`.

### `GET /options`

Returns the supported brand list and the active INR-to-PKR conversion rate.

### `POST /predict`

Example request:

```json
{
  "Brand": "Samsung",
  "Ram": 8,
  "Memory": 128,
  "Has_5G": true,
  "Battery": 5000,
  "Screen_size": 6.5,
  "Back_Camera_MP": 50,
  "Has_NFC": true,
  "Fast_Charge": 33,
  "os_version": 14
}
```

The four required fields are validated by Pydantic. Optional fields may be
omitted or sent as `null`. Unknown brands are mapped to `Other`, matching the
training-time handling.

The response contains the estimated, low, and high values in both INR and PKR,
as well as the number of optional fields used:

```json
{
  "price_inr": 25000,
  "low_inr": 19000,
  "high_inr": 33000,
  "price_pkr": 82500,
  "low_pkr": 62700,
  "high_pkr": 108900,
  "optional_answered": 4,
  "optional_total": 6
}
```

The range is derived from cross-validated log-price errors. It is an uncertainty
estimate for the dataset, not a guarantee of resale value.

## Docker deployment

Build and run the container from the repository root:

```bash
docker build -t smartpricer .
docker run --rm -p 7860:7860 -e INR_TO_PKR=3.3 smartpricer
```

Open <http://localhost:7860>.

The image installs `app/requirements.txt`, copies the application source, and
copies the current `model/model.joblib`. Training notebooks and raw datasets
are intentionally not included in the runtime image.

## Re-running the notebooks

The notebooks document the original analysis stages and expect the repository
data files. Their analysis environment may need additional packages beyond the
small runtime list in `app/requirements.txt`, including Jupyter, `openpyxl`,
Matplotlib, and Seaborn. Install those in a separate development environment
rather than adding notebook-only dependencies to the production image.

Run notebooks in this order:

1. `01_data_cleaning.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_eda.ipynb`
4. `train.py`

## Important limitations

- The model estimates listed smartphone prices, not a guaranteed resale price.
- Phone condition, age, region, warranty, accessories, and current demand are
  not model inputs.
- The dataset is relatively small and may not represent every brand or market.
- Rare brands are grouped into `Other`.
- The INR-to-PKR conversion is external to the model and should be updated with
  the current rate.
- The error band describes historical cross-validation behaviour; it does not
  measure every source of real-world uncertainty.
- `model/model.joblib` must be regenerated when the training data or feature
  definitions change.
