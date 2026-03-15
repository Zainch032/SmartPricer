# 📱 SmartPricer — End to End Data Science & ML Project

A complete machine learning project that predicts smartphone prices based on technical specifications. Built end-to-end from raw data collection to a deployed web application.

---

## 🧠 What I Did — End to End

### 🔗 Live Demo
- Hugging Face Space: [`SmartPricer` on Hugging Face](https://huggingface.co/spaces/Zainch032/Smart_Pricer)

### 1. 📊 Data Collection & Understanding
- Collected a real-world smartphone dataset with **25+ features** covering brand, processor, camera, battery, display, OS, connectivity and storage specs
- Used `df.info()` to check data types, null counts and memory usage
- Used `df.describe()` to understand statistical distribution of numerical features (min, max, mean, std)
- Used `df.nunique()` to see cardinality of each column — helped decide encoding strategy
- Identified that **price** ranged from budget phones under ₹5,000 to ultra-premium devices above ₹1,50,000

---

### 2. 🧹 Data Cleaning (Notebook: `Cleaning1.ipynb` & `cleaning2.ipynb`)

#### Initial Data Issues Fixed:
- **Column misalignment** — Fixed phones where battery data was in display column, camera data in wrong columns
- **Missing processor info** — Manually filled incomplete processor names (e.g., "Octa Core Processor" → full processor specs)
- **Brand standardization** — Fixed "OPPO" → "Oppo", extracted company names from model strings
- **Price filtering** — Removed phones below ₹3,400 (outliers/feature phones)
- **Data type fixes** — Converted RAM from MB to GB (512 MB → 0.5 GB), converted 1 TB storage to 1024 GB

#### Feature Extraction from Raw Strings:
- **Processor parsing** — Split "Snapdragon 888, Octa Core, 2.84 GHz" into Model, Name, Core, Speed
- **RAM & Memory** — Extracted from "12 GB RAM, 256 GB inbuilt" format
- **Battery & Fast Charge** — Parsed "5000 mAh Battery with 67W Fast Charging"
- **Display specs** — Extracted screen size, resolution, refresh rate from combined strings
- **Camera specs** — Split "108MP + 12MP + 5MP Triple Rear & 32MP Front" into separate features
- **Connectivity flags** — Extracted 5G, NFC, IR Blaster from SIM column
- **Memory card** — Parsed slot type (Dedicated/Hybrid) and capacity from card column
- **OS version** — Extracted numeric version from "Android v12" format

#### Final Cleaning Steps:
- **Missing value imputation** — Used KNN Imputer for numerical columns
- **Rating conversion** — Replaced "No Rating" with NaN, then imputed
- **Boolean conversion** — Converted True/False strings to 1/0 integers
- **Processor standardization** — Fixed typos, standardized brand names (Apple → Bionic, Mediatek → Dimensity)

---

### 3. 📈 Exploratory Data Analysis (Notebook: `eda_3.ipynb`)

#### Univariate Analysis (Distribution Plots):
- **Price distribution** — Histogram + KDE showing right-skewed data (skewness = 6.59), most phones under ₹50k
- **Rating distribution** — Histogram showing most phones rated 4.0-4.5, some outliers at 0 (No Rating)
- **Brand distribution** — Pie chart showing Xiaomi (15%), Samsung (12%), Realme (10%) as top brands
- **Processor brands** — Pie chart: Snapdragon (42%), Helio (20%), Dimensity (18%) dominate market
- **RAM distribution** — Pie chart: 4GB (30%), 6GB (25%), 8GB (20%) most common
- **Memory distribution** — Bar chart: 128GB (40%), 64GB (25%), 256GB (20%) most popular
- **Camera count** — Bar chart: Triple camera (50%), Dual (30%), Single (15%)
- **Slot types** — Pie chart: Dedicated (45%), Hybrid (15%), Not Supported (40%)

#### Bivariate Analysis (Feature vs Price):
- **Brand vs Price** — Bar chart: Apple/Leitz highest (₹1L+), Itel/Lava/Jio lowest (₹5k-10k)
- **Brand vs Rating** — Bar chart: Apple/Samsung highest ratings (80+), budget brands lower (70-75)
- **5G vs Price** — Point plot: 5G phones average ₹60k vs ₹25k for non-5G
- **NFC vs Price** — Bar chart: NFC phones average ₹70k vs ₹30k without NFC
- **IR Blaster vs Price** — Violin plot: Shows price distribution overlap, slight premium for IR Blaster
- **Processor Model vs Price** — Bar chart: Bionic (₹1.2L), Tensor (₹90k) highest; Unisoc (₹8k) lowest
- **Processor Model vs Rating** — Bar chart: Snapdragon/Dimensity/Helio rate 80+, Unisoc rates 65
- **Processor Core vs Price/Rating** — Dual-axis plot: 6-core (highest price), 8-core (highest rating), 4-core (lowest both)
- **Processor Speed vs Price** — Scatter plot: Positive correlation, 3.0+ GHz phones cost ₹80k+
- **Screen Size vs Rating** — Scatter plot: 6.5-6.7 inch screens have best ratings
- **OS vs Processor Core** — Stacked bar: Android uses 8-core, iOS uses 6-core predominantly

#### Correlation Analysis:
- **Heatmap** — Full correlation matrix of all numerical features
- **Top correlations with price**: RAM (0.65) > Processor Speed (0.58) > Back Camera MP (0.52) > Screen Size (0.45) > Refresh Rate (0.40)
- **Weak correlations**: Battery (0.15), os_version (0.12), capacity_gb (0.08)

---

### 4. ⚙️ Feature Engineering

This was the most important and time-consuming step — transforming raw messy columns into clean, model-ready features.

#### Camera Feature Extraction
Raw camera data was a single messy string like `"108MP + 12MP + 5MP, 32MP"`. Parsed and split into:
- `Back_Camera_MP` — main rear camera megapixels
- `Num_Back_Cam` — total number of rear cameras
- `Front_CAM_MP` — front camera megapixels
- `Num_Front_Cam` — number of front cameras

#### Processor Feature Extraction
Raw processor name like `"Snapdragon 888 Octa-core 2.84GHz"` was parsed into:
- `Processor_Model` — brand family (Snapdragon, Dimensity, Bionic, Exynos, Kirin, Tensor, Helio, Unisoc)
- `Processor_Core` — number of cores (4, 6, 8)
- `Processor_Speed` — clock speed in GHz

#### Connectivity Binary Flags
Extracted boolean features from spec strings:
- `Has_5G` — 1 if phone supports 5G, else 0
- `Has_NFC` — 1 if phone has NFC, else 0
- `Has_IR_Blaster` — 1 if phone has IR blaster, else 0

#### Storage Features
- `card_support` — 1 if expandable storage supported, else 0
- `slot_type` — type of slot: `Dedicated`, `Hybrid`, or `Not Slot`
- `capacity_gb` — maximum expandable storage capacity in GB

#### Fast Charging
- `Fast_Charge` — charging wattage extracted from spec (0 if no fast charge)

#### Final Feature Set
After all engineering, the model was trained on **24 clean, meaningful features** — each directly interpretable and relevant to price prediction.

---

### 5. 🔄 Preprocessing Pipeline
- Built a **ColumnTransformer** (`ct`) to handle both categorical and numerical features in one clean pipeline:
  - **Categorical columns** → Encoded using OrdinalEncoder / LabelEncoder
  - **Numerical columns** → Scaled using StandardScaler / MinMaxScaler
- Saved the transformer as `ct.pkl` using `pickle` — critical for deployment so the exact same transformations are applied at prediction time as during training

---

### 6. 🤖 Model Training (Notebook: `eda_3.ipynb`)

#### Model Comparison:
Tested multiple algorithms on 80/20 train/test split:
- **Linear Regression**: R² = 0.76
- **Random Forest**: R² = 0.85
- **Gradient Boosting**: R² = 0.79
- **XGBoost**: R² = 0.90 ✅ (Best performance)
- **LightGBM**: R² = 0.88
- **Ridge/ElasticNet**: R² = 0.76-0.79

#### Final Model (XGBoost):
- **Parameters**: n_estimators=260, learning_rate=0.05, max_depth=5, subsample=0.8
- **Performance**: R² = 0.903, RMSE = 7,724, MSE = 59,656,463
- **Preprocessing**: StandardScaler applied to all features before training
- **Saved artifacts**: `model.pkl` (trained XGBoost) and `ct.pkl` (ColumnTransformer)

---

### 7. 🚀 Deployment with FastAPI
- Built a production REST API using **FastAPI**
- `/predict` endpoint — accepts all 24 phone specs as JSON, runs `ct.transform()` + `model.predict()`, returns predicted price in INR
- `/processor-data` endpoint — serves processor lookup from real dataset for the frontend drill-down UI
- `/` route — serves `index.html` directly so no separate web server is needed
- CORS middleware enabled for cross-origin requests

---

### 8. 🌐 Frontend (HTML/CSS/JS)
- Built a clean, dark-themed UI in plain HTML, CSS and JavaScript — no framework needed
- **2-level processor drill-down**: pick brand (Snapdragon, Dimensity etc.) → pick exact model → Core & Speed auto-fill automatically
- **Toggle switches** for boolean features (5G, NFC, IR Blaster, Fast Charge, Memory Card)
- **Dropdowns** populated with exact unique values from the training dataset to prevent any mismatch with the model
- Price displayed in **PKR** (converted from INR on the frontend using exchange rate of 1 INR ≈ 3.5 PKR)
- Sends form data to FastAPI backend via `fetch()` API call
- Responsive design with smooth animations and loading states

---

## 🗂️ Project Structure

```
SmartPricer/
├── app/
│   ├── main.py          ← FastAPI backend (API + serves frontend)
│   ├── index.html       ← Frontend UI (HTML/CSS/JS)
│   ├── requirements.txt ← Python dependencies
│   ├── data/
│   │   └── data.csv     ← Final cleaned dataset (used for processor lookup)
│   └── model/
│       ├── model.pkl    ← Trained XGBoost model
│       └── ct.pkl       ← Fitted ColumnTransformer
├── data/                ← Raw & intermediate data files
│   ├── Smart-phone.xlsx
│   ├── Clean1_Smart_phone.csv
│   └── Clean2_Smart_Phones.csv
├── notebook/            ← Jupyter notebooks (EDA, cleaning, training)
│   ├── Cleaning1.ipynb
│   ├── cleaning2.ipynb
│   └── eda_3.ipynb
└── README.md
```

---

## ⚙️ Setup & Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Install dependencies
```bash
cd app
pip install -r requirements.txt
```

**Note:** If you're training the model yourself, you'll also need:
- `xgboost` for the ML model
- `matplotlib` and `seaborn` for EDA visualizations
- `jupyter` for running notebooks

### 2. Run the server
```bash
cd app
uvicorn main:app --reload
```

The `--reload` flag enables auto-reload on code changes (useful for development).

### 3. Open in browser
```
http://localhost:8000
```

### 4. API Documentation
Interactive API docs (Swagger UI) available at:
```
http://localhost:8000/docs
```

Alternative ReDoc documentation:
```
http://localhost:8000/redoc
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.8+ |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Model | XGBoost Regressor |
| Preprocessing | Scikit-learn (ColumnTransformer, StandardScaler, OrdinalEncoder) |
| Backend API | FastAPI, Uvicorn |
| API Validation | Pydantic |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Data Serialization | Pickle (for model persistence) |

---

## 📓 Notebooks Overview

The project includes three Jupyter notebooks documenting the complete workflow:

1. **`Cleaning1.ipynb`** — Initial data cleaning
   - Fixed column misalignments
   - Fixed wrong data placements
   - Standardized brand names
   - Removed outliers and feature phones

2. **`cleaning2.ipynb`** — Feature engineering
   - Extracted 24 features from raw text columns
   - Parsed processor, camera, battery, display specs
   - Created connectivity flags (5G, NFC, IR Blaster)
   - Handled missing values with KNN imputation

3. **`eda_3.ipynb`** — Analysis and modeling
   - Univariate and bivariate visualizations
   - Correlation analysis with heatmaps
   - Model comparison and selection
   - Final XGBoost training and evaluation

---

## 🔑 Key Learnings

- Real-world data is messy — **data cleaning and feature engineering took more time than model training**
- Always save the **ColumnTransformer alongside the model** — without `ct.pkl`, deployment predictions will be completely wrong
- **EDA is not optional** — visualizations revealed that RAM and processor speed matter far more than battery for pricing
- **FastAPI** makes it easy to serve ML models as production-ready REST APIs in just a few lines
- How to connect a plain HTML frontend to a Python backend using `fetch()` with no framework needed
- Feature engineering from raw strings (camera specs, processor names) dramatically improved model quality
