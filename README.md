# 📱 SmartPricer — End to End Data Science & ML Project

A complete machine learning project that predicts smartphone prices based on technical specifications. Built end-to-end from raw data collection to a deployed web application.

---

## 🧠 What I Did — End to End

### 1. 📊 Data Collection & Understanding
- Collected a real-world smartphone dataset with **25+ features** covering brand, processor, camera, battery, display, OS, connectivity and storage specs
- Used `df.info()` to check data types, null counts and memory usage
- Used `df.describe()` to understand statistical distribution of numerical features (min, max, mean, std)
- Used `df.nunique()` to see cardinality of each column — helped decide encoding strategy
- Identified that **price** ranged from budget phones under ₹5,000 to ultra-premium devices above ₹1,50,000

---

### 2. 🧹 Data Cleaning
- **Missing values** — analyzed null percentage per column, dropped columns with >50% missing, filled remaining with median (numerical) or mode (categorical)
- **Typos & inconsistencies** — fixed processor name errors like `Sanpdragon 680` → `Snapdragon 680`, standardized brand names to title case
- **Data type conversion** — converted boolean strings (`"True"/"False"`) to integers (`1/0`), resolution strings to numeric, OS version to float
- **Duplicate removal** — identified and dropped exact duplicate rows
- **Outlier handling** — found extreme battery values (21000 mAh, 22000 mAh) and kept them as they represent real niche devices
- **Irrelevant columns** — dropped model name and other non-predictive identifier columns

---

### 3. 📈 Exploratory Data Analysis (EDA)

#### Price Distribution
- Plotted **histogram** and **KDE plot** of price — found right-skewed distribution (most phones are budget, few are premium)
- Applied **log transformation** on price to check if it normalizes the distribution
- Plotted **boxplot** to visualize price spread and detect outliers

#### Brand Analysis
- **Bar chart** of average price per brand — Apple, Samsung flagships and Leitz were highest; Itel, Lava, Jio were lowest
- **Count plot** of number of phones per brand — Xiaomi, Samsung and Realme had the most entries in dataset
- **Box plot** of price grouped by brand — showed high variance within Samsung (budget to flagship range)

#### Feature vs Price Relationships
- **Scatter plot** RAM vs Price — clear positive correlation, 12GB+ phones were almost always premium
- **Scatter plot** Battery vs Price — weak correlation, budget phones now have large batteries too
- **Scatter plot** Back Camera MP vs Price — 108MP and 200MP sensors strongly associated with high price
- **Scatter plot** Processor Speed vs Price — higher GHz generally means higher price
- **Bar chart** Average price by 5G support — 5G phones cost significantly more on average
- **Bar chart** Average price by NFC — NFC phones were consistently priced higher
- **Bar chart** Average price by Refresh Rate — 120Hz+ phones had noticeably higher prices than 60Hz

#### Correlation Analysis
- Built **heatmap** of all numerical features vs price
- Found strongest positive correlations: **RAM > Processor Speed > Back Camera MP > Screen Size > Refresh Rate**
- Found weak or no correlation: **Battery capacity, os_version, capacity_gb**

#### Processor & OS Analysis
- **Pie chart** of OS distribution — Android dominated at 90%+, iOS small share but highest average price
- **Bar chart** of average price by Processor_Model — Bionic (Apple) and Tensor (Google) had highest averages, Unisoc the lowest

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

### 6. 🤖 Model Training
- Trained an **XGBoost Regressor** on the processed feature matrix
- Target variable: `price`
- Split data 80/20 into train/test sets using `train_test_split`
- Evaluated using **MAE**, **RMSE**, and **R² score**
- Saved the trained model as `model.pkl`

---

### 7. 🚀 Deployment with FastAPI
- Built a production REST API using **FastAPI**
- `/predict` endpoint — accepts all 24 phone specs as JSON, runs `ct.transform()` + `model.predict()`, returns predicted price
- `/processor-data` endpoint — serves processor lookup from real dataset for the frontend drill-down UI
- `/` route — serves `index.html` directly so no separate web server is needed

---

### 8. 🌐 Frontend (HTML/CSS/JS)
- Built a clean, dark-themed UI in plain HTML, CSS and JavaScript — no framework needed
- **2-level processor drill-down**: pick brand (Snapdragon, Dimensity etc.) → pick exact model → Core & Speed auto-fill automatically
- **Toggle switches** for boolean features (5G, NFC, IR Blaster, Fast Charge, Memory Card)
- **Dropdowns** populated with exact unique values from the training dataset to prevent any mismatch with the model
- Price displayed in **PKR** (auto-converted from INR using exchange rate)
- Sends form data to FastAPI backend via `fetch()` API call

---

## 🗂️ Project Structure

```
SmartPricer/
├── app/
│   ├── main.py          ← FastAPI backend (API + serves frontend)
│   ├── index.html       ← Frontend UI (HTML/CSS/JS)
│   ├── requirements.txt ← Python dependencies
│   ├── data/
│   │   └── data.csv     ← Final cleaned dataset
│   └── model/
│       ├── model.pkl    ← Trained XGBoost model
│       └── ct.pkl       ← Fitted ColumnTransformer
├── data/                ← Raw & intermediate data files
├── notebook/            ← Jupyter notebooks (EDA, cleaning, training)
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
cd app
pip install -r requirements.txt
```

### 2. Run the server
```bash
cd app
uvicorn main:app --reload
```

### 3. Open in browser
```
http://localhost:8000
```

API docs available at:
```
http://localhost:8000/docs
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Model | XGBoost |
| Preprocessing | Scikit-learn ColumnTransformer |
| Backend API | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Server | Uvicorn |

---

## 🔑 Key Learnings

- Real-world data is messy — **data cleaning and feature engineering took more time than model training**
- Always save the **ColumnTransformer alongside the model** — without `ct.pkl`, deployment predictions will be completely wrong
- **EDA is not optional** — visualizations revealed that RAM and processor speed matter far more than battery for pricing
- **FastAPI** makes it easy to serve ML models as production-ready REST APIs in just a few lines
- How to connect a plain HTML frontend to a Python backend using `fetch()` with no framework needed
- Feature engineering from raw strings (camera specs, processor names) dramatically improved model quality
