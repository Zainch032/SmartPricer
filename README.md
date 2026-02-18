# 📱 Phone Price Predictor — End to End Data Science & ML Project

A complete machine learning project that predicts smartphone prices based on technical specifications. Built end-to-end from raw data collection to a deployed web application.

---

## 🧠 What I Did — End to End

### 1. 📊 Data Collection & Understanding
- Collected a real-world smartphone dataset with 25+ features including brand, processor, camera, battery, display specs
- Explored the data using **Pandas** to understand distributions, data types, and relationships
- Used `df.info()`, `df.describe()`, `df.unique()` to understand each column

### 2. 🧹 Data Cleaning
- Handled missing values — filled or dropped based on column importance
- Fixed inconsistent values (e.g. typos in processor names like `Sanpdragon 680`)
- Converted data types (strings to floats, booleans to 0/1)
- Removed duplicate rows and irrelevant columns

### 3. 📈 Exploratory Data Analysis (EDA)
- Visualized price distribution across brands using **Matplotlib** and **Seaborn**
- Found correlations between price and RAM, processor speed, camera MP
- Identified outliers (e.g. phones with 21000 mAh battery) and decided how to handle them
- Used heatmaps to find the most price-impacting features

### 4. ⚙️ Feature Engineering
- Split raw camera string into `Num_Back_Cam`, `Back_Camera_MP`, `Num_Front_Cam`, `Front_CAM_MP`
- Extracted `Processor_Model`, `Processor_Core`, `Processor_Speed` from processor name
- Created binary flags: `Has_5G`, `Has_NFC`, `Has_IR_Blaster`
- Engineered `card_support`, `slot_type`, `capacity_gb` from storage info

### 5. 🔄 Preprocessing Pipeline
- Built a **ColumnTransformer** (`ct`) to handle:
  - **Categorical columns** → Encoded (OrdinalEncoder / LabelEncoder)
  - **Numerical columns** → Scaled (StandardScaler / MinMaxScaler)
- Saved the transformer as `ct.pkl` using `pickle` so the same transformations apply at prediction time

### 6. 🤖 Model Training
- Trained an **XGBoost Regressor** (`XGB_model`) on processed features
- Target variable: `price`
- Split data into train/test using `train_test_split`
- Evaluated using **MAE**, **RMSE**, and **R² score**
- Saved the trained model as `model.pkl`

### 7. 🚀 Deployment with FastAPI
- Built a REST API using **FastAPI**
- `/predict` endpoint accepts phone specs as JSON, runs the same `ct.transform()` + `model.predict()` pipeline, returns predicted price
- `/processor-data` endpoint serves processor lookup from the real dataset
- API serves the frontend `index.html` at the `/` route

### 8. 🌐 Frontend (HTML/CSS/JS)
- Built a clean, dark-themed UI in plain HTML, CSS and JavaScript
- 2-level processor drill-down: pick brand → pick model → Core & Speed auto-fill
- Toggle switches for boolean features (5G, NFC, IR Blaster etc.)
- Price displayed in **PKR** (converted from INR using exchange rate)
- Sends form data to FastAPI backend via `fetch()` API call

---

## 🗂️ Project Structure

```
phone_predictor/
├── main.py              ← FastAPI backend (API + serves frontend)
├── index.html           ← Frontend UI (HTML/CSS/JS)
├── requirements.txt     ← Python dependencies
├── data.csv             ← Cleaned dataset
├── model/
│   ├── model.pkl        ← Trained XGBoost model
│   └── ct.pkl           ← Fitted ColumnTransformer
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
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

- How to build a **full end-to-end ML pipeline** from raw data to live prediction
- Importance of saving the **ColumnTransformer** alongside the model — without it, predictions at deployment will be wrong
- How **FastAPI** makes it easy to serve ML models as REST APIs
- How to connect a plain HTML frontend to a Python backend using `fetch()`
- Real-world data is messy — data cleaning took more time than model training
