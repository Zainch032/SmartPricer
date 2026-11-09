# Phone Price Predictor (simplified)

Ask the user **4 things they actually know** (brand, RAM, storage, 5G). Up to 6 more are optional.

## Run
```bash
pip install -r requirements.txt
# put Clean2_Smart_Phones.csv (output of your cleaning2 notebook) in data/
python train.py --data data/Clean2_Smart_Phones.csv   # prints the ablation table, saves model/model.joblib
INR_TO_PKR=3.3 uvicorn app.main:app --reload           # open http://localhost:8000
```

## Read the ablation table first
`train.py` cross-validates the same model on 24 fields, 10 fields, 4 fields and 3 fields and compares
typical % error (MdAPE) and "share of phones priced within ±20%". If rows A–B are *much* better than F,
the processor/display fields matter on your data: add a "chipset class" question (Budget / Mid / Flagship)
rather than the 24-field form.

## Why it differs from the old version
| Old | New |
|---|---|
| 24 required fields incl. chipset, refresh rate, resolution | 4 required + 6 optional |
| `rating` as an input | removed (unknown for a user's own phone, and a proxy for price) |
| Scaler/imputer fitted on all data before the split | everything inside one Pipeline, fitted on train only |
| Predicts raw price; R² dominated by a few 100k+ phones | predicts log-price; reports typical % error |
| Single number | number + honest range (10th–90th percentile of CV error) |
| Fails or guesses when a field is blank | trained with feature dropout, so blanks are fine |
