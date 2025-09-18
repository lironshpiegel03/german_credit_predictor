# German Credit — High Amount Predictor

This project implements a complete predictive system per the assignment:
- Data cleaning & preprocessing (imputation, scaling, one-hot encoding)
- Model training with train/validation split and metrics
- Real-time predictions via a Streamlit UI
- Artifacts saved with `joblib` for reproducibility

## Dataset
We use the provided `german_credit_data.csv`. Since it lacks a target label, we define a proxy target:
**HighAmount = 1 if Credit amount > median, else 0.**

## How to run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# (Re)train the model
python src/train.py

# Run the app
streamlit run app/app.py
```

## Deploy options
- **Streamlit Cloud**: push this folder to GitHub, set the app entrypoint to `app/app.py` and add `requirements.txt`.
- **Hugging Face Spaces** (Gradio/Streamlit): create a Space, upload code, select Streamlit SDK.

## Files
- `data/german_credit_data.csv` — input data
- `src/train.py` — training pipeline
- `artifacts/` — model + metrics (after training)
- `app/app.py` — Streamlit UI
- `requirements.txt`

