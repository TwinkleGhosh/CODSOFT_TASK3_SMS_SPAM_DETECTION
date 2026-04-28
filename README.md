# SMS Spam Detector

Ensemble SMS spam classifier (Naive Bayes + Logistic Regression) with a Streamlit web UI.

## Project Structure

```
sms_spam_detector/
├── data/
│   └── spam.csv              # Raw dataset (label, text columns)
├── models/                   # Saved model files (auto-created after training)
│   ├── vectorizer.pkl
│   ├── nb_model.pkl
│   └── lr_model.pkl
├── src/
│   ├── preprocess.py         # Text cleaning & spam-flag injection
│   ├── train.py              # Training, evaluation, model persistence
│   └── predict.py            # Inference + CLI + adversarial tests
├── app/
│   └── app.py                # Streamlit web app
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
Place `spam.csv` inside the `data/` folder. Expected columns: `label` (spam/ham), `text`.

### 3. Train the models
```bash
cd src
python train.py
```
This prints classification reports and saves the three `.pkl` files to `models/`.

### 4a. Run the Streamlit app
```bash
streamlit run app/app.py
```

### 4b. Or use the CLI
```bash
cd src
python predict.py
```

## Model Details

| Step | Detail |
|---|---|
| Vectorizer | TF-IDF, bigrams, max 5 000 features |
| Models | Multinomial Naive Bayes · Logistic Regression |
| Ensemble | Predict spam if **either** model says spam |
| Preprocessing | Lowercasing, leet-speak normalisation, keyword boosting |