import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing import full_pipeline

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(ROOT_DIR, "data", "spam.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = (df["label"] == "spam").astype(int)
    df["text"] = df["text"].apply(full_pipeline)
    return df


def train(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # TF-IDF vectorizer with bigrams
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=5000
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model 1: Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)
    print("Naive Bayes Result:\n")
    print(classification_report(y_test, y_pred_nb))

    # Model 2: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    print("\nLogistic Regression Result:\n")
    print(classification_report(y_test, y_pred_lr))

    return vectorizer, nb_model, lr_model


def save_models(vectorizer, nb_model, lr_model, out_dir: str = MODEL_DIR):
    os.makedirs(out_dir, exist_ok=True)
    pickle.dump(vectorizer, open(f"{out_dir}/vectorizer.pkl", "wb"))
    pickle.dump(nb_model, open(f"{out_dir}/nb_model.pkl", "wb"))
    pickle.dump(lr_model, open(f"{out_dir}/lr_model.pkl", "wb"))
    print(f"\nModels saved to {out_dir}/")


def load_models(model_dir: str = MODEL_DIR):
    vectorizer = pickle.load(open(f"{model_dir}/vectorizer.pkl", "rb"))
    nb_model = pickle.load(open(f"{model_dir}/nb_model.pkl", "rb"))
    lr_model = pickle.load(open(f"{model_dir}/lr_model.pkl", "rb"))
    return vectorizer, nb_model, lr_model


if __name__ == "__main__":
    df = load_data()
    vectorizer, nb_model, lr_model = train(df)
    save_models(vectorizer, nb_model, lr_model)
