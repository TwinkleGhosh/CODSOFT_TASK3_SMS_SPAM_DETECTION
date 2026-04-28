from train import load_models
from preprocessing import full_pipeline


def predict(text: str, vectorizer, nb_model, lr_model) -> dict:

    clean = full_pipeline(text)
    tfidf = vectorizer.transform([clean])

    nb_pred = nb_model.predict(tfidf)[0]
    lr_pred = lr_model.predict(tfidf)[0]
    final = 1 if (nb_pred + lr_pred) >= 1 else 0
    proba = lr_model.predict_proba(tfidf)[0][1]

    return {
        "spam_probability": round(float(proba), 4),
        "is_spam": bool(final),
        "nb_pred": int(nb_pred),
        "lr_pred": int(lr_pred),
    }


#  Adversarial test harness
ADVERSARIAL_CASES = [
    "Fr33 m0ney n0w!!!",
    "W!n c@sh pr!ze",
    "Congratulations, you won ₹5000",
    "FREE entry into contest!!!",
    "Hey bro, call me later",
]


def run_adversarial_tests(vectorizer, nb_model, lr_model):
    print("\n--- Adversarial Testing ---")
    for msg in ADVERSARIAL_CASES:
        result = predict(msg, vectorizer, nb_model, lr_model)
        label = "Spam 🚫" if result["is_spam"] else "Not Spam ✅"
        print(f"{msg!r:45s} → {label}  (p={result['spam_probability']:.2f})")


if __name__ == "__main__":
    vectorizer, nb_model, lr_model = load_models()
    run_adversarial_tests(vectorizer, nb_model, lr_model)

    print("\n--- SMS Spam Detector (CLI) ---")
    while True:
        user_input = input("\nEnter a message (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        result = predict(user_input, vectorizer, nb_model, lr_model)
        print(f"Spam Probability : {result['spam_probability']:.2f}")
        print(f"Result           : {'Spam 🚫' if result['is_spam'] else 'Not Spam ✅'}")
