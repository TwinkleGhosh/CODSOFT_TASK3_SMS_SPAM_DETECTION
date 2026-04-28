import re


def preprocess(text: str) -> str:
    text = text.lower()

    # Normalize money symbols
    text = text.replace("₹", " rupees ")
    text = text.replace("rs", " rupees ")

    # Replace numbers
    text = re.sub(r"\d+", "number", text)

    # Fix common spam obfuscation tricks
    text = text.replace("0", "o")
    text = text.replace("1", "i")
    text = text.replace("@", "a")
    text = text.replace("$", "s")
    text = text.replace("3", "e")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    return text


def add_spam_flags(text: str) -> str:
    """Boost weight of known spam keywords by repeating a flag token."""
    spam_keywords = ["win", "won", "free", "prize", "cash", "offer", "money", "rupees"]

    for word in spam_keywords:
        if word in text:
            text += " spamword spamword spamword"
            break  # Only append once even if multiple keywords match

    return text


def full_pipeline(text: str) -> str:
    """Run both preprocessing steps in sequence."""
    return add_spam_flags(preprocess(text))
