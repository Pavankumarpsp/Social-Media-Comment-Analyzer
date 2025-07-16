import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text