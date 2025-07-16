import joblib
from preprocess import clean_text  # ✅ DO NOT import anything from app.py
import os
# Load the model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models/toxic_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models/toxic_vectorizer.pkl"))

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def classify_toxicity(comment):
    cleaned = clean_text(comment)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = [LABELS[i] for i, value in enumerate(prediction) if value == 1]
    return result if result else ["clean"]
