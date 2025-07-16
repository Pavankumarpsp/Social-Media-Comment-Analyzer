import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib
import re

df = pd.read_csv("data/train.csv")
df = df[['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

def clean(text):
    text = re.sub(r"http\S+|@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return text

df['comment_text'] = df['comment_text'].astype(str).apply(clean)

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['comment_text'])
y = df.iloc[:, 1:]

model = OneVsRestClassifier(LogisticRegression())
model.fit(X, y)

joblib.dump(model, 'models/toxic_model.pkl')
joblib.dump(vectorizer, 'models/toxic_vectorizer.pkl')

print("✅ Model trained and saved.")