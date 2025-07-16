from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocess import clean_text  # correct local import

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(comment):
    clean = clean_text(comment)
    score = analyzer.polarity_scores(clean)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'
