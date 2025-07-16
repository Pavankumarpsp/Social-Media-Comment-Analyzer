import streamlit as st
import pandas as pd
from youtube_scraper import scrape_comments
from sentiment_utils import analyze_sentiment
from toxicity_utils import classify_toxicity
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Social Media Comment Analyzer", layout="centered")
st.title("📊 Social Media Comment Analyzer")
st.write("Analyze YouTube comments for **sentiment**, **toxicity**, and generate **word clouds**.")

# ---------------- User Input ----------------
url = st.text_input("📎 Paste YouTube Video URL")
max_comments = st.text_input("🔢 Number of comments to fetch", value="100")

# ---------------- Analyze Button ----------------
if st.button("Analyze"):

    # Validate numeric input
    if not max_comments.isdigit():
        st.error("Please enter a valid number for max comments.")
        st.stop()

    max_comments = int(max_comments)

    # ---------------- Scrape Comments ----------------
    df = scrape_comments(url, max_comments)

    if df.empty or 'comment' not in df.columns:
        st.error("No valid comments found or failed to scrape.")
        st.stop()

    st.success(f"✅ {len(df)} comments fetched!")

    # ---------------- Sentiment Analysis ----------------
    df['sentiment'] = df['comment'].apply(analyze_sentiment)

    # ---------------- Toxicity Classification ----------------
    df['toxicity'] = df['comment'].apply(classify_toxicity)

    # ---------------- Sentiment Distribution Chart ----------------
    st.subheader("📌 Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # ---------------- Word Cloud ----------------
    st.subheader("☁️ Word Cloud")
    df_cleaned = df[df['comment'].apply(lambda x: isinstance(x, str) and x.strip() != '')]
    text = ' '.join(df_cleaned['comment'].tolist())

    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.image(wordcloud.to_array())
    else:
        st.warning("Not enough valid comment text to generate a word cloud.")

    # ---------------- Toxic Comments ----------------
    st.subheader("⚠️ Sample Toxic Comments")
    toxic_comments = df[df['toxicity'].apply(lambda labels: any(label in ['toxic', 'insult', 'threat', 'obscene'] for label in labels))]

    for i, row in toxic_comments.head(5).iterrows():
        st.warning(f"💬 {row['comment']}")

    # ---------------- Download Button ----------------
    st.subheader("📁 Download CSV")
    st.download_button("Download Analyzed Comments", data=df.to_csv(index=False), file_name="analyzed_comments.csv", mime="text/csv")

    st.info("✅ Project by Pavankumar P S – AI-based Comment Sentiment & Toxicity Classifier")
