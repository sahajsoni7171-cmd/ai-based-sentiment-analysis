import streamlit as st
import string
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# Download stopwords (runs once)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive 😊"
    elif polarity < 0:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# ---------------- UI ----------------
st.title("AI-Based Sentiment Analysis System")

user_text = st.text_area("Enter a review or sentence:")

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean = clean_text(user_text)
        sentiment = predict_sentiment(clean)

        st.subheader("Predicted Sentiment:")
        st.success(sentiment)

        # Word Cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(clean)

        st.subheader("Word Cloud")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)

import os
port = int(os.environ.get("PORT", 8501))

if __name__ == "__main__":
    import subprocess
    subprocess.run([
        "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ])