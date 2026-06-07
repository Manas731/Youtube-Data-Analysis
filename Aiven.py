import streamlit as st
import json
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from consumer_config import load_config, create_kafka_consumer
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load configuration
config = load_config()
YOUTUBE_API_KEY = config.get("youtube_api_key", "")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY) if YOUTUBE_API_KEY else None

def extract_video_id(url_or_id: str) -> str:
    pat = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    m = re.search(pat, url_or_id)
    return m.group(1) if m else url_or_id.strip()

def analyze_comment(text: str):
    score = TextBlob(text).sentiment.polarity
    category = (
        "POSITIVE" if score > 0.1 else
        "NEGATIVE" if score < -0.1 else
        "NEUTRAL"
    )
    return score, category

def generate_pie_chart(sentiment_counts, chart_container):
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    if sum(sizes) == 0:
        chart_container.write("No sentiment data available.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis("equal")
    chart_container.pyplot(fig)

def generate_word_cloud(all_text, cloud_container):
    if not all_text:
        cloud_container.write("No comments data available.")
        return
    text = " ".join(all_text)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    cloud_container.pyplot(fig)

def fetch_historical_sentiment(config, max_messages=500):
    consumer = create_kafka_consumer(config, "youtube-history-group", offset_reset="earliest")
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    words = []
    n = 0
    while n < max_messages:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        try:
            data = json.loads(msg.value().decode("utf-8"))
            cat = data.get("sentiment", "NEUTRAL").upper()
            text = data.get("text", "")
            counts[cat if cat in counts else "NEUTRAL"] += 1
            words.extend(text.split())
            n += 1
        except Exception:
            continue
    consumer.close()
    return counts, words

def fetch_live_kafka_stream(config):
    consumer = create_kafka_consumer(config, "youtube-live-group")
    live_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    live_words = []
    processed = 0
    stat_box = st.empty()
    pie_box = st.empty()
    cloud_box = st.empty()
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            data = json.loads(msg.value().decode("utf-8"))
            cat = data.get("sentiment", "NEUTRAL").upper()
            text = data.get("text", "")
            live_counts[cat if cat in live_counts else "NEUTRAL"] += 1
            live_words.extend(text.split())
            processed += 1
            stat_box.markdown(f"**Kafka messages processed:** {processed}")
            generate_pie_chart(live_counts, pie_box)
            generate_word_cloud(live_words, cloud_box)
    except KeyboardInterrupt:
        st.info("Kafka stream stopped.")
    finally:
        consumer.close()

def main():
    st.title("🎥 YouTube Sentiment Dashboard")
    url = st.text_input("🔗 Paste YouTube video / live-stream URL:")
    analysis_method = st.selectbox("Sentiment model", ["TextBlob"])
    if not url:
        st.stop()

    video_id = extract_video_id(url)
    st.write(f"Video ID detected: {video_id}")

    if st.checkbox("Show historical Kafka sentiment"):
        with st.spinner("Reading last 1,000 Kafka messages..."):
            counts, words = fetch_historical_sentiment(config, 1000)
        col1, col2 = st.columns(2)
        with col1:
            generate_pie_chart(counts, st)
        with col2:
            generate_word_cloud(words, st)

    if st.button("🚀 Start Kafka live stream"):
        fetch_live_kafka_stream(config)

if __name__ == "__main__":
    main()
