import json
import time
import re
from confluent_kafka import Producer
from textblob import TextBlob
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from consumer_config import load_config  # we reuse your config loader

config = load_config()
YOUTUBE_API_KEY = config.get("youtube_api_key", "")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY) if YOUTUBE_API_KEY else None

# Kafka Producer config
producer_conf = {
    "bootstrap.servers": config["bootstrap_server"],
    "security.protocol": "SASL_SSL",
    "sasl.mechanisms": "SCRAM-SHA-512",
    "sasl.username": config["auth"]["username"],
    "sasl.password": config["auth"]["password"],
    "ssl.ca.location": config["auth"]["ca_path"],
}
producer = Producer(producer_conf)
topic = config["topic"]

def analyze_comment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "POSITIVE"
    elif score < -0.1:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def extract_video_id(url_or_id: str) -> str:
    pat = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    m = re.search(pat, url_or_id)
    return m.group(1) if m else url_or_id.strip()

def delivery_report(err, msg):
    if err:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def produce_comments(video_id, max_comments=5000):
    if youtube is None:
        print("Missing YouTube API key in config.yaml")
        return

    try:
        meta = youtube.videos().list(id=video_id, part="status,liveStreamingDetails").execute()["items"][0]
    except (IndexError, KeyError):
        print("Invalid video ID or private/unavailable video")
        return

    # For live stream, use live chat messages
    live_chat_id = meta.get("liveStreamingDetails", {}).get("activeLiveChatId")

    count = 0
    if live_chat_id:
        print("Fetching live chat messages...")
        next_page_token = None
        while count < max_comments:
            response = youtube.liveChatMessages().list(
                liveChatId=live_chat_id,
                part="snippet,authorDetails",
                maxResults=200,
                pageToken=next_page_token
            ).execute()

            for item in response.get("items", []):
                text = item["snippet"]["displayMessage"]
                sentiment = analyze_comment(text)
                msg = json.dumps({"text": text, "sentiment": sentiment})
                producer.produce(topic, msg.encode("utf-8"), callback=delivery_report)
                producer.poll(0)
                count += 1
                if count >= max_comments:
                    break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
            time.sleep(1)

    else:
        print("Fetching regular comments...")
        request = youtube.commentThreads().list(
            videoId=video_id,
            part="snippet",
            maxResults=100,
            textFormat="plainText",
        )
        while request is not None and count < max_comments:
            response = request.execute()
            for item in response.get("items", []):
                text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                sentiment = analyze_comment(text)
                msg = json.dumps({"text": text, "sentiment": sentiment})
                producer.produce(topic, msg.encode("utf-8"), callback=delivery_report)
                producer.poll(0)
                count += 1
                if count >= max_comments:
                    break
            request = youtube.commentThreads().list_next(request, response)
            time.sleep(1)

    producer.flush()
    print(f"Produced {count} messages to Kafka topic '{topic}'")

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL or ID: ").strip()
    video_id = extract_video_id(video_url)
    produce_comments(video_id)
