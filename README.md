# Real-Time YouTube Comment Sentiment Pipeline

A streaming pipeline that ingests YouTube comments and live-chat messages, classifies sentiment, and renders a real-time dashboard — built on **Apache Kafka (Aiven Cloud)**, **YouTube Data API v3**, **TextBlob**, and **Streamlit**.

The producer fetches messages from YouTube and publishes pre-scored JSON to a Kafka topic; the Streamlit consumer subscribes to the same topic and updates a sentiment pie chart and word cloud as new messages land — sub-second end-to-end latency.

---

## Architecture

```
                YouTube Data API v3
                       │
                       ▼
        ┌──────────────────────────────┐
        │  producer.py                 │
        │  • fetches comments / chat   │
        │  • TextBlob sentiment score  │
        │  • emits JSON                │
        └───────────────┬──────────────┘
                        │
                        ▼
        ┌──────────────────────────────┐
        │  Aiven Cloud Kafka           │
        │  SASL/SCRAM-SHA-512 over SSL │
        │  topic: youtube-comments     │
        └────────┬───────────┬─────────┘
                 │           │
        (offset:earliest)  (offset:latest)
                 │           │
                 ▼           ▼
        ┌──────────────────────────────┐
        │  Aiven.py (Streamlit)        │
        │  • historical view (1k msgs) │
        │  • live stream view          │
        │  • pie chart + word cloud    │
        └──────────────────────────────┘
```

The design is intentionally *producer-side scoring* — sentiment is attached to each message as a JSON field before it lands in Kafka, so the consumer side stays lightweight and the topic doubles as a queryable scored history.

---

## Stack

- **Language**: Python 3.9+
- **Streaming**: Apache Kafka on [Aiven Cloud](https://aiven.io/kafka) (`confluent-kafka` client, SASL/SCRAM-SHA-512 over SSL with CA-pinned certificate)
- **Data source**: YouTube Data API v3 (`google-api-python-client`)
- **NLP**: TextBlob (polarity → POSITIVE / NEUTRAL / NEGATIVE)
- **UI**: Streamlit + Matplotlib + WordCloud
- **Config**: PyYAML, with environment-variable overrides for secrets

---

## Features

- **Two ingestion modes** — standard video comments via `commentThreads`, live-stream chat via `liveChatMessages`. Auto-detected from the video's `liveStreamingDetails` field.
- **In-flight sentiment scoring** — producer attaches `{"text": ..., "sentiment": ...}` JSON before publishing, so downstream consumers don't need their own NLP stack.
- **Two dashboard views** —
  - *Historical*: replays the last 1,000 messages from the earliest offset to give a quick snapshot.
  - *Live*: subscribes from the latest offset, updates the pie chart and word cloud as new messages arrive.
- **Secure broker connection** — SASL/SCRAM-SHA-512 over SSL with a CA-pinned certificate (Aiven Cloud-style auth).

---

## Repository structure

```
.
├── producer.py            # YouTube → Kafka producer (with TextBlob scoring)
├── Aiven.py               # Streamlit dashboard (Kafka consumer)
├── consumer_config.py     # Kafka consumer factory (SASL/SSL)
├── config.yaml            # Broker + API config (do NOT commit live secrets)
├── certs/                 # Aiven CA certificate (gitignored)
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Manas731/youtube-realtime-sentiment.git
cd youtube-realtime-sentiment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m textblob.download_corpora
```

Suggested `requirements.txt` (regenerate from your own venv with `pip freeze > requirements.txt`):

```
confluent-kafka>=2.3
google-api-python-client>=2.100
textblob>=0.17
streamlit>=1.30
matplotlib>=3.7
wordcloud>=1.9
PyYAML>=6.0
```

### 2. Provision a Kafka topic on Aiven Cloud

- Sign up at [aiven.io](https://aiven.io) and spin up a Kafka service (the free trial works fine for testing).
- Create a topic called `youtube-comments`.
- From the service *Overview* page, download the **CA certificate** and save it as `certs/ca.pem`.
- Note the SASL username, SASL password, and bootstrap server hostname.

### 3. Get a YouTube Data API v3 key

- console.cloud.google.com → APIs & Services → Library → enable **YouTube Data API v3**.
- Credentials → Create credentials → API key.
- Recommended: restrict the key to the YouTube Data API only.

### 4. Configure

Copy `config.yaml` and fill in your own credentials (do **not** commit the live values):

```yaml
auth:
  method: sasl
  username: <aiven-sasl-username>
  password: <aiven-sasl-password>
  ca_path:  ./certs/ca.pem

bootstrap_server: <your-aiven-broker-hostname>:<port>
topic: youtube-comments
youtube_api_key: <your-youtube-api-key>
```

Or override at runtime via environment variables (`consumer_config.py` already reads these):

```bash
export KAFKA_USERNAME=...
export KAFKA_PASSWORD=...
export KAFKA_CA_PATH=...
```

---

## Running

In one terminal, start the producer and paste a YouTube URL when prompted:

```bash
python producer.py
```

In a second terminal, start the dashboard:

```bash
streamlit run Aiven.py
```

Open the Streamlit URL (default `http://localhost:8501`), paste a video or live-stream URL into the dashboard, and toggle either **Show historical Kafka sentiment** or **Start Kafka live stream**.

---

## Roadmap

- Pluggable sentiment backends (multilingual BERT, Azure Cognitive Services Text Analytics) selectable from the UI
- Per-author and per-language aggregation
- Persist scored messages to a downstream warehouse (Postgres / Snowflake)
- Containerize producer + dashboard with Docker Compose

---

## Security note

`config.yaml` in this repo currently has placeholder values. **Before pushing to a public GitHub repo:**

1. Make sure `config.yaml` is added to `.gitignore` (or contains only placeholders, never live credentials).
2. Add `certs/` to `.gitignore` so the CA cert doesn't get committed.
3. If a live credential has ever been committed to git history (even once), rotate it — git history is permanent and bots scrape GitHub for leaked secrets within hours.

A minimal `.gitignore`:

```
__pycache__/
*.pyc
.venv/
venv/
.env

# secrets
config.yaml
certs/

# OS / IDE
.DS_Store
.vscode/
.idea/
```

---

## License

MIT — see `LICENSE`.

## Author

**Manas Singh** · Data Engineer @ Analytics Vidhya
[LinkedIn](https://www.linkedin.com/in/manas-singh-5248b126b/) · [GitHub](https://github.com/Manas731) · [Portfolio](https://www.notion.so/Manas-Singh-Data-Engineer-164dab092cdb8011b6c0e8465c54007d)
