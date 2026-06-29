# Post-Call Analytics (PCA) Backend

Standalone Flask backend for Post-Call Analytics - extracts call transcripts, analyzes sentiment, and generates business insights using AWS services.

## Features

- 🎙️ **Audio Upload & Transcription** - AWS Transcribe for speech-to-text
- 🤖 **AI-Powered Analysis** - AWS Bedrock (Claude) for sentiment & insights
- 📊 **Call Matrices** - Auto-extract 6 business metrics per call
- 💾 **ClickHouse Storage** - Fast, scalable database
- 🔊 **Audio Playback** - Presigned S3 URLs for secure streaming
- 💬 **Gen AI Chat** - Ask questions about any call

## Quick Start

### Prerequisites

- Python 3.9+
- AWS Account (credentials configured)
- ClickHouse server access

### Installation

1. **Clone and navigate:**
```bash
cd pca-backend
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. **Run the server:**
```bash
python app.py
```

Server starts at: `http://localhost:5000`

## Environment Variables

Create a `.env` file with:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1
S3_RECORDINGS_BUCKET=your-bucket-name

# ClickHouse Configuration
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-password

# Flask Configuration
PORT=5000
SECRET_KEY=your-secret-key

# Bedrock Model
PCA_MODEL_ID=global.anthropic.claude-haiku-4-5-20251001-v1:0
```

## API Endpoints

### Upload Audio
```bash
POST /api/pca/uploads
Content-Type: multipart/form-data

Form fields:
- file: audio file (.wav, .mp3, .m4a)
- callerName: string (optional)
- language: string (optional)
- notes: string (optional)

Response:
{
  "status": "success",
  "data": {
    "callId": "call-abc123",
    "analyzed": true
  }
}
```

### List Calls
```bash
GET /api/pca/calls?page=0&limit=10

Response:
{
  "status": "success",
  "data": {
    "calls": [...],
    "total_count": 40,
    "total_pages": 4
  }
}
```

### Get Call Details
```bash
GET /api/pca/calls/{callId}

Response:
{
  "status": "success",
  "data": {
    "customerName": "John Doe",
    "sentiment": {
      "overallSentiment": 8.0,
      "customerSatisfaction": 8.0,
      "agentPerformance": 7.0,
      "keyIndicators": [...]
    },
    "transcript": [...],
    "matrices": {...},
    "recordingUrl": "https://..."
  }
}
```

### Check Processing Status
```bash
GET /api/pca/calls/{callId}/processing

Response:
{
  "status": "success",
  "data": {
    "status": "ready",
    "currentStepIndex": 4,
    "steps": [...]
  }
}
```

### Chat with AI
```bash
POST /api/pca/calls/{callId}/chat
Content-Type: application/json

{
  "question": "What was the main issue discussed?"
}

Response:
{
  "status": "success",
  "data": {
    "answer": "The customer called about..."
  }
}
```

## Architecture

```
┌──────────────┐
│   Frontend   │
│   (React)    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│         Flask API (port 5000)        │
│  ┌────────────────────────────────┐  │
│  │  Routes (pca_routes.py)        │  │
│  └───────────┬────────────────────┘  │
│              ▼                        │
│  ┌────────────────────────────────┐  │
│  │  Services                       │  │
│  │  - pca_service.py               │  │
│  │  - transcribe_service.py        │  │
│  └───────────┬────────────────────┘  │
│              ▼                        │
│  ┌────────────────────────────────┐  │
│  │  ClickHouse Integration        │  │
│  │  (clickhouse_integration.py)   │  │
│  └────────────────────────────────┘  │
└──────────────┬───────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌─────────────┐   ┌─────────────┐
│     AWS     │   │ ClickHouse  │
│  Services   │   │   Database  │
│             │   │             │
│ - S3        │   │ - Records   │
│ - Transcribe│   │ - Analytics │
│ - Bedrock   │   └─────────────┘
└─────────────┘
```

## Project Structure

```
pca-backend/
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in git)
├── .env.example               # Environment template
├── clickhouse_integration.py  # ClickHouse database layer
├── transcribe_service.py      # AWS Transcribe integration
├── routes/
│   └── pca_routes.py          # API endpoints
├── services/
│   └── pca_service.py         # Business logic & AI analysis
├── HOW_IT_WORKS.md            # Detailed architecture docs
└── README.md                  # This file
```

## ClickHouse Tables

### voice_call_records
Stores call metadata (one row per call)

### voice_call_analytics
Stores AI analysis results for each call

## Call Matrices

Each analyzed call extracts 6 business metrics:

1. **Issue Type** - Primary reason for the call
2. **Resolution Status** - Resolved/Pending/Escalated/Unresolved
3. **Escalation Required** - Yes/No
4. **Payment Mentioned** - Yes/No
5. **Product Service** - Product or service discussed
6. **Customer Intent** - What the customer wanted to achieve

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
pip install black flake8
black .
flake8 .
```

### Database Verification
The backend automatically verifies ClickHouse connection on startup.

## Troubleshooting

### ClickHouse Connection Failed
- Verify `CLICKHOUSE_HOST` and `CLICKHOUSE_PORT` in `.env`
- Check firewall/security group allows port 8123
- Test connection: `curl http://your-host:8123`

### AWS Transcribe Timeout
- Large files take 1-5 minutes to transcribe
- Check AWS Transcribe quotas/limits
- Verify IAM permissions for Transcribe service

### Bedrock Analysis Failed
- Check model ID in `.env` matches available model
- Verify IAM permissions for Bedrock service
- Check AWS region supports Bedrock

## License

MIT

## Support

For issues or questions, contact your development team.
