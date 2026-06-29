"""
Standalone PCA Service
Simplified version for standalone PCA backend
"""
import json
import os
import sys
from datetime import datetime, timezone
import boto3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clickhouse_integration as ch
from clickhouse_integration import CallRecord, CallAnalytics

RECORDINGS_BUCKET = os.environ.get("S3_RECORDINGS_BUCKET", "sahaa-voiceai-recordings")
PCA_MODEL_ID = os.environ.get("PCA_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")


def _get_aws_clients():
    """Get AWS S3 and Bedrock clients using environment variables"""
    aws_creds = {
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "region_name": AWS_REGION,
    }
    
    s3_client = boto3.client("s3", **aws_creds)
    bedrock_client = boto3.client("bedrock-runtime", **aws_creds)
    
    return s3_client, bedrock_client


def load_transcript_from_s3(transcript_key):
    """Load transcript JSON from S3"""
    try:
        s3_client, _ = _get_aws_clients()
        obj = s3_client.get_object(Bucket=RECORDINGS_BUCKET, Key=transcript_key)
        return json.loads(obj["Body"].read())
    except Exception as e:
        print(f"[PCA] Failed to read transcript {transcript_key}: {e}")
        return None


def format_transcript(messages):
    """Format messages for LLM analysis"""
    if not messages:
        return "(No conversation)"
    lines = []
    for msg in messages:
        role = "Customer" if msg.get("role") == "user" else "Agent"
        lines.append(f"{role}: {msg.get('text', '')}")
    return "\n".join(lines)


def _messages_to_turns(messages, started_at=None):
    """Convert messages to frontend transcript format"""
    base = _parse_iso(started_at)
    turns = []
    for msg in messages or []:
        role = msg.get("role")
        speaker_type = "user" if role == "user" else "agent"
        ts = _parse_iso(msg.get("timestamp"))
        if base and ts:
            delta = max(0, int((ts - base).total_seconds()))
            stamp = f"{delta // 60:02d}:{delta % 60:02d}"
        else:
            stamp = "00:00"
        turns.append({
            "speaker": "User" if speaker_type == "user" else "Agent",
            "timestamp": stamp,
            "content": msg.get("text", ""),
            "type": speaker_type,
        })
    return turns


_ANALYSIS_SYSTEM_PROMPT = """You are a call-quality analyst for a customer-support voice platform.
You receive a full transcript of a completed call between an Agent and a Customer.
Analyse it and respond with ONLY a single valid JSON object — no prose, no markdown fences.

The JSON must have exactly these keys:
{
  "overall_sentiment": <number 0-10>,
  "customer_satisfaction": <number 0-10>,
  "agent_performance": <number 0-10>,
  "summary": "<3-5 sentence summary of the call>",
  "topics": ["<short topic>", ...],
  "action_items": ["<action item>", ...],
  "key_indicators": ["<short observation supporting the sentiment scores>", ...],
  "customer_name": "<customer name if mentioned, else null>",
  "hangup_reason": "<short reason the call ended, e.g. 'Issue resolved'>",
  "call_matrices": {
    "issue_type": "<primary reason for the call, e.g. 'Account inquiry', 'Payment issue', 'Technical support'>",
    "resolution_status": "<Was the issue resolved? 'Resolved', 'Pending', 'Escalated', or 'Unresolved'>",
    "escalation_required": "<Does this call need follow-up? 'Yes' or 'No'>",
    "payment_mentioned": "<Was payment or billing discussed? 'Yes' or 'No'>",
    "product_service": "<Product or service discussed, if any, else 'N/A'>",
    "customer_intent": "<What did the customer want to achieve? Brief description>"
  }
}

Scores are on a 0-10 scale (10 = best). Base them strictly on the transcript.
For call_matrices, extract the requested information from the conversation; use "N/A" if not mentioned."""


def _invoke_bedrock_analysis(conversation_text):
    """Call Bedrock for AI analysis"""
    try:
        _, bedrock_client = _get_aws_clients()
        
        resp = bedrock_client.converse(
            modelId=PCA_MODEL_ID,
            system=[{"text": _ANALYSIS_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": f"Call transcript:\n\n{conversation_text}"}]}],
            inferenceConfig={"maxTokens": 2000, "temperature": 0},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        return _parse_json(text)
    except Exception as e:
        print(f"[PCA] Bedrock analysis failed: {e}")
        return {}


def _parse_json(text):
    """Parse JSON from model output"""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except Exception:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except Exception:
                pass
    print("[PCA] Could not parse model output as JSON")
    return {}


def _to_score(value):
    try:
        return max(0.0, min(10.0, round(float(value), 2)))
    except (TypeError, ValueError):
        return None


def _parse_iso(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _parse_date(value, end_of_day=False):
    if not value:
        return None
    try:
        d = datetime.strptime(value, "%Y-%m-%d")
        if end_of_day:
            d = d.replace(hour=23, minute=59, second=59)
        return d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def analyze_call(call_id, force=False, end_reason=None):
    """Analyze a call and save to ClickHouse"""
    record = ch.get_record(call_id)
    if not record:
        print(f"[PCA] No call record for {call_id}")
        return None

    existing = ch.get_analytics(call_id)
    if existing and not force:
        return existing

    transcript = load_transcript_from_s3(record.transcript_s3_key) if record.transcript_s3_key else None
    messages = (transcript or {}).get("messages", [])
    conversation = format_transcript(messages)

    parsed = _invoke_bedrock_analysis(conversation)

    # Shape call_matrices into the frontend {matrix_n: {name, matrix}} structure
    raw_matrices = parsed.get("call_matrices", {}) or {}
    call_matrices = {}
    idx = 1
    for key, value in raw_matrices.items():
        if "duration" in key.lower():  # Skip duration metrics
            continue
        # Convert snake_case to Title Case for display
        display_name = key.replace("_", " ").title()
        call_matrices[f"matrix_{idx}"] = {
            "name": display_name,
            "matrix": str(value) if value else "N/A",
        }
        idx += 1

    analytics = existing or CallAnalytics(call_id=call_id)
    analytics.overall_sentiment = _to_score(parsed.get("overall_sentiment"))
    analytics.customer_satisfaction = _to_score(parsed.get("customer_satisfaction"))
    analytics.agent_performance = _to_score(parsed.get("agent_performance"))
    analytics.summary = parsed.get("summary") or ""
    analytics.topics = parsed.get("topics") or []
    analytics.action_items = parsed.get("action_items") or []
    analytics.key_indicators = parsed.get("key_indicators") or []
    analytics.customer_name = parsed.get("customer_name")
    analytics.hangup_reason = end_reason or parsed.get("hangup_reason")
    analytics.call_matrices = call_matrices
    analytics.raw_model_response = parsed or None
    analytics.model_id = PCA_MODEL_ID

    try:
        ch.upsert_analytics(analytics)
    except Exception as e:
        print(f"[PCA] Failed to persist analytics: {e}")
    
    return analytics


def ingest_call(payload):
    """Ingest call data from worker or upload"""
    call_id = payload.get("session_id") or payload.get("call_id")
    if not call_id:
        raise ValueError("session_id is required")

    started = _parse_iso(payload.get("started_at"))
    ended = _parse_iso(payload.get("ended_at"))
    duration = 0
    if started and ended:
        duration = max(0, int((ended - started).total_seconds()))

    record = ch.get_record(call_id) or CallRecord(call_id=call_id)

    record.agent_id = payload.get("agent_id") or record.agent_id or ''
    record.account_id = payload.get("account_id") or record.account_id or ''
    record.room_name = payload.get("room") or payload.get("room_name") or record.room_name or ''
    record.from_phone = payload.get("from_phone") or record.from_phone or ''
    record.to_phone = payload.get("to_phone") or record.to_phone or ''
    record.call_source = payload.get("call_source") or record.call_source or "upload"
    record.status = payload.get("status") or record.status or "answered"
    record.language = payload.get("language") or record.language or ''
    record.started_at = started or record.started_at or datetime.now()
    record.ended_at = ended or record.ended_at or datetime.now()
    record.duration_seconds = duration or record.duration_seconds or 0
    record.transcript_s3_key = payload.get("transcript_key") or record.transcript_s3_key or ''
    record.recording_s3_key = payload.get("recording_key") or record.recording_s3_key or ''

    ch.upsert_record(record)

    analytics = None
    if (record.status or "").lower() == "answered":
        analytics = analyze_call(call_id, force=True, end_reason=payload.get("end_reason"))
    
    return record, analytics


def get_call_details(call_id):
    """Get full call details with analytics"""
    record = ch.get_record(call_id)
    if not record:
        return None
    
    analytics = ch.get_analytics(call_id)
    transcript = load_transcript_from_s3(record.transcript_s3_key) if record.transcript_s3_key else None
    messages = (transcript or {}).get("messages", [])
    started = record.started_at.isoformat() if record.started_at else \
        (transcript or {}).get("started_at")

    # Presign recording URL
    recording_url = None
    if record.recording_s3_key:
        try:
            s3_client, _ = _get_aws_clients()
            recording_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": RECORDINGS_BUCKET, "Key": record.recording_s3_key},
                ExpiresIn=3600,
            )
        except Exception as e:
            print(f"[PCA] Could not presign recording: {e}")

    data = {
        "customerName": (analytics.customer_name if analytics else None) or record.from_phone or "Unknown",
        "phoneNumber": record.from_phone or record.to_phone or "—",
        "hangupReason": (analytics.hangup_reason if analytics else None) or "—",
        "language": record.language or "—",
        "callDuration": record._format_duration(),
        "status": record.status,
        "callStart": record.started_at.strftime("%d/%m/%Y, %H:%M:%S") if record.started_at else "—",
        "callEnd": record.ended_at.strftime("%d/%m/%Y, %H:%M:%S") if record.ended_at else "—",
        "transcript": _messages_to_turns(messages, started),
        "recordingUrl": recording_url,
        "chatConversation": [{
            "chat_id": "1",
            "type": "SYSTEM",
            "body": "Hi! I can help you analyse this call. What would you like to know?",
        }],
        # Frontend expects these fields at top level
        "summary": "",
        "topics": [],
        "actionItems": [],
        "matrices": {},
        # Frontend expects sentiment as nested object
        "sentiment": {
            "overallSentiment": 0,
            "customerSatisfaction": 0,
            "agentPerformance": 0,
            "keyIndicators": [],
        }
    }
    
    if analytics:
        # Merge analytics data
        data["summary"] = analytics.summary or ""
        data["topics"] = analytics.topics or []
        data["actionItems"] = analytics.action_items or []
        data["matrices"] = analytics.call_matrices or {}
        data["sentiment"] = {
            "overallSentiment": float(analytics.overall_sentiment) if analytics.overall_sentiment is not None else 0,
            "customerSatisfaction": float(analytics.customer_satisfaction) if analytics.customer_satisfaction is not None else 0,
            "agentPerformance": float(analytics.agent_performance) if analytics.agent_performance is not None else 0,
            "keyIndicators": analytics.key_indicators or [],
        }
    
    return data


def get_call_logs(agent_id=None, start_date=None, end_date=None, page=0, limit=10):
    """Get paginated call logs"""
    start = _parse_date(start_date)
    end = _parse_date(end_date, end_of_day=True)
    
    rows, total_count = ch.query_records(agent_id, start, end, page, limit)
    analytics_map = ch.get_analytics_map([r.call_id for r in rows])
    
    call_logs = [r.to_log_dict(analytics_map.get(r.call_id)) for r in rows]
    
    total_pages = (total_count + int(limit) - 1) // int(limit) if limit else 1
    return {"call_logs": call_logs, "total_count": total_count, "total_pages": total_pages}


def get_agent_analytics(agent_id=None, start_date=None, end_date=None):
    """Get aggregate analytics for agent"""
    start = _parse_date(start_date)
    end = _parse_date(end_date, end_of_day=True)
    
    records, _ = ch.query_records(agent_id, start, end)
    
    total_calls = len(records)
    answered = [r for r in records if (r.status or "").lower() == "answered"]
    durations = [r.duration_seconds or 0 for r in answered]
    
    analytics_map = ch.get_analytics_map([r.call_id for r in answered])
    sentiments = [float(a.overall_sentiment) for a in analytics_map.values() if a.overall_sentiment is not None]

    return {
        "total_calls": total_calls,
        "avg_duration": round(sum(durations) / len(durations), 1) if durations else 0,
        "avg_sentiment": round(sum(sentiments) / len(sentiments), 2) if sentiments else 0,
        "pickup_rate": round(len(answered) / total_calls * 100, 1) if total_calls else 0,
        "answered_calls": len(answered),
    }


def chat_about_call(call_id, question):
    """Answer a question about a completed call"""
    record = ch.get_record(call_id)
    if not record:
        return "I couldn't find this call."

    transcript = load_transcript_from_s3(record.transcript_s3_key) if record.transcript_s3_key else None
    conversation = format_transcript((transcript or {}).get("messages", []))
    analytics = ch.get_analytics(call_id)
    analysis_ctx = json.dumps(analytics.to_dict()) if analytics else "(no analysis available)"

    system = ("You are an assistant answering questions about a single completed support call. "
              "Use the transcript and analysis provided. Be concise and factual. "
              "If the answer isn't in the call, say so. Plain text only.")
    user_text = (f"Call transcript:\n{conversation}\n\n"
                 f"Existing analysis:\n{analysis_ctx}\n\n"
                 f"Question: {question}")
    try:
        _, bedrock_client = _get_aws_clients()
        resp = bedrock_client.converse(
            modelId=PCA_MODEL_ID,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user_text}]}],
            inferenceConfig={"maxTokens": 800, "temperature": 0.2},
        )
        return resp["output"]["message"]["content"][0]["text"].strip()
    except Exception as e:
        print(f"[PCA] chat_about_call failed: {e}")
        return "Sorry, I couldn't process that right now."
