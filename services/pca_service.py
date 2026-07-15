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

import pca_clickhouse as ch
from pca_clickhouse import CallRecord, CallAnalytics
from services import validation_service

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
        role = "Customer" if msg.get("role") == "user" else "Customer Support"
        lines.append(f"{role}: {msg.get('text', '')}")
    return "\n".join(lines)


def _messages_to_turns(messages, started_at=None):
    """Convert messages to frontend transcript format"""
    turns = []
    for msg in messages or []:
        role = msg.get("role")
        speaker_type = "user" if role == "user" else "agent"
        
        # Use the pre-formatted timestamp if available (from transcribe_service)
        # Otherwise fall back to calculating from start_time
        if 'timestamp' in msg and isinstance(msg['timestamp'], str) and ':' in msg['timestamp']:
            stamp = msg['timestamp']
        elif 'start_time' in msg:
            # Calculate from start_time in seconds
            start_seconds = float(msg['start_time'])
            minutes = int(start_seconds // 60)
            seconds = int(start_seconds % 60)
            stamp = f"{minutes:02d}:{seconds:02d}"
        else:
            stamp = "00:00"
        turns.append({
            "speaker": "Customer" if speaker_type == "user" else "Customer Support",
            "timestamp": stamp,
            "content": msg.get("text", ""),
            "type": speaker_type,
        })
    return turns


_ANALYSIS_SYSTEM_PROMPT = """You are a call-quality analyst for a customer-support voice platform.
You receive a full transcript of a completed call between a Customer Support Agent and a Customer.
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
  "avg_wait_time": <total seconds customer was on hold/waiting>,
  "sla_compliance": <percentage 0-100 based on: answer within 30 seconds (25%), issue resolution within 5 minutes (50%), first call resolution (25%)>,
  "abandonment_rate": <0 if call was answered, 100 if abandoned>,
  "learning_suggestions": "<coaching suggestion for the agent on how to improve this interaction>",
  "competitor_intelligence": [
    {
      "competitor_name": "<company name if mentioned by customer, else null>",
      "product_mentioned": "<product/feature mentioned in comparison, else null>",
      "comparison_type": "<what aspect was compared: cheaper|better_feature|quality|service|other>",
      "customer_sentiment": "<appreciation|complaint|query|suggestion|neutral>",
      "details": "<brief description of what the customer said about competitor>",
      "timestamp": "<approximate time in call when mentioned>"
    }
  ],
  "compliance_flags": [
    {
      "flag": "<flag name>",
      "severity": "<HIGH|MEDIUM|LOW>",
      "description": "<detailed description with specific context and evidence from the call>",
      "timestamp": "<approximate time in call when detected, e.g., '02:15'>",
      "evidence": "<direct quote or specific statement from transcript that triggered this flag>"
    }
  ],
  "call_matrices": {
    "issue_type": "<primary reason for the call, e.g. 'Account inquiry', 'Payment issue', 'Technical support'>",
    "resolution_status": "<Was the issue resolved? 'Resolved', 'Pending', 'Escalated', or 'Unresolved'>",
    "escalation_required": "<Does this call need follow-up? 'Yes' or 'No'>",
    "payment_mentioned": "<Was payment or billing discussed? 'Yes' or 'No'>",
    "product_service": "<Product or service discussed, if any, else 'N/A'>",
    "customer_intent": "<What did the customer want to achieve? Brief description>"
  }
}

ANALYTICS FIELDS EXPLANATION:

**avg_wait_time**: Calculate total seconds customer was on hold/waiting
- Count "on hold" occurrences in transcript
- Example: If customer on hold twice for 2 minutes each = 240 seconds
- Return: Total seconds as number

**sla_compliance**: Calculate percentage (0-100) based on:
- Answer within 30 seconds from start: 25 points
- Issue resolution/satisfactory outcome within 5 minutes: 50 points  
- First call resolution (no escalation needed): 25 points
- Total = (points earned / 100) * 100 = percentage
- Example: Answered in 15 sec (25) + resolved in 4 min (50) + no escalation (25) = 100%

**abandonment_rate**: 
- 0 = Call was answered and completed
- 100 = Call was abandoned/dropped before completion

COMPLIANCE FLAGS TO DETECT:

**HIGH SEVERITY:**
- "Abusive Language" - Profanity, offensive language, threats from either party
- "Data Security Breach" - Credit card numbers, passwords, or sensitive data spoken
- "Legal Threats" - Customer threatens lawsuit or legal action
- "Unauthorized Commitment" - Agent promises refunds/actions beyond authority
- "Privacy Violation" - Personal data of other customers mentioned

**MEDIUM SEVERITY:**
- "Compliance Violation" - Missing mandatory disclosures (recording notice, privacy policy)
- "Misinformation" - Agent provides incorrect product/policy information
- "Escalation Ignored" - Customer requests supervisor but not escalated
- "Long Hold Time" - Customer mentions being on hold for extended period
- "Unprofessional Behavior" - Agent rude, dismissive, or lacks empathy

**LOW SEVERITY:**
- "No Recording Notice" - Agent didn't inform customer about call recording
- "Incomplete Information" - Agent couldn't provide complete answer to customer query
- "Customer Satisfaction Risk" - Customer clearly dissatisfied at end of call

IMPORTANT: Only include flags that are clearly evident in the transcript. If no compliance issues detected, return empty array: "compliance_flags": []

SCORING GUIDELINES (0-10 scale, where 10 = best):

**overall_sentiment**: Rate the overall tone and emotional quality of the call
- 8-10: Positive, friendly, cooperative atmosphere throughout
- 5-7: Neutral or mixed emotions, some tension but generally professional
- 2-4: Negative tone, frustration, complaints, or dissatisfaction evident
- 0-1: Highly negative, angry, hostile interaction

**customer_satisfaction**: Rate how satisfied the customer appears to be
- 8-10: Customer expresses clear satisfaction, thanks agent, issue fully resolved
- 5-7: Customer accepts the outcome, no strong positive/negative signals
- 2-4: Customer expresses frustration, dissatisfaction, or unresolved concerns
- 0-1: Customer is very unhappy, threatens to escalate, or explicitly dissatisfied

**agent_performance**: Rate the agent's effectiveness and professionalism
- 8-10: Excellent communication, empathy, problem-solving, professional throughout
- 5-7: Adequate performance, handles basic tasks but may lack polish or efficiency
- 2-4: Poor communication, lacks empathy, struggles to help, unprofessional moments
- 0-1: Very poor performance, rude, unable to assist, or major errors

Be realistic with scores. A typical successful call should score 6-8, not perfect 10s.
Base all ratings strictly on evidence from the transcript provided.

LEARNING & DEVELOPMENT SUGGESTIONS:
Analyze the agents performance and suggest ONE specific improvement:
- Focus on areas where agent could handle better next time
- Provide actionable coaching (not generic praise)
- Example: "The customer raised a concern at 03:40 about delivery time. Instead of saying 'that's our standard', you could have acknowledged their urgency and offered alternative solutions like expedited shipping or specific delivery date confirmation."
- Keep suggestion concise (2-3 sentences max)
- Only suggest improvements, not praise

COMPETITOR INTELLIGENCE EXTRACTION:
When customer mentions any competitor company or compares Wakefit with another brand:
- Extract competitor company name exactly as stated
- Extract specific product/feature mentioned (e.g., "memory foam", "warranty", "price")
- Classify comparison type: cheaper, better_feature, quality, service, or other
- Classify customer sentiment: appreciation (customer liked competitor), complaint (customer dissatisfied with competitor), query (customer asking questions), suggestion (customer suggesting Wakefit adopt something), neutral (factual comparison)
- Include exact details from customer's statement
- Include timestamp of mention
- Return empty array if NO competitor mentions: "competitor_intelligence": []

COMPLIANCE FLAGS TO DETECT:

**HIGH SEVERITY:**
- "Abusive Language" - Profanity, offensive language, threats from either party
- "Data Security Breach" - Credit card numbers, passwords, or sensitive data spoken
- "Legal Threats" - Customer threatens lawsuit or legal action
- "Unauthorized Commitment" - Agent promises refunds/actions beyond authority
- "Privacy Violation" - Personal data of other customers mentioned

**MEDIUM SEVERITY:**
- "Compliance Violation" - Missing mandatory disclosures (recording notice, privacy policy)
- "Misinformation" - Agent provides incorrect product/policy information
- "Escalation Ignored" - Customer requests supervisor but not escalated
- "Long Hold Time" - Customer mentions being on hold for extended period
- "Unprofessional Behavior" - Agent rude, dismissive, or lacks empathy

**LOW SEVERITY:**
- "No Recording Notice" - Agent didn't inform customer about call recording
- "Incomplete Information" - Agent couldn't provide complete answer to customer query
- "Customer Satisfaction Risk" - Customer clearly dissatisfied at end of call

IMPORTANT: Only include flags that are clearly evident in the transcript. If no compliance issues detected, return empty array: "compliance_flags": []

SCORING GUIDELINES (0-10 scale, where 10 = best):

**overall_sentiment**: Rate the overall tone and emotional quality of the call
- 8-10: Positive, friendly, cooperative atmosphere throughout
- 5-7: Neutral or mixed emotions, some tension but generally professional
- 2-4: Negative tone, frustration, complaints, or dissatisfaction evident
- 0-1: Highly negative, angry, hostile interaction

**customer_satisfaction**: Rate how satisfied the customer appears to be
- 8-10: Customer expresses clear satisfaction, thanks agent, issue fully resolved
- 5-7: Customer accepts the outcome, no strong positive/negative signals
- 2-4: Customer expresses frustration, dissatisfaction, or unresolved concerns
- 0-1: Customer is very unhappy, threatens to escalate, or explicitly dissatisfied

**agent_performance**: Rate the agent's effectiveness and professionalism
- 8-10: Excellent communication, empathy, problem-solving, professional throughout
- 5-7: Adequate performance, handles basic tasks but may lack polish or efficiency
- 2-4: Poor communication, lacks empathy, struggles to help, unprofessional moments
- 0-1: Very poor performance, rude, unable to assist, or major errors

Be realistic with scores. A typical successful call should score 6-8, not perfect 10s.
Base all ratings strictly on evidence from the transcript provided."""


def _invoke_bedrock_analysis(conversation_text):
    """Call Bedrock for AI analysis"""
    try:
        _, bedrock_client = _get_aws_clients()
        
        resp = bedrock_client.converse(
            modelId=PCA_MODEL_ID,
            system=[{"text": _ANALYSIS_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": f"Call transcript:\n\n{conversation_text}"}]}],
            inferenceConfig={"maxTokens": 4096},
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


def analyze_call(call_id, force=False, end_reason=None, run_validation=True):
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

    # Run validation analysis (Phase 2)
    if run_validation and conversation and conversation != "(No conversation)":
        try:
            print(f"[PCA] Running validation for {call_id}")
            validation_results = validation_service.validate_call_transcript(conversation)
            
            if validation_results and "validation" in validation_results:
                val_data = validation_results["validation"]
                analytics.validation_results = validation_results
                analytics.validation_score = val_data.get("total_earned_score")
                analytics.validation_percentage = val_data.get("percentage")
                analytics.skill_level = val_data.get("skill_level")
                print(f"[PCA] Validation complete: {analytics.skill_level} ({analytics.validation_percentage}%)")
        except Exception as e:
            print(f"[PCA] Validation failed for {call_id}: {e}")
            import traceback
            traceback.print_exc()

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
    
    # Prefer explicit duration_seconds from payload, otherwise calculate from timestamps
    duration = payload.get("duration_seconds")
    if duration is None and started and ended:
        duration = max(0, int((ended - started).total_seconds()))
    duration = duration or 0

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
    record.duration_seconds = duration
    record.transcript_s3_key = payload.get("transcript_key") or record.transcript_s3_key or ''
    record.recording_s3_key = payload.get("recording_key") or record.recording_s3_key or ''
    record.audio_size = payload.get("audio_size") or record.audio_size
    record.uploaded_filename = payload.get("uploaded_filename") or record.uploaded_filename or ''
    record.notes = payload.get("notes") or record.notes or ''

    ch.upsert_record(record)

    analytics = None
    if (record.status or "").lower() == "answered":
        analytics = analyze_call(call_id, force=True, end_reason=payload.get("end_reason"))
    
    return record, analytics


def get_call_details(call_id):
    """Get full call details with analytics"""
    from datetime import timedelta, timezone
    
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
    
    # Convert times to IST (UTC+5:30)
    def to_ist(dt):
        if not dt:
            return None
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = dt.replace(tzinfo=timezone.utc)
        ist_offset = timedelta(hours=5, minutes=30)
        ist_dt = dt.astimezone(timezone.utc) + ist_offset
        return ist_dt.strftime("%d/%m/%Y, %H:%M:%S")
    
    # Convert audio size to MB
    audio_size_mb = None
    if record.audio_size:
        audio_size_mb = round(record.audio_size / (1024 * 1024), 2)

    data = {
        "customerName": (analytics.customer_name if analytics else None) or record.from_phone or "Unknown",
        "phoneNumber": record.from_phone or "—",
        "uploadedFile": record.uploaded_filename or "—",
        "uploadedAt": to_ist(record.created_on),
        "audioSize": audio_size_mb,
        "notes": record.notes or "",
        "hangupReason": (analytics.hangup_reason if analytics else None) or "—",
        "language": record.language or "—",
        "callDuration": record._format_duration(),
        "status": record.status,
        "callStart": to_ist(record.started_at) or "—",
        "callEnd": to_ist(record.ended_at) or "—",
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
        
        # Add compliance flags if available
        if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
            compliance_flags = analytics.raw_model_response.get("compliance_flags", [])
            data["complianceFlags"] = compliance_flags if compliance_flags else []
        else:
            data["complianceFlags"] = []
        
        # Add learning suggestions if available
        if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
            learning_suggestions = analytics.raw_model_response.get("learning_suggestions")
            data["learningSuggestions"] = learning_suggestions if learning_suggestions else ""
        else:
            data["learningSuggestions"] = ""
        
        # Add competitor intelligence if available
        if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
            competitor_intelligence = analytics.raw_model_response.get("competitor_intelligence", [])
            data["competitorIntelligence"] = competitor_intelligence if competitor_intelligence else []
        else:
            data["competitorIntelligence"] = []
        
        # Add validation data if available (Phase 2)
        if analytics.validation_results:
            data["validation"] = analytics.validation_results
            data["validationScore"] = float(analytics.validation_score) if analytics.validation_score else 0
            data["validationPercentage"] = float(analytics.validation_percentage) if analytics.validation_percentage else 0
            data["skillLevel"] = analytics.skill_level or "Novice"
    else:
        data["complianceFlags"] = []
    
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


def delete_call(call_id):
    """Delete call record, analytics, and S3 files"""
    try:
        # Get record first to get S3 keys
        record = ch.get_record(call_id)
        if not record:
            print(f"[PCA] Call {call_id} not found")
            return False
        
        # Delete from S3
        s3_client, _ = _get_aws_clients()
        s3_keys_to_delete = []
        
        if record.transcript_s3_key:
            s3_keys_to_delete.append({'Key': record.transcript_s3_key})
        
        if record.recording_s3_key:
            s3_keys_to_delete.append({'Key': record.recording_s3_key})
        
        if s3_keys_to_delete:
            try:
                s3_client.delete_objects(
                    Bucket=RECORDINGS_BUCKET,
                    Delete={'Objects': s3_keys_to_delete}
                )
                print(f"[PCA] Deleted {len(s3_keys_to_delete)} S3 objects for {call_id}")
            except Exception as e:
                print(f"[PCA] Failed to delete S3 objects for {call_id}: {e}")
        
        # Delete from ClickHouse
        success = ch.delete_call(call_id)
        
        if success:
            print(f"[PCA] Successfully deleted call {call_id}")
        
        return success
        
    except Exception as e:
        print(f"[PCA] delete_call failed for {call_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
