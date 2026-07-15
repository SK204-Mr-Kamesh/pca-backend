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
    """Format messages for LLM analysis with precise timestamps"""
    if not messages:
        return "(No conversation)"
    
    lines = []
    for msg in messages:
        role = "Customer" if msg.get("role") == "user" else "Customer Support"
        
        # Include timestamp for accurate coaching suggestions
        timestamp = "00:00"
        if 'timestamp' in msg and isinstance(msg['timestamp'], str) and ':' in msg['timestamp']:
            timestamp = msg['timestamp']
        elif 'start_time' in msg:
            # Calculate from start_time in seconds
            start_seconds = float(msg['start_time'])
            minutes = int(start_seconds // 60)
            seconds = int(start_seconds % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
        
        lines.append(f"[{timestamp}] {role}: {msg.get('text', '')}")
    
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


_ANALYSIS_SYSTEM_PROMPT = """You are an expert call quality analyst for Wakefit customer support operations.

ROLE CLARIFICATION:
- "Customer Support" = Wakefit agent (the employee being evaluated)  
- "Customer" = Person calling for help (the caller)

TASK: Analyze the provided transcript and return ONLY a valid JSON object (no markdown, no commentary).

🚨 CRITICAL: TIMESTAMP ACCURACY REQUIREMENT 🚨
The transcript includes timestamps in [MM:SS] format. When referencing any timestamps in your analysis:
- Use ONLY the EXACT timestamps shown in the transcript
- DO NOT guess, approximate, or create new timestamps
- Format all timestamp references as [MM:SS] exactly as shown in transcript
- This applies to: key_indicators, learning_suggestions, compliance_flags, coaching_priorities

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED OUTPUT SCHEMA (18+ metrics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "overall_sentiment": <number 0-10>,
  "customer_satisfaction": <number 0-10>,  
  "agent_performance": <number 0-10>,
  "summary": "<3-5 sentence call summary>",
  "topics": ["<topic1>", "<topic2>", ...],
  "action_items": ["<action1>", "<action2>", ...],
  "key_indicators": ["<evidence1>", "<evidence2>", ...],
  "customer_name": "<name or null>",
  "hangup_reason": "<reason>",
  "avg_wait_time": <seconds customer was on hold>,
  "sla_compliance": <percentage 0-100>,
  "abandonment_rate": <0 or 100>,
  "learning_suggestions": "<coaching advice>",
  "competitor_intelligence": [<competitor mentions>],
  "compliance_flags": [<policy violations>],
  "call_matrices": {<call metrics>},
  "coaching_priorities": [<insights with scores>]
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETAILED FIELD INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. overall_sentiment (0-10 scale)
   Guidelines:
   - 9-10: Excellent interaction, customer very happy, resolved quickly
   - 7-8: Positive interaction, customer satisfied with outcome
   - 5-6: Neutral, business-like, no strong emotions
   - 3-4: Negative, customer frustrated or issue unresolved  
   - 1-2: Very negative, customer angry or threatening escalation
   
   Base on: Customer tone, language used, satisfaction with resolution

2. customer_satisfaction (0-10 scale)
   Guidelines:
   - 9-10: Customer explicitly expresses satisfaction ("thank you so much", "perfect", "excellent service")
   - 7-8: Customer accepts solution, shows appreciation ("okay thanks", "that helps")
   - 5-6: Customer accepts but neutral ("fine", "okay", no emotion)
   - 3-4: Customer expresses dissatisfaction ("this is not acceptable", repeats concern)
   - 1-2: Customer very upset ("I want manager", threatens escalation, hangs up angry)
   
   Evidence Required: Quote exact phrase showing satisfaction/dissatisfaction with timestamp

3. agent_performance (0-10 scale) 
   Guidelines:
   - 9-10: Professional, empathetic, solves quickly, perfect communication, proactive
   - 7-8: Good performance with minor areas for improvement
   - 5-6: Adequate but lacks polish, missed empathy opportunities
   - 3-4: Poor communication, lacks knowledge, unprofessional moments
   - 1-2: Unacceptable (rude, misinformation, policy violation, escalation needed)
   
   Evidence Required: 2-3 specific examples from transcript with timestamps

4. summary (string, exactly 3-5 sentences)
   Format:
   - Sentence 1: Reason for call (what customer wanted)
   - Sentence 2: Main issue or concern discussed  
   - Sentence 3-4: Actions taken by agent
   - Sentence 5: Resolution status and customer response
   
   Example:
   "Customer called regarding delayed delivery of Order #12345 placed on Jan 15. Customer expected delivery by Jan 20 but had not received the mattress. Agent checked system and found shipment delayed due to logistics issue in customer's area. Agent initiated priority delivery for next day and offered 10% refund as compensation. Customer accepted the solution and thanked the agent."

5. topics (array of 3-5 strings max)
   CRITICAL: Use ONLY standardized topic names from this taxonomy to ensure consistent analytics
   
   📋 STANDARD WAKEFIT TOPIC TAXONOMY:
   
   🚚 DELIVERY ISSUES:
   - "Delivery Delay" (for any late/delayed deliveries)
   - "Delivery Damaged" (for damaged products during delivery)  
   - "Delivery Wrong Address" (for address/location issues)
   - "Delivery Missed" (for missed delivery attempts)
   - "Delivery Rescheduling" (for delivery date changes)
   
   💰 PAYMENT & REFUNDS:
   - "Refund Request" (for any refund requests)
   - "Payment Issue" (for payment failures/problems)
   - "Billing Query" (for billing questions)
   - "Price Match Request" (for price matching)
   
   📦 PRODUCT ISSUES:
   - "Product Quality Issue" (for defects, comfort, quality complaints)
   - "Product Size Issue" (for wrong size/dimensions)
   - "Product Exchange" (for product exchanges)
   - "Product Information" (for product details/specifications)
   - "Assembly Issue" (for setup/installation problems)
   
   📞 ORDER MANAGEMENT:
   - "Order Status Inquiry" (for order tracking/status)
   - "Order Cancellation" (for order cancellations)  
   - "Order Modification" (for changes to existing orders)
   - "New Order Placement" (for placing new orders)
   
   🔧 TECHNICAL SUPPORT:
   - "Website Issue" (for website/app problems)
   - "Account Access" (for login/account issues)
   - "Warranty Claim" (for warranty-related requests)
   - "Installation Support" (for setup assistance)
   
   📋 GENERAL:
   - "General Inquiry" (for general questions)
   - "Complaint Escalation" (for escalated complaints)
   - "Feedback/Review" (for customer feedback)
   
   RULES:
   - Use EXACT names from taxonomy above
   - Match the customer's issue to the closest standard topic
   - Include Order ID when mentioned (e.g., "Order Status Inquiry - #12345")
   - Maximum 3-5 topics per call
   - If issue doesn't fit taxonomy, use closest match
   
   Examples:
   ✅ Good: ["Delivery Delay - #12345", "Refund Request", "Product Quality Issue"]
   ❌ Bad: ["Late Delivery", "Money Back", "Defective Product"]

6. action_items (array of 2-4 strings)
   Guidelines:
   - List concrete follow-up actions with timeline
   - Specify who is responsible
   - Include deadlines if mentioned
   
   Example:
   [
     "Agent to process 10% refund within 24 hours", 
     "Customer to receive replacement mattress by Feb 5",
     "Quality team to investigate reported defect in Batch #ABC123"
   ]

7. key_indicators (array of 3-4 strings)
   Guidelines:
   - Each indicator = evidence-based sentiment observation
   - Include EXACT timestamp from transcript using [MM:SS] format
   - Focus on emotional tone shifts in conversation
   - CRITICAL: Use only the timestamps shown in transcript - DO NOT guess or approximate
   
   Example:
   [
     "Customer expressed frustration at [01:25] about repeated follow-ups needed",
     "Agent showed genuine empathy at [02:10] with sincere apology", 
     "Customer tone improved significantly at [03:40] after refund offer",
     "Call ended positively at [04:20] with customer thanking agent multiple times"
   ]

8. customer_name (string or null)
   - Extract full name if clearly mentioned in transcript
   - If only first name mentioned, use first name
   - If no name mentioned, return null

9. hangup_reason (string, one concise phrase)
   Examples: "Issue resolved", "Escalated to supervisor", "Customer hung up", "Call completed", "Technical difficulties"

10. avg_wait_time (number, seconds)
    Calculate total seconds customer was on hold:
    - Count "please hold", "one moment", "let me check" occurrences  
    - Estimate hold duration based on context
    - Sum all hold periods in the call
    - Return total seconds as number
    
    Example: If customer on hold twice (2 minutes + 1 minute) = 180 seconds

11. sla_compliance (number, percentage 0-100)
    Calculate based on 3 criteria:
    - Answer within 30 seconds from call start: 25 points
    - Issue resolution within 5 minutes: 50 points
    - First call resolution (no escalation needed): 25 points
    - Formula: (points earned / 100) * 100
    
    Example: Answered in 15 sec (25) + resolved in 4 min (50) + no escalation (25) = 100%

12. abandonment_rate (number, 0 or 100)
    - 0 = Call was answered and completed normally
    - 100 = Call was abandoned/dropped before completion

13. learning_suggestions (string, 2-3 sentences)
    Provide ONE specific, actionable coaching suggestion:
    - Focus on biggest improvement opportunity 
    - Reference EXACT timestamp from transcript (use [MM:SS] format provided)
    - Suggest alternative approach
    - CRITICAL: Use only the timestamps shown in transcript - DO NOT guess or approximate
    
    Example: "At [03:40] when customer expressed concern about delivery time, instead of saying 'that's our standard policy', you could have acknowledged their urgency first and then offered alternative solutions like expedited shipping or specific delivery date confirmation."

14. competitor_intelligence (array of objects or empty array)
    When customer mentions competitor brands, extract:
    [
      {
        "competitor_name": "<exact company name mentioned>",
        "product_mentioned": "<specific product/feature>", 
        "comparison_type": "cheaper|better_feature|quality|service|other",
        "customer_sentiment": "appreciation|complaint|query|suggestion|neutral",
        "details": "<what customer said about competitor>",
        "timestamp": "<MM:SS when mentioned>"
      }
    ]
    If NO competitors mentioned: return []

15. compliance_flags (array of objects or empty array)
    Detect policy violations with precise evidence:
    [
      {
        "flag": "<violation type>",
        "severity": "HIGH|MEDIUM|LOW", 
        "description": "<detailed description>",
        "timestamp": "[MM:SS]",
        "evidence": "<exact quote from transcript>"
      }
    ]
    
    CRITICAL: Use EXACT timestamps from transcript in [MM:SS] format - DO NOT guess or approximate
    
    HIGH SEVERITY: Abusive language, data security breach, legal threats, unauthorized commitments
    MEDIUM SEVERITY: Missing disclosures, misinformation, ignored escalations, unprofessional behavior  
    LOW SEVERITY: No recording notice, incomplete information, customer satisfaction risk
    
    If NO violations detected: return []

16. call_matrices (object)
    {
      "issue_type": "<primary reason for call>",
      "resolution_status": "Resolved|Pending|Escalated|Unresolved", 
      "escalation_required": "Yes|No",
      "payment_mentioned": "Yes|No",
      "product_service": "<specific product discussed>",
      "customer_intent": "<what customer wanted to achieve>"
    }

17. coaching_priorities (array of 3-5 objects)
    Generate coaching insights with scores:
    [
      {
        "priority": "<specific skill to improve>",
        "score": <current performance 0-10>,
        "evidence": "<what happened in call with EXACT timestamp [MM:SS]>",
        "suggestion": "<how to improve>"
      }
    ]
    
    CRITICAL: Include EXACT timestamps from transcript in evidence field using [MM:SS] format
    Focus on: greeting quality, listening skills, empathy, product knowledge, closing technique

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Transcript:
"[00:00] Customer Support: Good morning, thank you for calling Wakefit. How can I help you today?
[00:15] Customer: Hi, I ordered a mattress last week, order number 12345, but it hasn't arrived yet.
[00:35] Customer Support: I understand your concern. Let me check your order status. Please hold for a moment.
[00:50] Customer: Okay, but I really need it by tomorrow for my new apartment.
[01:20] Customer Support: I can see your order here. There's been a delay due to logistics issues in your area. I sincerely apologize for this inconvenience. 
[01:45] Customer: This is really frustrating. I specifically chose Wakefit because you promised quick delivery.
[02:05] Customer Support: I completely understand your frustration, and I want to make this right. I can arrange priority delivery for tomorrow and also process a 10% refund for the delay.
[02:35] Customer: That would be great, thank you. Will I get confirmation?
[02:45] Customer Support: Absolutely. You'll receive SMS and email confirmation within an hour. Is there anything else I can help you with?
[03:00] Customer: No, that covers it. Thanks for sorting this out.
[03:10] Customer Support: Thank you for your patience. Have a wonderful day!"

Expected Output:
{
  "overall_sentiment": 7,
  "customer_satisfaction": 8, 
  "agent_performance": 8,
  "summary": "Customer called about delayed delivery of Order #12345 placed last week. Customer needed mattress by next day for new apartment but order was delayed due to logistics issues. Agent apologized, arranged priority delivery for next day, and offered 10% refund as compensation. Customer accepted the solution and expressed satisfaction with the resolution.",
  "topics": ["Delivery Delay - #12345", "Refund Request"],
  "action_items": ["Priority delivery arranged for tomorrow", "10% refund to be processed", "SMS and email confirmation to be sent within 1 hour"],
  "key_indicators": ["Customer expressed frustration at [01:45] about delivery promise", "Agent showed empathy at [02:05] with sincere apology", "Customer satisfaction increased at [02:35] after compensation offer", "Call ended positively at [03:10] with customer thanking agent"],
  "customer_name": null,
  "hangup_reason": "Issue resolved satisfactorily",
  "avg_wait_time": 30,
  "sla_compliance": 100,
  "abandonment_rate": 0,
  "learning_suggestions": "Excellent handling of customer frustration. Consider proactively checking delivery status before customer calls to prevent such situations.",
  "competitor_intelligence": [],
  "compliance_flags": [],
  "call_matrices": {
    "issue_type": "Delivery Delay",
    "resolution_status": "Resolved",
    "escalation_required": "No", 
    "payment_mentioned": "Yes",
    "product_service": "Mattress Order #12345",
    "customer_intent": "Get mattress delivered on time"
  },
  "coaching_priorities": [
    {"priority": "Proactive Communication", "score": 7, "evidence": "Could have proactively updated customer about delay", "suggestion": "Implement automated delay notifications"},
    {"priority": "Empathy Expression", "score": 9, "evidence": "Excellent empathy shown when customer expressed frustration", "suggestion": "Continue using empathetic language"},
    {"priority": "Solution Offering", "score": 8, "evidence": "Good compensation package offered", "suggestion": "Consider offering options to customer"}
  ]
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL REMINDERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Return ONLY valid JSON (no markdown blocks, no commentary)
2. All timestamps in MM:SS format  
3. Base all scores on explicit evidence from transcript
4. Keep evidence statements concise (1-2 sentences max)
5. Use realistic scores (avoid perfect 10s unless exceptional)
6. Include all required fields, use null for missing data
7. Ensure arrays are properly formatted with quotes
8. Double-check JSON syntax before responding"""


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
    """Get full call details with analytics and agent performance metrics"""
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
        
        # Add agent performance metrics from validation results
        if analytics.validation_results:
            data["agentPerformanceMetrics"] = _extract_agent_performance_metrics(analytics.validation_results)
        else:
            data["agentPerformanceMetrics"] = {}
        
        # Add compliance flags if available
        if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
            compliance_flags = analytics.raw_model_response.get("compliance_flags", [])
            data["complianceFlags"] = compliance_flags if compliance_flags else []
            
            # Add learning suggestions if available
            learning_suggestions = analytics.raw_model_response.get("learning_suggestions")
            data["learningSuggestions"] = learning_suggestions if learning_suggestions else ""
            
            # Add competitor intelligence if available
            competitor_intelligence = analytics.raw_model_response.get("competitor_intelligence", [])
            data["competitorIntelligence"] = competitor_intelligence if competitor_intelligence else []
            
            # Add coaching priorities if available
            coaching_priorities = analytics.raw_model_response.get("coaching_priorities", [])
            data["coachingPriorities"] = coaching_priorities if coaching_priorities else []
            
            # Add SLA compliance and hold time
            data["slaCompliance"] = analytics.raw_model_response.get("sla_compliance", 0)
            data["avgHoldTime"] = analytics.raw_model_response.get("avg_wait_time", 0)
        else:
            data["complianceFlags"] = []
            data["learningSuggestions"] = ""
            data["competitorIntelligence"] = []
            data["coachingPriorities"] = []
            data["slaCompliance"] = 0
            data["avgHoldTime"] = 0
        
        # Add validation data if available (Phase 2)
        if analytics.validation_results:
            data["validation"] = analytics.validation_results
            data["validationScore"] = float(analytics.validation_score) if analytics.validation_score else 0
            data["validationPercentage"] = float(analytics.validation_percentage) if analytics.validation_percentage else 0
            data["skillLevel"] = analytics.skill_level or "Novice"
    else:
        data["complianceFlags"] = []
        data["agentPerformanceMetrics"] = {}
        data["learningSuggestions"] = ""
        data["competitorIntelligence"] = []
        data["coachingPriorities"] = []
        data["slaCompliance"] = 0
        data["avgHoldTime"] = 0
    
    return data


def _extract_agent_performance_metrics(validation_results):
    """
    Extract agent performance metrics from validation results
    Maps validation scores to required 9 metrics with evidence
    """
    if not validation_results or "validation" not in validation_results:
        return {}
    
    validation = validation_results["validation"]
    
    metrics = {
        "greetings": {
            "score": validation.get("greetings", {}).get("score", 0),
            "evidence": validation.get("greetings", {}).get("evidence", "No evidence available"),
            "max": 5
        },
        "crmQueryParaphrase": {
            "score": validation.get("crm_query_paraphrase", {}).get("score", 0), 
            "evidence": validation.get("crm_query_paraphrase", {}).get("evidence", "No evidence available"),
            "max": 5
        },
        "energyAndClarity": {
            "score": validation.get("energy_enthusiasm_pace", {}).get("score", 0),
            "evidence": validation.get("energy_enthusiasm_pace", {}).get("evidence", "No evidence available"), 
            "max": 5
        },
        "activeListening": {
            "score": validation.get("listening_acknowledgment", {}).get("score", 0),
            "evidence": validation.get("listening_acknowledgment", {}).get("evidence", "No evidence available"),
            "max": 5
        },
        "grammarAndStructure": {
            "score": validation.get("grammar_vocabulary", {}).get("score", 0),
            "evidence": validation.get("grammar_vocabulary", {}).get("evidence", "No evidence available"),
            "max": 5
        },
        "empathyAndApology": {
            "score": validation.get("apology_empathy", {}).get("score", 0),
            "evidence": validation.get("apology_empathy", {}).get("evidence", "No evidence available"),
            "max": 5
        },
        "holdProcedure": {
            "score": validation.get("dead_air_hold_process", {}).get("score", 0),
            "evidence": validation.get("dead_air_hold_process", {}).get("evidence", "No evidence available"),
            "max": 6
        },
        "probingAndUnderstanding": {
            "score": validation.get("good_right_probing", {}).get("score", 0),
            "evidence": validation.get("good_right_probing", {}).get("evidence", "No evidence available"),
            "max": 12
        },
        "correctClosing": {
            "score": validation.get("correct_closing", {}).get("score", 0),
            "evidence": validation.get("correct_closing", {}).get("evidence", "No evidence available"),
            "max": 6
        }
    }
    
    return metrics


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
