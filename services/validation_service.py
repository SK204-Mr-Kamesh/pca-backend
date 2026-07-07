"""
Call Validation Service
Implements soft skills validation matrix with 7 categories
Each parameter scores: 0 (Novice), 2.5 (Intermediate), 5 (Expert)
"""
import json
import os
import boto3
from typing import Dict, List, Any


AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
PCA_MODEL_ID = os.environ.get("PCA_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")


def _get_bedrock_client():
    """Get AWS Bedrock client"""
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


VALIDATION_SYSTEM_PROMPT = """You are an expert call quality analyst evaluating agent performance.

--- CORE EVALUATION MATRIX ---
Analyze the transcript and assign one marking for each parameter: "Expert", "Intermediate", "Novice", or "BI/CI".

1. Greetings (Weight: 5)
   - Did agent greet with energetic tone? Right salutation?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

2. CRM Query Paraphrase (Weight: 5)
   - Did agent confirm product details? Paraphrase concern without making customer repeat?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

3. Energy & Enthusiasm (Weight: 5)
   - Energetic, clear speech, appropriate pace, confident (no unexplained pauses >2 sec)?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

4. Listening & Acknowledgment (Weight: 5)
   - Active listening, appropriate responses, no interruptions?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

5. Grammar (Weight: 5)
   - Grammatically correct sentences? No incomplete sentences or jargon?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

6. Apology/Empathy (Weight: 5)
   - Genuine empathy/apology at right time with appropriate tone?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

7. Dead Air/Hold Process (Weight: 6)
   - Followed hold protocols? Explained wait time, sought permission, proper sign-back, no dead air >10 sec?
   - Expert (6), Intermediate (3), Novice (0), BI/CI (0), N/A (if hold not required)

--- BI/CI RULE ---
"BI/CI" = Business/Customer Impact violation (extreme rudeness, policy violation, misinformation, data exposure).
If ANY parameter is BI/CI, set `is_critical_escalation: true`.

--- SCORING ---
- Total = Sum of all scores
- Max Possible = 36 (or 30 if parameter 7 is N/A)
- Percentage = (Total / Max) * 100
- Skill Level: Expert (>=80%), Intermediate (50-79.9%), Novice (<50%)

Respond with ONLY valid JSON (no markdown):
{
  "validation": {
    "greetings": {"marking": "Expert|Intermediate|Novice|BI/CI", "score": 5.0|2.5|0.0, "evidence": "..."},
    "crm_query_paraphrase": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "..."},
    "energy_enthusiasm_pace": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "..."},
    "listening_acknowledgment": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "..."},
    "grammar_vocabulary": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "..."},
    "apology_empathy": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "..."},
    "dead_air_hold_process": {"marking": "...|N/A", "score": 6.0|3.0|0.0, "evidence": "..."},
    "total_earned_score": <sum>,
    "max_possible_score": 36|30,
    "percentage": <calculated>,
    "skill_level": "Expert|Intermediate|Novice",
    "is_critical_escalation": true|false
  }
}

Be strict and realistic. Base all scores on transcript evidence."""


def _parse_validation_json(text: str) -> Dict:
    """Parse JSON from model output with fallback strategies"""
    if not text:
        return {}
    
    cleaned = text.strip()
    
    # Remove markdown fences if present
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
    
    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    print("[VALIDATION] Could not parse model output as JSON")
    return {}


def validate_call_transcript(conversation_text: str) -> Dict[str, Any]:
    """
    Validate a call transcript against the soft skills matrix
    
    Args:
        conversation_text: Formatted transcript (Customer: ...\nAgent: ...\n)
    
    Returns:
        Dictionary containing validation results with marking, scores, and evidence
    """
    try:
        bedrock_client = _get_bedrock_client()
        
        user_message = f"Evaluate this call:\n\n{conversation_text}"
        
        response = bedrock_client.converse(
            modelId=PCA_MODEL_ID,
            system=[{"text": VALIDATION_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": 1500, "temperature": 0.1},
        )
        
        output_text = response["output"]["message"]["content"][0]["text"].strip()
        results = _parse_validation_json(output_text)
        
        # Validate and ensure proper structure
        if results and "validation" in results:
            return results
        
        # Fallback if AI didn't wrap in "validation" key
        if results and "greetings" in results:
            return {"validation": results}
        
        return _get_empty_validation()
        
    except Exception as e:
        print(f"[VALIDATION] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return _get_empty_validation()


def _add_totals(results: Dict) -> Dict:
    """No longer needed - AI calculates totals"""
    return results


def _get_empty_validation() -> Dict:
    """Return empty validation structure for error cases"""
    return {
        "validation": {
            "greetings": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "crm_query_paraphrase": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "energy_enthusiasm_pace": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "listening_acknowledgment": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "grammar_vocabulary": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "apology_empathy": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "dead_air_hold_process": {"marking": "N/A", "score": 0.0, "evidence": "Analysis failed"},
            "total_earned_score": 0,
            "max_possible_score": 30,
            "percentage": 0,
            "skill_level": "Novice",
            "is_critical_escalation": false
        }
    }

