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


VALIDATION_SYSTEM_PROMPT = """You are an expert call quality analyst evaluating customer support agent performance.

--- CORE EVALUATION MATRIX ---
Analyze the transcript and assign one marking for each parameter: "Expert", "Intermediate", "Novice", or "BI/CI".
Note: "Customer Support" is the Wakefit agent being evaluated. "Customer" is the person calling in.

1. Greetings (Weight: 5)
   - Did Customer Support greet with energetic tone? Right salutation?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

2. CRM Query Paraphrase (Weight: 5)
   - Did Customer Support confirm product details? Paraphrase concern without making customer repeat?
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

8. Good/Right Probing (Weight: 12)
   - Did correct and relevant probing to understand customer query? Look for negative probing.
   - Expert (12), Novice (0), BI/CI (0)

9. Correct Closing (Weight: 6)
   - Did Customer Support close call with Wakefit brand name and thank customer?
   - Expert (6), Intermediate (3), Novice (0), BI/CI (0)

--- BI/CI RULE ---
"BI/CI" = Business/Customer Impact violation (extreme rudeness, policy violation, misinformation, data exposure).
If ANY parameter is BI/CI, set `is_critical_escalation: true`.

--- SCORING ---
- Total = Sum of all scores
- Max Possible = 54 (or 48 if parameter 7 is N/A)
- Percentage = (Total / Max) * 100
- Skill Level: Expert (>=80%), Intermediate (50-79.9%), Novice (<50%)

--- EVIDENCE FORMAT ---
Keep evidence brief (1-2 sentences MAX). State what happened, not why you scored it.
Examples:
✓ Good: "Customer Support greeted at 00:23 with 'Hello, welcome to Wakefit'"
✗ Bad: "The agent demonstrated excellent greeting skills by providing a warm and professional welcome to the customer which showed good energy and enthusiasm throughout the initial interaction"

✓ Good: "No Wakefit brand name in closing at 09:45"
✗ Bad: "The agent failed to mention the brand name during the call closing phase which is a requirement for proper closing procedures according to the standards"

✓ Good: "Customer Support interrupted customer multiple times at 02:15, 03:40"
✗ Bad: "Throughout the conversation the agent interrupted the customer on several occasions which indicates poor listening skills"

Respond with ONLY valid JSON (no markdown):
{
  "validation": {
    "greetings": {"marking": "Expert|Intermediate|Novice|BI/CI", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "crm_query_paraphrase": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "energy_enthusiasm_pace": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "listening_acknowledgment": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "grammar_vocabulary": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "apology_empathy": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "dead_air_hold_process": {"marking": "...|N/A", "score": 6.0|3.0|0.0, "evidence": "Brief 1-2 sentence observation"},
    "good_right_probing": {"marking": "Expert|Novice|BI/CI", "score": 12.0|0.0, "evidence": "Brief 1-2 sentence observation"},
    "correct_closing": {"marking": "Expert|Intermediate|Novice|BI/CI", "score": 6.0|3.0|0.0, "evidence": "Brief 1-2 sentence observation"},
    "total_earned_score": <sum>,
    "max_possible_score": 54|48,
    "percentage": <calculated>,
    "skill_level": "Expert|Intermediate|Novice",
    "is_critical_escalation": true|false
  }
}

Be strict and realistic. Keep evidence concise and factual. Evaluate Customer Support agent only, not the customer."""


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
            inferenceConfig={"maxTokens": 4096},
        )
        
        output_text = response["output"]["message"]["content"][0]["text"].strip()
        results = _parse_validation_json(output_text)
        
        # Validate and ensure proper structure
        if results and "validation" in results:
            validation_data = results["validation"]
            
            # Manual calculation of totals from individual scores (don't trust LLM math)
            total_score = 0.0
            max_score = 54  # Always 54 total points
            
            score_map = {
                "greetings": 5,
                "crm_query_paraphrase": 5,
                "energy_enthusiasm_pace": 5,
                "listening_acknowledgment": 5,
                "grammar_vocabulary": 5,
                "apology_empathy": 5,
                "dead_air_hold_process": 6,
                "good_right_probing": 12,
                "correct_closing": 6
            }
            
            for param_name, max_val in score_map.items():
                if param_name in validation_data:
                    param_data = validation_data[param_name]
                    if isinstance(param_data, dict) and 'score' in param_data:
                        try:
                            score = float(param_data['score'])
                            total_score += score
                        except (TypeError, ValueError):
                            pass
            
            # Recalculate totals manually (override LLM calculations)
            percentage = (total_score / max_score) * 100
            
            # Update validation data with correct totals
            validation_data["total_earned_score"] = round(total_score, 1)
            validation_data["max_possible_score"] = max_score
            validation_data["percentage"] = round(percentage, 2)
            
            # Recalculate skill level based on correct percentage
            if percentage >= 80:
                validation_data["skill_level"] = "Expert"
            elif percentage >= 50:
                validation_data["skill_level"] = "Intermediate"
            else:
                validation_data["skill_level"] = "Novice"
            
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
            "good_right_probing": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "correct_closing": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "total_earned_score": 0,
            "max_possible_score": 48,
            "percentage": 0,
            "skill_level": "Novice",
            "is_critical_escalation": False
        }
    }




# ────────────────────────────────────────────────────────────────────────────
# IN-STORE VALIDATION SYSTEM (5 Metrics for Sales Executives)
# ────────────────────────────────────────────────────────────────────────────

INSTORE_VALIDATION_SYSTEM_PROMPT = """You are an expert in-store interaction quality analyst evaluating sales executive performance.

--- CORE EVALUATION MATRIX (5 METRICS) ---
Analyze the transcript and assign one marking for each parameter: "Expert", "Intermediate", "Novice", or "BI/CI".
Note: "Sales Executive" is the Wakefit employee being evaluated. "Customer" is the person shopping in store.

1. Communication (Weight: 5)
   - Clear, professional tone, active listening, asks clarifying questions?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

2. Discovery (Weight: 5)
   - Did Sales Executive ask the right questions to understand customer needs? Identify pain points?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

3. Solution Fit (Weight: 5)
   - Were product recommendations appropriate for the customer's needs? Explained benefits?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

4. Sales Execution (Weight: 5)
   - Did Sales Executive handle objections effectively? Ask for the sale? Define next steps?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

5. Customer Experience (Weight: 5)
   - Did customer feel valued, respected, and satisfied with the interaction?
   - Expert (5), Intermediate (2.5), Novice (0), BI/CI (0)

--- BI/CI RULE ---
"BI/CI" = Business/Customer Impact violation (extreme rudeness, policy violation, aggressive tactics, misinformation).
If ANY parameter is BI/CI, set `is_critical_escalation: true`.

--- SCORING ---
- Total = Sum of all scores
- Max Possible = 25 points
- Percentage = (Total / Max) * 100
- Skill Level: Expert (>=80%), Intermediate (50-79.9%), Novice (<50%)

--- EVIDENCE FORMAT ---
Keep evidence brief (1-2 sentences MAX). State what happened, not why you scored it.
Examples:
✓ Good: "Sales Executive asked about customer's sleep position at 02:15 before recommending mattress"
✗ Bad: "The sales executive demonstrated excellent discovery skills by asking targeted questions"

✓ Good: "Customer objected about price at 04:30, Sales Executive offered financing option"
✗ Bad: "The employee handled objections skillfully by providing alternative solutions"

Respond with ONLY valid JSON (no markdown):
{
  "validation": {
    "communication": {"marking": "Expert|Intermediate|Novice|BI/CI", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "discovery": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "solution_fit": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "sales_execution": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "customer_experience": {"marking": "...", "score": 5.0|2.5|0.0, "evidence": "Brief 1-2 sentence observation"},
    "total_earned_score": <sum>,
    "max_possible_score": 25,
    "percentage": <calculated>,
    "skill_level": "Expert|Intermediate|Novice",
    "is_critical_escalation": true|false
  }
}

Be strict and realistic. Keep evidence concise and factual. Evaluate Sales Executive only, not the customer."""


def validate_instore_transcript(conversation_text: str) -> Dict[str, Any]:
    """
    Validate an in-store interaction transcript against the 5-metric sales matrix
    
    Args:
        conversation_text: Formatted transcript (Customer: ...\nSales Executive: ...\n)
    
    Returns:
        Dictionary containing validation results with 5 sales metrics
    """
    try:
        bedrock_client = _get_bedrock_client()
        
        user_message = f"Evaluate this in-store interaction:\n\n{conversation_text}"
        
        response = bedrock_client.converse(
            modelId=PCA_MODEL_ID,
            system=[{"text": INSTORE_VALIDATION_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": 4096},
        )
        
        output_text = response["output"]["message"]["content"][0]["text"].strip()
        results = _parse_validation_json(output_text)
        
        # Validate and ensure proper structure
        if results and "validation" in results:
            return results
        
        # Fallback if AI didn't wrap in "validation" key
        if results and "communication" in results:
            return {"validation": results}
        
        return _get_empty_instore_validation()
        
    except Exception as e:
        print(f"[INSTORE-VALIDATION] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return _get_empty_instore_validation()


def _get_empty_instore_validation() -> Dict:
    """Return empty in-store validation structure for error cases"""
    return {
        "validation": {
            "communication": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "discovery": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "solution_fit": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "sales_execution": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "customer_experience": {"marking": "Novice", "score": 0.0, "evidence": "Analysis failed"},
            "total_earned_score": 0,
            "max_possible_score": 25,
            "percentage": 0,
            "skill_level": "Novice",
            "is_critical_escalation": False
        }
    }
