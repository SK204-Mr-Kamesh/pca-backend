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


VALIDATION_SYSTEM_PROMPT = """You are an expert call quality analyst evaluating agent performance on customer support calls.
You will receive a call transcript and must evaluate the agent's soft skills across 7 categories.

For each parameter, assign a score:
- 5.0 = Expert level
- 2.5 = Intermediate level  
- 0.0 = Novice level

EVALUATION CATEGORIES:

1. GREETINGS: Did agent open with energetic, warm greeting within 3 seconds?
2. CRM QUERY: Did agent paraphrase/confirm customer's issue clearly?
3. ENERGY & ENTHUSIASM: Was speech clear, confident, appropriately paced?
4. ACKNOWLEDGMENT: Did agent acknowledge customer appropriately without interrupting?
5. GRAMMAR: Did agent use grammatically correct sentences?
6. APOLOGY/EMPATHY: Did agent show empathy at appropriate moments?
7. HOLD PROCESS: Did agent follow proper hold procedures (explain, seek permission, sign back)?

Respond with ONLY valid JSON (no markdown fences):
{
  "greetings": {"score": 5.0 | 2.5 | 0.0, "validation": "<observation>"},
  "crm_query": {"score": 5.0 | 2.5 | 0.0, "validation": "<observation>"},
  "energy_enthusiasm": {"score": 5.0 | 2.5 | 0.0, "validation": "<observation>"},
  "acknowledgment": {"score": 5.0 | 2.5 | 0.0, "validation": "<observation>"},
  "grammar": {"score": 5.0 | 2.5 | 0.0, "validation": "<observation>"},
  "apology_empathy": {"score": 5.0 | 2.5 | 0.0, "validation": "<observation>"},
  "hold_process": {"score": 5.0 | 2.5 | 0.0, "validation": "<observation or 'N/A' if no hold>"}
}

Be realistic and strict. Base scores strictly on transcript evidence."""


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
        Dictionary containing validation results with scores and validations
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
        validation_results = _parse_validation_json(output_text)
        
        # Calculate totals
        if validation_results:
            validation_results = _add_totals(validation_results)
        
        return validation_results
        
    except Exception as e:
        print(f"[VALIDATION] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return _get_empty_validation()


def _add_totals(results: Dict) -> Dict:
    """Calculate total scores"""
    weights = {"greetings": 5, "crm_query": 5, "energy_enthusiasm": 5, 
               "acknowledgment": 5, "grammar": 5, "apology_empathy": 6, "hold_process": 6}
    
    total = 0
    weighted = 0
    max_possible = sum(weights.values()) * 5
    
    for key, weight in weights.items():
        if key in results:
            score = results[key].get("score", 0)
            total += score
            weighted += score * weight
    
    percentage = (weighted / max_possible * 100) if max_possible > 0 else 0
    
    if percentage >= 80:
        level = "Expert"
    elif percentage >= 50:
        level = "Intermediate"
    else:
        level = "Novice"
    
    results["total_score"] = round(total, 1)
    results["weighted_score"] = round(weighted, 1)
    results["percentage"] = round(percentage, 1)
    results["skill_level"] = level
    
    return results


def _get_empty_validation() -> Dict:
    """Return empty validation structure for error cases"""
    return {
        "greetings": {"score": 0.0, "validation": "Analysis failed"},
        "crm_query": {"score": 0.0, "validation": "Analysis failed"},
        "energy_enthusiasm": {"score": 0.0, "validation": "Analysis failed"},
        "acknowledgment": {"score": 0.0, "validation": "Analysis failed"},
        "grammar": {"score": 0.0, "validation": "Analysis failed"},
        "apology_empathy": {"score": 0.0, "validation": "Analysis failed"},
        "hold_process": {"score": 0.0, "validation": "Analysis failed"},
        "total_score": 0,
        "weighted_score": 0,
        "percentage": 0,
        "skill_level": "Novice"
    }


def format_validation_for_frontend(validation_results: Dict) -> Dict[str, Any]:
    """Format validation results for frontend display"""
    categories = {
        "greetings": {"name": "Greetings", "weight": 5},
        "crm_query": {"name": "CRM Query", "weight": 5},
        "energy_enthusiasm": {"name": "Energy & Enthusiasm", "weight": 5},
        "acknowledgment": {"name": "Acknowledgment", "weight": 5},
        "grammar": {"name": "Grammar", "weight": 5},
        "apology_empathy": {"name": "Apology/Empathy", "weight": 6},
        "hold_process": {"name": "Hold Process", "weight": 6}
    }
    
    matrix = []
    for key, config in categories.items():
        if key in validation_results:
            data = validation_results[key]
            matrix.append({
                "category": config["name"],
                "weight": config["weight"],
                "score": data.get("score", 0),
                "validation": data.get("validation", "N/A")
            })
    
    return {
        "validation_matrix": matrix,
        "total_score": validation_results.get("total_score", 0),
        "weighted_score": validation_results.get("weighted_score", 0),
        "percentage": validation_results.get("percentage", 0),
        "skill_level": validation_results.get("skill_level", "Novice")
    }


def get_validation_categories_info() -> Dict:
    """Return validation categories configuration for frontend"""
    return {
        "categories": [
            {"name": "Greetings", "weight": 5},
            {"name": "CRM Query", "weight": 5},
            {"name": "Energy & Enthusiasm", "weight": 5},
            {"name": "Acknowledgment", "weight": 5},
            {"name": "Grammar", "weight": 5},
            {"name": "Apology/Empathy", "weight": 6},
            {"name": "Hold Process", "weight": 6}
        ],
        "scoring": {"expert": 5.0, "intermediate": 2.5, "novice": 0.0}
    }
