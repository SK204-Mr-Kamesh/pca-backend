"""
In-Store Interaction Analysis Service
Analyzes in-store sales interactions between Sales Executive and Customer
"""
import json
import os
import boto3
from datetime import datetime, timezone

AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')
PCA_MODEL_ID = os.environ.get('PCA_MODEL_ID', 'global.anthropic.claude-haiku-4-5-20251001-v1:0')


def _get_bedrock_client():
    """Get AWS Bedrock client"""
    return boto3.client(
        'bedrock-runtime',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=AWS_REGION
    )


def format_transcript_for_instore(messages):
    """Format messages for in-store analysis"""
    if not messages:
        return "(No conversation)"
    lines = []
    for msg in messages:
        role = "Customer" if msg.get("role") == "user" else "Sales Executive"
        lines.append(f"{role}: {msg.get('text', '')}")
    return "\n".join(lines)


_INSTORE_ANALYSIS_PROMPT = """You are an in-store interaction quality analyst for Wakefit retail stores.
You receive a transcript of a sales interaction between a Sales Executive and a Customer in a physical store.
Analyze it and respond with ONLY a single valid JSON object — no prose, no markdown fences.

The JSON must have exactly these keys:
{
  "overall_sentiment": <number 0-10>,
  "customer_satisfaction": <number 0-10>,
  "sales_executive_performance": <number 0-10>,
  "summary": "<3-5 sentence summary of the interaction>",
  "topics": ["<short topic>", ...],
  "action_items": ["<action item>", ...],
  "key_indicators": ["<short observation supporting the sentiment scores>", ...],
  "customer_name": "<customer name if mentioned, else null>",
  "interaction_outcome": "<short description of how the interaction ended>",
  "products_discussed": [
    {
      "product_name": "<specific product name, e.g., 'OrthoLite Memory Foam Mattress'>",
      "category": "<product category, e.g., 'Mattress', 'Storage', 'Sofa'>",
      "sub_category": "<more specific category if mentioned>",
      "discussion_summary": "<brief summary of what was discussed about this product>",
      "customer_interest_level": "<High|Medium|Low>",
      "price_discussed": "<Yes|No>",
      "price_amount": "<price if mentioned, else null>",
      "objections_raised": ["<list of customer objections or concerns>"],
      "sales_outcome": "<purchased|interested|not_interested|deferred>",
      "outcome_reason": "<why the customer made this decision>"
    }
  ],
  "interaction_matrices": {
    "interaction_code": "<unique identifier if mentioned, e.g., 'INT-MG-00400920'>",
    "primary_category": "<main product category discussed>",
    "overall_sales_outcome": "<successful|unsuccessful|deferred_decision>",
    "l1_pillow": "<Customer|Examiner|Uncertainty (fluent concern)|The customer was in the phonecall pit|etc.>",
    "l2_pillow": "<Deferred decision making|Uncertainty (fluent concern)|etc.>",
    "l3_pillow": "<specific detail about customer concern or decision status>",
    "customer_intent": "<What did the customer want? Brief description>",
    "competitor_mentioned": "<Yes|No - did customer mention competitors?>",
    "follow_up_required": "<Yes|No - does this need follow-up?>"
  }
}

IMPORTANT NOTES ON PRODUCTS:
- If multiple products are discussed, create separate entries in the "products_discussed" array
- Each product should have its own sales outcome and reason
- Track customer interest level per product separately
- Note specific objections for each product

SCORING GUIDELINES (0-10 scale, where 10 = best):

**overall_sentiment**: Rate the overall tone and emotional quality of the interaction
- 8-10: Positive, friendly, engaged customer throughout
- 5-7: Neutral or mixed emotions, customer browsing casually
- 2-4: Negative tone, customer frustrated, uninterested, or dissatisfied
- 0-1: Highly negative, customer upset or hostile

**customer_satisfaction**: Rate how satisfied the customer appears to be
- 8-10: Customer very happy with service, made purchase or strong intent to purchase
- 5-7: Customer neutral, still deciding, no strong signals either way
- 2-4: Customer expresses concerns, doubts, or dissatisfaction
- 0-1: Customer very unhappy, leaves dissatisfied

**sales_executive_performance**: Rate the sales executive's effectiveness
- 8-10: Excellent product knowledge, engagement, customer service, closes effectively
- 5-7: Adequate performance, answers questions but lacks proactive selling
- 2-4: Poor engagement, limited product knowledge, misses opportunities
- 0-1: Very poor performance, unprofessional, pushes too hard or ignores customer

IMPORTANT NOTES:
- Focus on retail/in-store context (not phone support)
- Assess product knowledge and sales techniques
- Note any upsell or cross-sell attempts
- Identify customer hesitations or objections
- Look for closing techniques and purchase signals
- Be realistic with scores - typical interaction should score 6-8

Base all ratings strictly on evidence from the transcript provided."""


def analyze_instore_interaction(messages, interaction_id=None):
    """
    Analyze in-store interaction using Claude
    Returns analysis dict with sentiment, matrices, and insights
    """
    if not messages:
        return {}
    
    conversation_text = format_transcript_for_instore(messages)
    
    try:
        bedrock_client = _get_bedrock_client()
        
        response = bedrock_client.converse(
            modelId=PCA_MODEL_ID,
            system=[{"text": _INSTORE_ANALYSIS_PROMPT}],
            messages=[{
                "role": "user",
                "content": [{"text": f"In-store interaction transcript:\n\n{conversation_text}"}]
            }],
            inferenceConfig={"maxTokens": 4096} 
        )
        
        text = response["output"]["message"]["content"][0]["text"].strip()
        parsed = _parse_json(text)
        
        print(f"[InStore] Analysis complete for {interaction_id or 'interaction'}")
        return parsed
        
    except Exception as e:
        print(f"[InStore] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _parse_json(text):
    """Parse JSON from model output"""
    if not text:
        return {}
    
    cleaned = text.strip()
    
    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split('\n')
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = '\n'.join(lines)
    
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"[InStore] JSON parse error: {e}")
        # Try to extract JSON from text
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                extracted = cleaned[start:end + 1]
                return json.loads(extracted)
            except Exception as exp:
                print(f"[InStore] Extracted JSON parse error: {exp}")
    
    print("[InStore] Could not parse model output as JSON")
    return {}


def _to_score(value):
    """Convert value to score between 0-10"""
    try:
        return max(0.0, min(10.0, round(float(value), 2)))
    except (TypeError, ValueError):
        return None
