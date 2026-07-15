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
  "learning_suggestions": "<coaching suggestion for the sales executive on how to improve this interaction>",
  "competitor_intelligence": [
    {
      "competitor_name": "<company name if mentioned by customer, else null>",
      "product_mentioned": "<product/feature mentioned in comparison, else null>",
      "comparison_type": "<what aspect was compared: cheaper|better_feature|quality|service|delivery|warranty|other>",
      "customer_sentiment": "<appreciation|complaint|query|suggestion|neutral>",
      "details": "<brief description of what the customer said about competitor>",
      "timestamp": "<approximate time when mentioned, if trackable>"
    }
  ],
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
    "communication": {
      "score": "<score out of 10>",
      "evidence": "<1-2 sentences explaining why - e.g., 'Clear articulation with minimal hesitations' or 'Multiple grammatical errors and unclear phrasing'>"
    },
    "discovery": {
      "score": "<score out of 10>",
      "evidence": "<1-2 sentences explaining why - e.g., 'Asked about budget and preferences' or 'Failed to probe customer needs'>"
    },
    "solution_fit": {
      "score": "<score out of 10>",
      "evidence": "<1-2 sentences - e.g., 'Recommended products matched stated requirements' or 'Recommendations were generic'>"
    },
    "sales_execution": {
      "score": "<score out of 10>",
      "evidence": "<1-2 sentences - e.g., 'Addressed all objections professionally' or 'Failed to handle price objection'>"
    },
    "customer_experience": {
      "score": "<score out of 10>",
      "evidence": "<1-2 sentences - e.g., 'Customer felt heard and respected throughout' or 'Customer expressed frustration'>"
    },
    "primary_category": "<main product category discussed>",
    "overall_sales_outcome": "<successful|unsuccessful|deferred_decision>",
    "customer_intent": "<What did the customer want to achieve? Brief description>",
    "competitor_mentioned": "<Yes|No - did customer mention competitors?>",
    "follow_up_required": "<Yes|No - does this need follow-up?>"
  }
}

LEARNING & DEVELOPMENT SUGGESTIONS FOR SALES EXECUTIVES:
Analyze the sales executive's performance and suggest ONE specific improvement:
- Focus on sales technique, product knowledge, or customer engagement
- Provide actionable coaching with specific examples
- Example: "The customer expressed concern about mattress firmness at 02:30. You could have asked follow-up questions about their sleep position and firmness preferences before recommending, which would have improved solution fit."
- Example: "When the customer mentioned budget constraints at 04:15, instead of pushing premium models, you could have shown value-adds in mid-range options or discussed financing options to address their concern."
- Keep suggestion concise (2-3 sentences max)

EVIDENCE FORMAT FOR INTERACTION MATRICES:
For each score (communication, discovery, solution_fit, sales_execution, customer_experience), provide concise evidence:
- State what happened (factual observation)
- Do NOT explain why it's good/bad
- Do NOT provide coaching (save that for learning_suggestions)
- Keep to 1-2 sentences MAX
- Reference specific timestamps or direct quotes if possible

Examples of GOOD evidence:
✓ "Sales executive asked about sleep position and preferences at 02:15 before recommending"
✓ "Multiple grammatical errors and hesitations throughout; unclear communication at 01:30"
✓ "Customer objected about price at 04:45; executive offered financing option immediately"
✓ "Customer left without making decision; no next steps defined"

Examples of BAD evidence:
✗ "The sales executive demonstrated excellent communication skills by clearly articulating product features"
✗ "Poor sales technique resulted in customer dissatisfaction"
✗ "Executive did well with objection handling"- Only suggest improvements, not praise

COMPETITOR INTELLIGENCE EXTRACTION FOR RETAIL:
When customer mentions any competitor store or brand or compares Wakefit with another furniture/mattress brand:
- Extract competitor company/brand name exactly as stated
- Extract specific product/feature mentioned (e.g., "memory foam", "ergonomic design", "durability", "price point")
- Classify comparison type: cheaper, better_feature, quality, service, delivery, warranty, other
- Classify customer sentiment:
  - appreciation: customer praised competitor's product/service
  - complaint: customer dissatisfied with competitor
  - query: customer asking questions about competitor vs. Wakefit
  - suggestion: customer suggesting Wakefit adopt competitor's approach
  - neutral: factual comparison without emotion
- Include exact details from customer's statement
- Return empty array if NO competitor mentions: "competitor_intelligence": []

PRODUCTS DISCUSSED:
- If multiple products are discussed, create separate entries for each product
- Track customer interest and objections per product
- Record final outcome (purchased, interested, not interested, deferred decision)

SALES EXECUTIVE EVALUATION MATRIX (Scoring 0-10 for each):

**Communication** (Clarity, tone, listening, professionalism):
- 8-10: Clear articulation, professional tone, listens actively, asks clarifying questions
- 5-7: Generally clear, mostly professional, listens adequately
- 2-4: Unclear communication, occasional unprofessional moments, poor listening
- 0-1: Unintelligible, very unprofessional, dismissive of customer

**Discovery** (Understanding customer needs and asking right questions):
- 8-10: Asks targeted questions, identifies pain points, understands budget/preferences
- 5-7: Asks some questions, gets basic understanding
- 2-4: Minimal questioning, misses key requirements
- 0-1: No questioning, ignores customer needs

**Solution Fit** (How well product recommendations matched customer needs):
- 8-10: Recommendations perfectly match requirements, explains relevant benefits
- 5-7: Recommendations mostly appropriate, some benefit explanation
- 2-4: Poor recommendations, weak benefit explanation
- 0-1: Completely irrelevant recommendations

**Sales Execution** (Handling objections, closing, next steps):
- 8-10: Addresses all concerns, asks for sale, defines clear next steps
- 5-7: Addresses some objections, may or may not close
- 2-4: Struggles with objections, weak closing
- 0-1: Fails to address objections, no attempt to close

**Customer Experience** (Overall impression - respect, value, satisfaction):
- 8-10: Customer feels valued, respected, satisfied with experience
- 5-7: Neutral experience, customer satisfied with basics
- 2-4: Customer feels undervalued, somewhat dissatisfied
- 0-1: Customer feels disrespected, very dissatisfied

SCORING GUIDELINES (0-10 scale for overall metrics):

**overall_sentiment**: Rate the overall tone and emotional quality
- 8-10: Positive, friendly, engaged customer throughout
- 5-7: Neutral or mixed emotions, customer browsing casually
- 2-4: Negative tone, customer frustrated, uninterested, or dissatisfied
- 0-1: Highly negative, customer upset or hostile

**customer_satisfaction**: Rate how satisfied the customer appears
- 8-10: Customer very happy with service, made purchase or strong intent
- 5-7: Customer neutral, still deciding, no strong signals
- 2-4: Customer expresses concerns, doubts, or dissatisfaction
- 0-1: Customer very unhappy, leaves dissatisfied

**sales_executive_performance**: Rate the sales executive's overall effectiveness
- 8-10: Excellent product knowledge, engagement, technique, closes effectively
- 5-7: Adequate performance, answers questions but lacks proactive selling
- 2-4: Poor engagement, limited product knowledge, misses opportunities
- 0-1: Very poor performance, unprofessional, pushes too hard or ignores customer

Be realistic with scores. Typical interaction should score 6-8, not perfect 10s.
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
