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
  "customer_satisfaction": <number 0-10>,
  "summary": "<3-5 sentence summary of the interaction>",
  "topics": ["<short topic>", ...],
  "action_items": ["<action item>", ...],
  "key_indicators": ["<short observation supporting the sentiment scores>", ...],
  "customer_name": "<customer name if mentioned, else null>",
  "interaction_outcome": "<short description of how the interaction ended>",
  "learning_suggestions": "<coaching suggestion for the sales executive on how to improve this interaction>",
  "coaching_priorities": [
    {
      "priority": "<short coaching area name, e.g., 'Product Knowledge', 'Objection Handling', 'Active Listening'>",
      "score": <number 0-10 rating current performance in this area>,
      "evidence": "<1-2 sentence example from transcript showing why this needs coaching>"
    }
  ],
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
  },
  "compliance_flags": [
    {
      "flag": "<Misrepresentation|Pressure Tactics|Privacy Violations|Discrimination|Safety Issues|Price Manipulation|Data Security|Professional Conduct|Brand Name Inconsistency>",
      "severity": "<critical|high|medium|low>",
      "description": "<what policy or standard was violated>",
      "evidence": "<exact quote from transcript showing the violation>",
      "timestamp": "<approximate time when it occurred, e.g., [02:30]>"
    }
  ],
  "sla_compliance": <percentage 0-100>
}

TOPICS DISCUSSED:
You MUST select topics from this predefined list only. Do NOT generate random topics:
- Mattresses (Memory Foam, Ortho, Dual Comfort, Latex)
- Pillows (Cervical, Memory Foam, Regular)
- Bed Frames & Storage Beds
- Sofas & Recliners
- Study Tables & Chairs
- Dining Tables & Chairs
- Wardrobes & Storage Solutions
- Shoe Racks
- Bedding (Sheets, Comforters, Protectors)
- Customization Options
- Delivery & Installation
- Warranty & Return Policy
- Pricing & Discounts
- Product Comparison
- Trial Period
- Financing Options
- Product Care & Maintenance

Select only the topics that were actually discussed in the interaction. If a topic was discussed, include it in the topics array.

LEARNING & DEVELOPMENT SUGGESTIONS FOR SALES EXECUTIVES:
Analyze the sales executive's performance and suggest ONE specific improvement:
- Focus on sales technique, product knowledge, or customer engagement
- Provide actionable coaching with specific examples
- Example: "The customer expressed concern about mattress firmness at 02:30. You could have asked follow-up questions about their sleep position and firmness preferences before recommending, which would have improved solution fit."
- Example: "When the customer mentioned budget constraints at 04:15, instead of pushing premium models, you could have shown value-adds in mid-range options or discussed financing options to address their concern."
- Keep suggestion concise (2-3 sentences max)

COACHING PRIORITIES (Identify 1-3 specific improvement areas):
For each coaching priority, provide:
- **priority**: Short name (2-4 words) describing the coaching area
  Common examples: "Product Knowledge", "Objection Handling", "Active Listening", "Needs Discovery", 
  "Closing Technique", "Empathy Building", "Time Management", "Follow-up Skills", "Brand Alignment"
- **score**: Current performance in this area (0-10 scale)
  - 0-3: Critical gap, immediate coaching needed
  - 4-6: Moderate weakness, development opportunity
  - 7-8: Average performance, room for improvement
  - 9-10: Strong performance, minor refinement
- **evidence**: 1-2 sentence example from transcript showing why this area needs coaching
  Include timestamp reference if possible

Identify 1-3 most impactful coaching opportunities based on the transcript.
If performance is excellent across all areas, you may return empty array: "coaching_priorities": []

Examples:
{
  "priority": "Objection Handling",
  "score": 4,
  "evidence": "At [04:15], customer raised price concern but executive only offered discount without explaining value proposition"
}
{
  "priority": "Product Knowledge", 
  "score": 5,
  "evidence": "Executive couldn't answer specific question about mattress foam density at [02:30], causing customer doubt"
}

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

**IMPORTANT: Only extract DIRECT BRAND COMPETITORS - NOT marketplaces or online platforms**

INCLUDE (Direct Furniture/Mattress Brand Competitors):
✓ Sleepwell, Kurlon, Duroflex, Pepperfry, Urban Ladder, IKEA
✓ Godrej Interio, Nilkamal, @home, Hometown, FabIndia
✓ Casper, Emma, Sunday, SleepyCat, The Sleep Company
✓ Any other furniture/mattress brand stores mentioned by customer

EXCLUDE (Do NOT track these as competitors):
✗ Amazon, Flipkart, Myntra, Snapdeal, Meesho, eBay
✗ Any e-commerce marketplace or online shopping platform
✗ Payment platforms, delivery services, logistics companies

EXTRACTION RULES:
When customer mentions a DIRECT BRAND COMPETITOR (from INCLUDE list above):
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

If customer ONLY mentions marketplaces (Amazon, Flipkart, etc.) or NO competitors at all:
- Return empty array: "competitor_intelligence": []

Examples:
- Customer: "I saw this on Amazon" → competitor_intelligence = [] (marketplace, not competitor)
- Customer: "Sleepwell has better pricing" → competitor_intelligence = [{"competitor_name": "Sleepwell", ...}] (direct competitor)
- Customer: "Flipkart delivery is fast" → competitor_intelligence = [] (marketplace service, not product competitor)

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

**customer_satisfaction**: Rate how satisfied the customer appears
- 8-10: Customer very happy with service, made purchase or strong intent
- 5-7: Customer neutral, still deciding, no strong signals
- 2-4: Customer expresses concerns, doubts, or dissatisfaction
- 0-1: Customer very unhappy, leaves dissatisfied

SLA COMPLIANCE FOR IN-STORE SALES (GRADUATED scoring: 0%, 25%, 50%, 75%, 100%)

DEFINITION: SLA Compliance measures adherence to Wakefit's in-store sales standards - the predefined
performance standards and customer service commitments.

SCORING FRAMEWORK (Graduated - Per Interaction Basis):

An interaction receives a graduated compliance score based on how many criteria are met:

COMPLIANCE CRITERIA (Check all 3):

1. GREETING & ENGAGEMENT (Threshold)
   - Standard: Sales executive must greet customer within first 30 seconds of interaction
   - Measurement: Check if greeting/acknowledgment appears in first messages
   - Status: YES if proper greeting within 30 sec, NO if no greeting or delayed
   - Evidence: Look at first sales executive message for greeting

2. NEEDS DISCOVERY (Threshold)
   - Standard: Sales executive must ask discovery questions before recommending products
   - Measurement: Does executive ask about needs, preferences, budget before pitching?
   - Status: YES if asks questions, NO if immediately pushes products without discovery
   - Evidence: Check if executive asks questions like "What are you looking for?", "What's your budget?", etc.

3. PRODUCT KNOWLEDGE & ACCURACY (Threshold)
   - Standard: All product information provided must be accurate (no false claims)
   - Measurement: Check if any misinformation, false promises, or inaccurate specs mentioned
   - Status: YES if all information accurate, NO if any false/misleading claims
   - Evidence: Look for compliance_flags with "Misrepresentation" category

SCORING LOGIC (Graduated Based on Criteria Met):
- If 0 criteria met → sla_compliance = 0 (no compliance)
- If 1 criterion met → sla_compliance = 25 (minimal compliance)
- If 2 criteria met → sla_compliance = 50 (partial compliance)
- If 3 criteria met → sla_compliance = 75 (substantial compliance)
- If all 3 criteria met PLUS excellent customer experience → sla_compliance = 100 (full compliance)

RATIONALE: Graduated scoring recognizes that partial adherence is better than none,
allowing for realistic retail metrics where perfect compliance is rare.

Example Calculation:
- Interaction A: Greeting ✅, Discovery ✅, Accurate info ✅ → 3/3 criteria → 75-100%
- Interaction B: No greeting ❌, Discovery ✅, Accurate info ✅ → 2/3 criteria → 50%

OVERALL SENTIMENT & SALES EXECUTIVE PERFORMANCE CALCULATION:
Do NOT include "overall_sentiment" or "sales_executive_performance" in the response. 
These will be automatically calculated as:
- sales_executive_performance = average of 5 SALES EXECUTIVE EVALUATION MATRIX scores (communication, discovery, solution_fit, sales_execution, customer_experience)
- overall_sentiment = (customer_satisfaction + sales_executive_performance) / 2

SCORING GUIDELINES (0-10 scale for overall metrics):

**customer_satisfaction**: Rate how satisfied the customer appears
- 8-10: Customer very happy with service, made purchase or strong intent
- 5-7: Customer neutral, still deciding, no strong signals
- 2-4: Customer expresses concerns, doubts, or dissatisfaction
- 0-1: Customer very unhappy, leaves dissatisfied

COMPLIANCE & RISK FLAGS:
Check for violations of sales policies and ethical standards. Use these EXACT category names:
- **Misrepresentation**: False claims about products, features, or benefits
- **Pressure Tactics**: Aggressive selling, rushing customer, creating false urgency
- **Privacy Violations**: Improper handling of customer personal information
- **Discrimination**: Biased treatment based on protected characteristics
- **Safety Issues**: Ignoring safety concerns or product warnings
- **Price Manipulation**: Unauthorized discounts or misleading pricing
- **Data Security**: Insecure handling of payment information
- **Professional Conduct**: Rudeness, inappropriate language, disrespect
- **Brand Name Inconsistency**: Using wrong company name or inconsistent branding

For each flag, return exactly these fields in this exact order:
- **category**: ONE of the exact categories listed above (FIRST field)
- **severity**: critical | high | medium | low
  - critical: immediate escalation needed
  - high: management review required
  - medium: coaching needed
  - low: minor note
- **description**: Brief explanation of what policy/standard was violated
- **evidence**: Exact quote from transcript showing the violation
- **timestamp**: Approximate time in format [MM:SS] when it occurred

If NO compliance issues found, return empty array: "compliance_flags": []

Example of CORRECT compliance flag structure:
{
  "category": "Brand Name Inconsistency",
  "severity": "medium",
  "description": "Sales executive used incorrect company name during closing",
  "evidence": "thank you so much for choosing Great Fit",
  "timestamp": "[10:11]"
}

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
        
        # Calculate sales_executive_performance from the 5 SALES EXECUTIVE EVALUATION MATRIX metrics
        interaction_matrices = parsed.get('interaction_matrices', {})
        if interaction_matrices:
            scores = []
            for metric in ['communication', 'discovery', 'solution_fit', 'sales_execution', 'customer_experience']:
                metric_data = interaction_matrices.get(metric, {})
                if isinstance(metric_data, dict) and 'score' in metric_data:
                    try:
                        score = float(metric_data['score'])
                        scores.append(score)
                    except (TypeError, ValueError):
                        pass
            
            if scores:
                # Average of 5 metrics (already on 0-10 scale)
                sales_executive_performance = sum(scores) / len(scores)
                parsed['sales_executive_performance'] = round(sales_executive_performance, 2)
                
                # Also store as average_score in interaction_matrices
                interaction_matrices['average_score'] = round(sales_executive_performance, 2)
            else:
                parsed['sales_executive_performance'] = None
        else:
            parsed['sales_executive_performance'] = None
        
        # Calculate overall_sentiment = (customer_satisfaction + sales_executive_performance) / 2
        csat = parsed.get('customer_satisfaction')
        sales_perf = parsed.get('sales_executive_performance')
        
        if csat is not None and sales_perf is not None:
            try:
                overall_sentiment = (float(csat) + float(sales_perf)) / 2
                parsed['overall_sentiment'] = round(overall_sentiment, 2)
            except (TypeError, ValueError):
                parsed['overall_sentiment'] = None
        else:
            parsed['overall_sentiment'] = None
        
        # Calculate interaction_matrices total and percentage
        if interaction_matrices and scores:
            # Each metric is out of 10, so total is out of 50
            total_possible = 50
            total_points = sum(scores)
            percentage = (total_points / total_possible) * 100
            
            interaction_matrices['total_points'] = round(total_points, 2)
            interaction_matrices['total_possible'] = total_possible
            interaction_matrices['percentage'] = round(percentage, 2)
            parsed['interaction_matrices'] = interaction_matrices
        
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
