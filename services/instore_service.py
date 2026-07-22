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
    """Format messages for in-store analysis with precise timestamps and participant tracking"""
    if not messages:
        return "(No conversation)"
    lines = []
    for msg in messages:
        # Determine participant role
        role = msg.get("speaker_role") or msg.get("role")  # Use explicit speaker_role if available
        speaker_id = msg.get("speaker_id", "")
        
        # Map role to participant type
        if role == "user" or role == "customer":
            participant = "Customer"
        elif role == "agent" or role == "sales_executive":
            participant = "Sales Executive"
        elif role == "manager" or role == "supervisor":
            participant = "Store Manager"
        elif role == "staff" or role == "other_staff":
            participant = "Staff Member"
        else:
            # Fallback based on speaker_id pattern if provided
            if speaker_id and "cust" in speaker_id.lower():
                participant = "Customer"
            else:
                participant = "Sales Executive"
        
        # Add speaker_id for multi-exec scenarios (e.g., "Sales Executive (Ananya)")
        if speaker_id and participant != "Customer":
            participant = f"{participant} ({speaker_id})"
        
        # Include timestamp for accurate coaching suggestions and compliance flags
        timestamp = "00:00"
        if 'timestamp' in msg and isinstance(msg['timestamp'], str) and ':' in msg['timestamp']:
            timestamp = msg['timestamp']
        elif 'start_time' in msg:
            # Calculate from start_time in seconds
            start_seconds = float(msg['start_time'])
            minutes = int(start_seconds // 60)
            seconds = int(start_seconds % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
        
        lines.append(f"[{timestamp}] {participant}: {msg.get('text', '')}")
    return "\n".join(lines)


_INSTORE_ANALYSIS_PROMPT = """You are an EXPERT in-store sales interaction analyst for Wakefit retail stores. Your analysis must be EXTREMELY PRECISE and evidence-based.

ANALYZE the transcript and return ONLY a valid JSON object (no ```json markdown, no commentary).

## CRITICAL: EXACT TIMESTAMPS REQUIRED
- Use EXACT [MM:SS] timestamps from transcript in ALL evidence and compliance flags
- NEVER approximate or guess timestamps
- Timestamps are MANDATORY for precision

## JSON STRUCTURE (use EXACTLY these field names):

{
  "customer_satisfaction": <0-10>,
  "summary": "<3-5 precise sentences>",
  "topics": ["<from predefined list only>"],
  "action_items": ["<specific actionable item>"],
  "key_indicators": ["<factual observation with timestamp [MM:SS]>"],
  "customer_name": "<name or null>",
  "interaction_outcome": "<precise description of final outcome>",
  "learning_suggestions": "<2-3 sentences with specific example and timestamp [MM:SS]>",
  "coaching_priorities": [{
    "priority": "<2-4 word area name>",
    "score": <0-10>,
    "evidence": "<factual 1-2 sentences with timestamp [MM:SS]>",
    "suggestion": "<2-3 specific actionable sentences>"
  }],
  "competitor_intelligence": [{
    "competitor_name": "<brand name ONLY, not marketplaces>",
    "product_mentioned": "<specific product/feature>",
    "comparison_type": "<cheaper|better_feature|quality|service|delivery|warranty|other>",
    "customer_sentiment": "<appreciation|complaint|query|suggestion|neutral>",
    "details": "<precise description of what customer said>",
    "timestamp": "<[MM:SS] exact from transcript>"
  }],
  "products_discussed": [{
    "product_name": "<specific product name>",
    "category": "<Mattress|Pillow|Bed Frame|Sofa|etc>",
    "sub_category": "<specific type if mentioned>",
    "discussion_summary": "<factual summary of discussion>",
    "customer_interest_level": "<High|Medium|Low>",
    "price_discussed": "<Yes|No>",
    "price_amount": "<exact amount or null>",
    "objections_raised": ["<specific customer objection>"],
    "sales_outcome": "<purchased|interested|not_interested|deferred>",
    "outcome_reason": "<factual reason why>"
  }],
  "interaction_matrices": {
    "communication": {"score": <0-10>, "evidence": "<factual observation only>"},
    "discovery": {"score": <0-10>, "evidence": "<factual observation only>"},
    "solution_fit": {"score": <0-10>, "evidence": "<factual observation only>"},
    "sales_execution": {"score": <0-10>, "evidence": "<factual observation only>"},
    "customer_experience": {"score": <0-10>, "evidence": "<factual observation only>"},
    "primary_category": "<main product category discussed>",
    "overall_sales_outcome": "<successful|unsuccessful|deferred_decision>",
    "customer_intent": "<what customer specifically wanted>",
    "competitor_mentioned": "<Yes|No>",
    "follow_up_required": "<Yes|No>",
    "total_products_discussed": <number of products>,
    "products_sold": <number of products with sales_outcome=purchased>,
    "conversion_rate_count": "<products_sold/total_products_discussed like 2/5>",
    "conversion_rate_percentage": <percentage number>
  },
  "compliance_flags": [{
    "flag": "<Misrepresentation|Pressure Tactics|Privacy Violations|Discrimination|Safety Issues|Price Manipulation|Data Security|Professional Conduct|Brand Name Inconsistency>",
    "severity": "<critical|high risk|medium risk|low risk>",
    "description": "<what policy was violated>",
    "evidence": "<exact quote from transcript>",
    "timestamp": "<[MM:SS] exact from transcript>"
  }],
  "sla_compliance": <0|25|50|75|100>
}

## OVERALL SALES OUTCOME RULES (CRITICAL - FOLLOW EXACTLY):

Determine overall_sales_outcome based on products sold:

**successful** - Use when:
- At least ONE product was sold (sales_outcome = "purchased")
- Customer completed payment for any product
- products_sold >= 1

**unsuccessful** - Use when:
- ZERO products were sold (products_sold = 0)
- ALL products have sales_outcome = "not_interested"
- Customer left without buying anything

**deferred_decision** - Use when:
- ZERO products were sold (products_sold = 0)
- At least ONE product has sales_outcome = "interested" or "deferred"
- Customer showed interest but didn't purchase today

**CONVERSION RATE CALCULATION**:
- total_products_discussed = total count of items in products_discussed array
- products_sold = count of products with sales_outcome = "purchased"
- conversion_rate_count = products_sold / total_products_discussed (e.g., "2/5")
- conversion_rate_percentage = (products_sold / total_products_discussed) * 100 (e.g., 40)
- If no products discussed, set all to 0

## ULTRA-PRECISE SCORING GUIDELINES (0-10 scale):

**Communication** (Clarity, professionalism, listening):
- 10: Flawless communication, perfect clarity, exceptional listening, zero hesitations
- 9: Excellent communication, clear and professional, active listening
- 8: Very good communication, mostly clear, good listening
- 7: Good communication, generally clear, adequate listening  
- 6: Acceptable communication, some unclear moments, basic listening
- 5: Average communication, frequent unclear moments, passive listening
- 4: Below average, often unclear, poor listening skills
- 3: Poor communication, very unclear, dismissive listening
- 2: Very poor, mostly unclear, ignores customer
- 1: Terrible communication, unintelligible
- 0: No meaningful communication

**Discovery** (Understanding needs BEFORE recommending):
- 10: Asks 4+ targeted questions, fully understands all needs, budget, preferences
- 9: Asks 3+ targeted questions, understands most needs and budget
- 8: Asks 2-3 good questions, understands basic needs and some preferences
- 7: Asks 2 questions, gets basic understanding of needs
- 6: Asks 1-2 basic questions, partial understanding
- 5: Asks 1 generic question, minimal understanding
- 4: Very minimal questioning, makes assumptions
- 3: Almost no questioning, guesses at needs
- 2: No real questioning, ignores customer needs
- 1: Jumps straight to selling without any discovery
- 0: Completely ignores customer

**Solution Fit** (Do recommendations match actual stated needs):
- 10: Perfect match to ALL stated needs, explains exactly why it fits
- 9: Excellent match to most needs, good explanation of fit
- 8: Good match to main needs, some explanation
- 7: Decent match to basic needs, basic explanation
- 6: Partial match to some needs
- 5: Mediocre match, generic recommendations
- 4: Poor match to stated needs
- 3: Very poor match, mostly irrelevant
- 2: Completely wrong recommendations
- 1: Recommendations contradict stated needs
- 0: No recommendations or completely inappropriate

**Sales Execution** (Objection handling, closing, next steps):
- 10: Expertly handles ALL objections, smooth natural close, crystal clear next steps
- 9: Handles most objections well, good close attempt, clear next steps
- 8: Addresses main objections, decent close, some next steps
- 7: Handles some objections, attempts close, basic next steps
- 6: Struggles with objections, weak close attempt
- 5: Minimal objection handling, no clear close
- 4: Poor objection handling, avoids closing
- 3: Fails to address concerns, no attempt to close
- 2: Makes objections worse, pushes without addressing concerns
- 1: Completely ignores objections, aggressive pushing
- 0: No sales process whatsoever

**Customer Experience** (Respect, value, satisfaction):
- 10: Customer feels extremely valued, respected, and satisfied throughout
- 9: Customer feels very valued and respected, very positive experience
- 8: Customer feels valued and respected, positive experience
- 7: Customer feels adequately respected, decent experience
- 6: Customer feels neutral, neither positive nor negative
- 5: Customer feels somewhat undervalued, neutral experience
- 4: Customer feels undervalued, somewhat negative experience
- 3: Customer feels disrespected, negative experience
- 2: Customer feels very disrespected, very negative experience
- 1: Customer feels insulted or mistreated
- 0: Completely unacceptable treatment

**customer_satisfaction** (Overall happiness with interaction):
- 10: Customer ecstatic, made purchase with enthusiasm
- 9: Customer very happy, strong purchase intent or completed purchase
- 8: Customer satisfied and pleased, considering purchase seriously
- 7: Customer content, some purchase interest
- 6: Customer neutral, undecided
- 5: Customer slightly disappointed, minimal interest
- 4: Customer somewhat dissatisfied, expressed concerns
- 3: Customer unhappy, significant concerns
- 2: Customer very dissatisfied, frustrated
- 1: Customer angry or upset
- 0: Customer extremely upset, stormed out

## SLA COMPLIANCE - GRADUATED SCORING (0%, 25%, 50%, 75%, 100%):

**Check these 3 CRITICAL criteria:**
1. **GREETING & ENGAGEMENT**: Did sales executive greet customer within first 30 seconds? (YES/NO)
2. **NEEDS DISCOVERY**: Did sales executive ask discovery questions BEFORE recommending products? (YES/NO)  
3. **PRODUCT ACCURACY**: Was ALL product information accurate with NO false claims? (YES/NO)

**SCORING LOGIC:**
- 0 criteria met = 0% (Failed all standards)
- 1 criterion met = 25% (Minimal compliance)
- 2 criteria met = 50% (Partial compliance) 
- 3 criteria met = 75% (Substantial compliance)
- 3 criteria met + customer_satisfaction 8+ = 100% (Excellent compliance)

## COMPLIANCE FLAGS - BE EXTREMELY STRICT:

**Misrepresentation**: Any false product claims, incorrect specifications, misleading information
**Price Manipulation**: Inconsistent pricing, fake discounts, artificial urgency, hidden costs
**Pressure Tactics**: Aggressive closing, making customer uncomfortable, not accepting "no"
**Data Security**: Verbally sharing sensitive information (cards, passwords, personal details)
**Professional Conduct**: Off-topic discussions, inappropriate behavior, unprofessional language
**Brand Name Inconsistency**: Calling Wakefit by wrong name or mispronouncing consistently

## TOPICS - SELECT ONLY FROM THIS EXACT LIST:
- Storage Solutions
- Customization Options
- Delivery & Installation
- Warranty & Return Policy
- Pricing & Discounts
- Product Comparison
- Trial Period
- Financing Options
- Product Care & Maintenance

## COACHING PRIORITIES - IDENTIFY 1-3 HIGHEST IMPACT IMPROVEMENTS:

**Priority Areas:** Product Knowledge, Objection Handling, Active Listening, Needs Discovery, 
Closing Technique, Empathy Building, Time Management, Follow-up Skills, Price Justification, 
Communication Clarity, Solution Presentation, Customer Rapport

**For each priority:**
- Use SPECIFIC examples with EXACT timestamps [MM:SS]
- Provide ACTIONABLE suggestions (not generic advice)
- Focus on highest ROI improvements

## EVIDENCE FORMAT - CRITICAL REQUIREMENTS:

**For interaction_matrices evidence:**
- State ONLY what happened (factual observations)
- Include timestamps [MM:SS] when available
- NO editorializing, NO coaching language, NO opinions
- Maximum 2 sentences
- Reference specific quotes or actions

**GOOD Examples:**
- "Executive asked about sleep position and budget at [02:15] before making recommendations"
- "Customer objected to price at [04:45]; executive immediately offered financing options"  
- "Multiple hesitations and unclear explanations throughout conversation"
- "Customer expressed frustration at [06:30] when executive couldn't answer density question"

**BAD Examples:**
- "Executive demonstrated excellent discovery skills"
- "Poor sales technique led to customer dissatisfaction"  
- "Good objection handling throughout"
- "Executive needs to improve communication"

## COMPETITOR INTELLIGENCE - DIRECT BRANDS ONLY:

**INCLUDE (Track these direct competitors):**
- Sleepwell, Kurlon, Duroflex, Pepperfry, Urban Ladder, IKEA
- Godrej Interio, Nilkamal, @home, Hometown, FabIndia  
- Casper, Emma, Sunday, SleepyCat, The Sleep Company
- Any furniture/mattress brand with physical stores

**EXCLUDE (Do NOT track):**
- Amazon, Flipkart, Myntra, Snapdeal, Meesho, eBay (marketplaces)
- Payment platforms, delivery services, logistics companies
- Generic online shopping mentions

If NO direct brand competitors mentioned → return empty array []

## TIMESTAMP ACCURACY - MANDATORY:

- Use EXACT timestamps from transcript: [MM:SS] format
- NEVER approximate, estimate, or guess timestamps
- For compliance flags: timestamp when violation first occurred
- For evidence: timestamp when specific behavior happened
- If no timestamp visible, state the facts without timestamp

## CRITICAL FINAL REQUIREMENTS:

1. Return ONLY valid JSON (no markdown code fences)
2. All timestamps must be EXACT from transcript
3. Evidence must be factual observations only
4. Scoring must follow the precise 0-10 rubrics above
5. Be ruthlessly objective and evidence-based

Base ALL analysis strictly on transcript evidence."""


def analyze_instore_interaction(messages, interaction_id=None, language_details=None):
    """
    Analyze in-store interaction using Claude
    Returns analysis dict with sentiment, matrices, and insights
    
    Args:
        messages: Transcript messages
        interaction_id: ID of the interaction (for logging)
        language_details: Language breakdown dictionary from detect_language_improved()
    """
    if not messages:
        return {}
    
    conversation_text = format_transcript_for_instore(messages)
    
    try:
        bedrock_client = _get_bedrock_client()
        
        # Use Claude Sonnet 4.5 - better at structured output than Haiku
        model_id = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"  # Sonnet 4.5 for reliable JSON
        
        response = bedrock_client.converse(
            modelId=PCA_MODEL_ID,
            system=[{"text": _INSTORE_ANALYSIS_PROMPT}],
            messages=[{
                "role": "user",
                "content": [{"text": f"In-store interaction transcript:\n\n{conversation_text}"}]
            }],
            inferenceConfig={"maxTokens": 4096, "temperature": 0.3}
        )
        
        text = response["output"]["message"]["content"][0]["text"].strip()
        parsed = _parse_json(text)
        
        # If parsing failed, return empty dict
        if not parsed:
            print(f"[InStore] Analysis failed due to JSON parse error for {interaction_id or 'interaction'}")
            # Return minimal valid structure so the system doesn't crash
            return {
                "customer_satisfaction": None,
                "summary": "Analysis failed due to parsing error",
                "topics": [],
                "action_items": [],
                "key_indicators": [],
                "customer_name": None,
                "interaction_outcome": "Unknown",
                "learning_suggestions": "Unable to generate coaching suggestions",
                "coaching_priorities": [],
                "competitor_intelligence": [],
                "products_discussed": [],
                "interaction_matrices": {},
                "compliance_flags": [],
                "sla_compliance": None,
                "sales_executive_performance": None,
                "overall_sentiment": None
            }
        
        # Store language breakdown in raw_model_response for analytics aggregation
        if language_details and isinstance(language_details, dict):
            parsed['language_breakdown'] = language_details.get('language_breakdown', {})
        
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
    """Parse JSON from model output with robust error recovery"""
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
    
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[InStore] JSON parse error at line {e.lineno} col {e.colno}: {e.msg}")
        print(f"[InStore] Error near: {e.doc[max(0, e.pos-50):e.pos+50]}")
    
    # Try to extract JSON from text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            extracted = cleaned[start:end + 1]
            return json.loads(extracted)
        except json.JSONDecodeError as exp:
            print(f"[InStore] Extracted JSON parse error at line {exp.lineno}: {exp.msg}")
            
            # Last resort: try to fix common JSON issues
            try:
                import re
                fixed = extracted
                
                # Fix 1: Remove orphaned values (value without key)
                # Pattern: "},<value>," or "],<value>," → remove the orphaned value
                # This handles cases like "},"3/8","conversion_rate_percentage":" → remove "3/8"
                fixed = re.sub(r'([}\]]),\s*"[^"]*"\s*,', r'\1,', fixed)
                fixed = re.sub(r'([}\]]),\s*\d+(\.\d+)?\s*,', r'\1,', fixed)
                
                # Fix 2: Replace unescaped quotes inside strings
                # Pattern: Find ": "something" and fix to ": \"something\"
                # This handles cases where nested quotes aren't escaped
                fixed = re.sub(r'": "([^"]*)"([^"]*)"', r'": "\1\\\\\2"', fixed)
                
                # Fix 3: Find unterminated strings (quote followed by comma/newline without closing)
                # Look for patterns like: "key": "value that continues
                lines = fixed.split('\n')
                fixed_lines = []
                for i, line in enumerate(lines):
                    # Check if line has unclosed quote
                    quote_count = line.count('"') - line.count('\\"')
                    if quote_count % 2 == 1:  # Odd number = unclosed quote
                        # Try to close the quote before the next line
                        line = line.rstrip(',').rstrip() + '"'
                    fixed_lines.append(line)
                fixed = '\n'.join(fixed_lines)
                
                # Fix 4: Remove control characters (non-printable chars)
                fixed = ''.join(char for char in fixed if ord(char) >= 32 or char in '\n\r\t')
                
                # Try to parse the fixed version
                result = json.loads(fixed)
                print(f"[InStore] Successfully recovered JSON after fixes")
                return result
            except Exception as fix_error:
                print(f"[InStore] Failed to auto-fix JSON: {fix_error}")
                
                # Final attempt: Try to extract just the critical parts
                try:
                    # At least try to salvage something
                    if '"overall_sentiment"' in cleaned or '"customer_satisfaction"' in cleaned:
                        print("[InStore] Attempting partial extraction of key fields...")
                        partial = {}
                        
                        # Extract numeric fields
                        import re
                        sentiment_match = re.search(r'"overall_sentiment"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
                        if sentiment_match:
                            partial['overall_sentiment'] = float(sentiment_match.group(1))
                        
                        csat_match = re.search(r'"customer_satisfaction"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
                        if csat_match:
                            partial['customer_satisfaction'] = float(csat_match.group(1))
                        
                        perf_match = re.search(r'"sales_executive_performance"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
                        if perf_match:
                            partial['sales_executive_performance'] = float(perf_match.group(1))
                        
                        if partial:
                            print(f"[InStore] Extracted partial data: {partial}")
                            return partial
                except:
                    pass
    
    print("[InStore] Could not parse model output as JSON")
    return {}


def _to_score(value):
    """Convert value to score between 0-10"""
    try:
        return max(0.0, min(10.0, round(float(value), 2)))
    except (TypeError, ValueError):
        return None
