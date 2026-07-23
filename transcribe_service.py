"""
Audio transcription service using ElevenLabs
"""
import os
import json
import boto3
from datetime import datetime
from elevenlabs.client import ElevenLabs

AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')
S3_RECORDINGS_BUCKET = os.environ.get('S3_RECORDINGS_BUCKET', 'sahaa-voiceai-recordings')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')
ELEVENLABS_BASE_URL = os.environ.get('ELEVENLABS_BASE_URL', 'https://api.elevenlabs.in')


def _get_s3_client():
    """Get AWS S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=AWS_REGION
    )


def _get_elevenlabs_client():
    """Get ElevenLabs client with regional support and extended timeout"""
    return ElevenLabs(
        api_key=ELEVENLABS_API_KEY,
        base_url=ELEVENLABS_BASE_URL,
        timeout=7200.0
    )


def _detect_speakers_with_llm(raw_transcript, interaction_type='pca'):
    """Use LLM to detect speaker roles in transcript with high precision"""
    
    if interaction_type == 'instore':
        speaker_detection_prompt = """You are an EXPERT at analyzing in-store sales transcripts. Your ONLY job is to identify WHO IS WHO in the conversation.

**CRITICAL ACCURACY REQUIREMENT**: You MUST get this 100 percentage correct. Read the ENTIRE transcript multiple times.

**THE FUNDAMENTAL RULE**:
- CUSTOMER = person who WANTS TO BUY something, asks about prices, makes purchase decisions
- SALES EXECUTIVE = person who WORKS at the store, explains products, provides prices, processes orders

**HOW TO IDENTIFY CUSTOMERS** (they are BUYING):
They will say things like:
- "kitna paisa hai?" "price kya hai?" "how much?" (asking price)
- "mujhe chahiye" "main le lunga" "I want to buy" (expressing purchase intent)
- "budget mera X hai" (discussing their budget)
- "dikhao" "show me" (asking to see products)
- "theek hai" "ok main le leta hoon" (making decisions)
- They ASK MORE than they ANSWER
- They show uncertainty: "sochta hoon", "confirm kar dete hain"

**HOW TO IDENTIFY SALES EXECUTIVES** (they are SELLING):
They will say things like:
- "ye mattress hai" "ye model hai" (introducing/explaining products)
- "price itna hai" "discount milega" (giving prices and offers)
- "sir" "madam" repeatedly (addressing customers respectfully)
- "aap kya dekhna chahte ho?" (asking what customer wants)
- "delivery 10 din mein hogi" (explaining logistics)
- "warranty 12 saal ki hai" (explaining product features)
- They ANSWER MORE than they ASK
- They show product knowledge and authority

**CRITICAL ANALYSIS STEPS**:
1. Read the ENTIRE transcript word by word
2. For EACH speaker, list what they say and count:
   - How many times they ask prices vs give prices
   - How many times they ask questions vs answer questions
   - Do they use respectful terms (sir/madam) or are they addressed with respect?
   - Do they explain products or ask about products?
3. The person asking prices and making decisions = CUSTOMER
4. The person giving prices and explaining products = SALES EXECUTIVE

**RESPONSE FORMAT** (ONLY JSON, NO MARKDOWN, NO EXPLANATIONS):
{
  "speakers": {
    "speaker_0": "customer|sales_executive|manager|staff|other",
    "speaker_1": "customer|sales_executive|manager|staff|other",
    "speaker_2": "customer|sales_executive|manager|staff|other"
  },
  "primary_sales_executive": "speaker_X",
  "analysis": "speaker_0 asks 'kitna paisa hai' at 5 places, says 'mujhe chahiye', negotiates price = CUSTOMER. speaker_1 says 'ye mattress Rs 20,000', 'sir aapko', explains warranty = SALES EXECUTIVE.",
  "confidence": "high|medium|low"
}

**CRITICAL**: 
- Return ONLY the JSON object
- NO markdown backticks (```), NO "```json"
- NO explanations outside the JSON
- Use EXACT evidence from transcript in analysis field"""
    else:
        speaker_detection_prompt = """You are an expert at analyzing call transcripts with PERFECT ACCURACY required.

CRITICAL: Analyze the FULL transcript provided and identify which speaker is the CUSTOMER and which is the CUSTOMER SUPPORT AGENT.

MANDATORY ANALYSIS STEPS:
1. Read through the ENTIRE transcript carefully word by word
2. Identify who provides COMPANY SUPPORT (Wakefit agent) - uses professional language, helps resolve issues
3. Identify who is CALLING FOR HELP (Customer) - describes problems, asks for solutions

CUSTOMER SUPPORT AGENT INDICATORS:
- Professional greeting: "thank you for calling", "How can I help", "Wakefit support"
- Uses customer names or references their order
- Explains policies: "warranty", "delivery", "refund policy", "EMI"
- Takes action: "let me check", "I can see", "I'll arrange"
- Provides solutions: "we can deliver", "refund available", "here's what we can do"

CUSTOMER INDICATORS:
- Describes problem: "not received", "damaged", "late delivery", "wrong item"
- Asks for help: "what can you do", "can you help", "when will it arrive"
- References their purchase: "order number", "date of purchase", "I ordered"
- Expresses frustration/concern: "worried", "frustrated", "need it urgently"
- Makes requests: "can you refund", "can you deliver", "can you check"

Return ONLY valid JSON (absolutely NO markdown, NO explanation):
{
  "customer_speaker": "speaker_0|speaker_1",
  "support_speaker": "speaker_0|speaker_1",
  "analysis": "<specific evidence from transcript>",
  "confidence": "high"
}"""
    
    try:
        bedrock_client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "ap-south-1"),
        )
        
        model_id = os.environ.get("PCA_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
        
        resp = bedrock_client.converse(
            modelId=model_id,
            system=[{"text": speaker_detection_prompt}],
            messages=[{"role": "user", "content": [{"text": f"Transcript:\n\n{raw_transcript}"}]}],
            inferenceConfig={"maxTokens": 4000},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        
        # Log the raw LLM response for debugging
        print(f"[TRANSCRIBE] LLM raw response for {interaction_type}:")
        print(f"[TRANSCRIBE] {text[:500]}")  # First 500 chars
        
        # Strip markdown code fences if present
        import json as json_module
        cleaned_text = text
        if text.startswith('```'):
            # Remove opening fence (```json or ```)
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_text = '\n'.join(lines).strip()
            print(f"[TRANSCRIBE] Stripped markdown fences, cleaned text: {cleaned_text[:200]}")
        
        # Parse JSON response with strict validation
        try:
            parsed = json_module.loads(cleaned_text)
            
            # Validate response structure
            if interaction_type == 'instore':
                if 'speakers' in parsed and 'primary_sales_executive' in parsed:
                    print(f"[TRANSCRIBE] Speaker detection SUCCESS: {parsed.get('speakers')}")
                    print(f"[TRANSCRIBE] Analysis: {parsed.get('analysis', 'N/A')}")
                    return parsed
                else:
                    print(f"[TRANSCRIBE] INVALID response structure - missing required fields")
                    print(f"[TRANSCRIBE] Has 'speakers': {'speakers' in parsed}")
                    print(f"[TRANSCRIBE] Has 'primary_sales_executive': {'primary_sales_executive' in parsed}")
            else:
                if 'customer_speaker' in parsed and 'support_speaker' in parsed:
                    return parsed
                else:
                    print(f"[TRANSCRIBE] INVALID response structure for PCA")
            
            # If response missing required fields, try extraction
            print(f"[TRANSCRIBE] Invalid response structure: {text[:200]}")
        except Exception as parse_error:
            print(f"[TRANSCRIBE] JSON parse error: {parse_error}")
            # Try to extract JSON from text by finding { and }
            start = cleaned_text.find('{')
            end = cleaned_text.rfind('}') + 1
            if start >= 0 and end > start:
                try:
                    extracted = json_module.loads(cleaned_text[start:end])
                    
                    # Validate extracted structure
                    if interaction_type == 'instore':
                        if 'speakers' in extracted and 'primary_sales_executive' in extracted:
                            print(f"[TRANSCRIBE] Speaker detection SUCCESS (extracted): {extracted.get('speakers')}")
                            return extracted
                    else:
                        if 'customer_speaker' in extracted and 'support_speaker' in extracted:
                            return extracted
                    
                    print(f"[TRANSCRIBE] Extracted JSON missing required fields")
                except Exception as extract_error:
                    print(f"[TRANSCRIBE] Extracted JSON also failed: {extract_error}")
        
        print(f"[TRANSCRIBE] Could not parse speaker detection response")
        return None
        
    except Exception as e:
        print(f"[TRANSCRIBE] Speaker detection LLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return None



def upload_audio_to_s3(audio_file, call_id):
    """Upload audio file to S3 and return S3 key"""
    s3_client = _get_s3_client()
    
    # S3 key format: {call_id}/recording.wav
    s3_key = f"{call_id}/recording.wav"
    
    # Upload to S3
    audio_file.seek(0)
    s3_client.upload_fileobj(
        audio_file,
        S3_RECORDINGS_BUCKET,
        s3_key,
        ExtraArgs={'ContentType': 'audio/wav'}
    )
    
    return s3_key


def _word_format(text):
    """Enhanced word formatting with 50+ corrections for Wakefit domain"""
    if not text:
        return text
    
    import re
    
    # Expanded Wakefit brand corrections (50+ variations)
    wakefit_patterns = [
        # Common misheard variations
        (r'\bwiprit\b', 'Wakefit'), (r'\brakefirt\b', 'Wakefit'), (r'\bwakefeet\b', 'Wakefit'),
        (r'\bwakefeat\b', 'Wakefit'), (r'\bwakepit\b', 'Wakefit'), (r'\bwikfit\b', 'Wakefit'),
        (r'\bwikfeet\b', 'Wakefit'), (r'\bvakefit\b', 'Wakefit'), (r'\bvakfit\b', 'Wakefit'),
        (r'\bweakfit\b', 'Wakefit'), (r'\bwekfit\b', 'Wakefit'), (r'\bwagfit\b', 'Wakefit'),
        # Spaced variations
        (r'\bwak\s*fit\b', 'Wakefit'), (r'\bwake\s*fit\b', 'Wakefit'), (r'\bwik\s*fit\b', 'Wakefit'),
        (r'\bvake\s*fit\b', 'Wakefit'), (r'\bwak\s*feet\b', 'Wakefit'), (r'\bwake\s*feet\b', 'Wakefit'),
        (r'\bwak\s*pit\b', 'Wakefit'), (r'\bwake\s*pit\b', 'Wakefit'), (r'\brake\s*fit\b', 'Wakefit'),
        (r'\brake\s*firt\b', 'Wakefit'), (r'\brak\s*fit\b', 'Wakefit'), (r'\brak\s*feet\b', 'Wakefit'),
        # Hindi influenced variations
        (r'\bवेकफिट\b', 'Wakefit'), (r'\bवाकफिट\b', 'Wakefit'), (r'\bवेक\s*फिट\b', 'Wakefit'),
        # Regional pronunciations
        (r'\bwepfit\b', 'Wakefit'), (r'\bwakephit\b', 'Wakefit'), (r'\bwegfit\b', 'Wakefit'),
        (r'\bwagphit\b', 'Wakefit'), (r'\bwekphit\b', 'Wakefit'), (r'\bwakepit\b', 'Wakefit'),
        # Catch-all pattern for similar sounding variations
        (r'\b[wrv][aei]k?[ec]?\s*(?:fit|pit|feet|firt|feit|feat|fert|phit)\b', 'Wakefit'),
        # Already correct
        (r'\bwakefit\b', 'Wakefit'),
    ]
    
    # Product name corrections
    product_patterns = [
        (r'\bortho\s*medic\b', 'Ortho-Medic'), (r'\bortho\s*medik\b', 'Ortho-Medic'),
        (r'\bmemory\s*foam\b', 'Memory Foam'), (r'\bmemori\s*foam\b', 'Memory Foam'),
        (r'\bduet\s*mattress\b', 'Duet Mattress'), (r'\bduet\s*matress\b', 'Duet Mattress'),
        (r'\belev8\b', 'Elev8'), (r'\belevate\b', 'Elev8'), (r'\belev\s*8\b', 'Elev8'),
        (r'\bzen\s*mattress\b', 'Zen Mattress'), (r'\bzen\s*matress\b', 'Zen Mattress'),
        (r'\btrack\s*mattress\b', 'Track Mattress'), (r'\btrack\s*matress\b', 'Track Mattress'),
    ]
    
    # Common business terms
    business_patterns = [
        (r'\border\s*i\.?d\.?\b', 'Order ID'), (r'\border\s*number\b', 'Order Number'),
        (r'\bdelivery\s*status\b', 'delivery status'), (r'\btrack\s*order\b', 'track order'),
        (r'\breturn\s*policy\b', 'return policy'), (r'\bwarranty\s*period\b', 'warranty period'),
        (r'\bcustomer\s*care\b', 'customer care'), (r'\bcustomer\s*support\b', 'customer support'),
        (r'\brefund\s*process\b', 'refund process'), (r'\bdelivery\s*charges\b', 'delivery charges'),
    ]
    
    # Apply all pattern corrections
    corrected_text = text
    all_patterns = wakefit_patterns + product_patterns + business_patterns
    
    for pattern, replacement in all_patterns:
        corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
    
    # Enhanced digit conversion (spoken numbers to digits)
    digit_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'double': '', 'triple': ''
    }
    
    def replace_spoken_digits(match):
        spoken = match.group(0).lower()
        words = spoken.split()
        digits = []
        i = 0
        while i < len(words):
            word = words[i]
            if word in ['double', 'triple'] and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in digit_map and next_word not in ['double', 'triple']:
                    repeat = 2 if word == 'double' else 3
                    digits.append(digit_map[next_word] * repeat)
                    i += 2
                    continue
            if word in digit_map and word not in ['double', 'triple']:
                digits.append(digit_map[word])
            i += 1
        
        result = ''.join(digits)
        return result if result else match.group(0)  # Fallback to original if no conversion
    
    # Apply digit conversion for sequences like "nine seven eight double six"
    digit_pattern = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|double|triple)(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|double|triple))*\b'
    corrected_text = re.sub(digit_pattern, replace_spoken_digits, corrected_text, flags=re.IGNORECASE)
    
    # Email formatting (spoken "at" and "dot")
    email_pattern = r'\b(\w+(?:\s*dot\s*\w+)*)\s+at\s+(\w+)\s+dot\s+(\w+)\b'
    def format_email(match):
        local = match.group(1).replace(' dot ', '.')
        domain = match.group(2)
        tld = match.group(3)
        return f"{local}@{domain}.{tld}"
    
    corrected_text = re.sub(email_pattern, format_email, corrected_text, flags=re.IGNORECASE)
    
    # Phone number formatting (spoken digits with spaces)
    phone_pattern = r'\b(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\b'
    def format_phone(match):
        digits = ''.join(match.groups())
        return digits  # Return as continuous number
    
    corrected_text = re.sub(phone_pattern, format_phone, corrected_text)
    
    return corrected_text


def transcribe_audio(s3_key, language_code='en-US', interaction_type='pca'):
    """
    Transcribe audio using ElevenLabs with enhanced language detection
    
    Args:
        s3_key: S3 key where audio file is stored
        language_code: Language code for transcription
        interaction_type: 'pca' for customer support calls, 'instore' for sales interactions
    
    Returns transcript messages in format: [{'role': 'user'/'agent', 'text': '...', 'timestamp': '...'}]
    """
    s3_client = _get_s3_client()
    elevenlabs = _get_elevenlabs_client()
    
    # Download audio from S3 to memory
    from io import BytesIO
    audio_buffer = BytesIO()
    s3_client.download_fileobj(S3_RECORDINGS_BUCKET, s3_key, audio_buffer)
    audio_buffer.seek(0)
    
    # Map language codes: en-US -> eng, hi-IN -> hin
    lang_map = {
        'en-US': 'eng',
        'hi-IN': 'hin'
    }
    elevenlabs_lang = lang_map.get(language_code, 'eng')
    
    # Transcribe with ElevenLabs
    transcription = elevenlabs.speech_to_text.convert(
        file=audio_buffer,
        model_id="scribe_v2",
        tag_audio_events=True,
        language_code=elevenlabs_lang,
        diarize=True,
        diarization_threshold='0.15'
    )
    
    # Parse ElevenLabs response into messages
    messages = _parse_elevenlabs_transcript(transcription, interaction_type)
    
    # Enhanced language detection after transcription
    detected_language, language_details = detect_language_improved(messages)
    print(f"[TRANSCRIBE] Language detected: {detected_language}")
    print(f"[TRANSCRIBE] Language breakdown: {language_details}")
    
    # Store language details in messages for later use
    for msg in messages:
        msg['language_details'] = language_details
    
    # Apply word corrections
    for msg in messages:
        if 'text' in msg:
            msg['text'] = _word_format(msg['text'])
    
    return messages


def detect_language_improved(transcript_messages):
    """
    Enhanced language detection for ALL Indian languages with percentage calculation.
    Returns both the primary language AND a breakdown of all detected languages.
    Percentages add up to 100% including unrecognized characters.
    """
    if not transcript_messages:
        return 'en-US', {}
    
    # Unicode ranges for different scripts (UNIQUE ranges - no overlap)
    script_ranges = {
        'english': [(ord('a'), ord('z')), (ord('A'), ord('Z')), (ord('0'), ord('9'))],
        'devanagari': [(0x0900, 0x097F)],       # Hindi, Marathi, Sanskrit
        'gujarati': [(0x0A80, 0x0AFF)],        # Gujarati
        'gurmukhi': [(0x0A00, 0x0A7F)],        # Punjabi
        'tamil': [(0x0B80, 0x0BFF)],           # Tamil
        'telugu': [(0x0C60, 0x0C7F)],          # Telugu
        'kannada': [(0x0C80, 0x0CFF)],         # Kannada
        'malayalam': [(0x0D00, 0x0D7F)],       # Malayalam
    }
    
    # Map scripts to languages
    script_to_languages = {
        'english': ['english'],
        'devanagari': ['hindi'],  # Count Devanagari as Hindi (more common)
        'gujarati': ['gujarati'],
        'gurmukhi': ['punjabi'],
        'tamil': ['tamil'],
        'telugu': ['telugu'],
        'kannada': ['kannada'],
        'malayalam': ['malayalam'],
    }
    
    char_counts = {lang: 0 for lang in ['english', 'hindi', 'marathi', 'gujarati', 'tamil', 'telugu', 'punjabi', 'kannada', 'malayalam']}
    total_chars = 0
    other_chars = 0
    
    for msg in transcript_messages:
        text = msg.get('text', '')
        total_chars += len(text)
        
        for char in text:
            char_code = ord(char)
            
            # Find which script this character belongs to
            detected_script = None
            for script, ranges in script_ranges.items():
                for start, end in ranges:
                    if start <= char_code <= end:
                        detected_script = script
                        break
                if detected_script:
                    break
            
            # Map script to language(s) and count
            if detected_script:
                languages = script_to_languages.get(detected_script, [])
                for lang in languages:
                    char_counts[lang] += 1
            else:
                # Count other characters (spaces, punctuation, symbols) as "other"
                other_chars += 1
    
    # Calculate percentages (total will now equal 100%)
    language_percentages = {}
    for lang, count in char_counts.items():
        language_percentages[lang] = round((count / total_chars * 100), 2) if total_chars > 0 else 0.0
    
    # Add "other" category if there are unrecognized characters
    if other_chars > 0:
        language_percentages['other'] = round((other_chars / total_chars * 100), 2)
    
    # Filter out languages with 0% and sort by percentage (descending)
    detected_languages = {lang: pct for lang, pct in language_percentages.items() if pct > 0}
    sorted_languages = sorted(detected_languages.items(), key=lambda x: x[1], reverse=True)
    
    # Debug logging
    print(f"[LANG-DETECT] Language breakdown: {dict(sorted_languages)}")
    
    # Map to language codes
    lang_code_map = {
        'english': 'en-US',
        'hindi': 'hi-IN',
        'gujarati': 'gu-IN',
        'marathi': 'mr-IN',
        'tamil': 'ta-IN',
        'telugu': 'te-IN',
        'punjabi': 'pa-IN',
        'kannada': 'kn-IN',
        'malayalam': 'ml-IN',
    }
    
    # Get primary language (highest percentage, excluding "other")
    primary_languages = [(lang, pct) for lang, pct in sorted_languages if lang != 'other']
    if primary_languages:
        primary_lang = primary_languages[0][0]
        primary_code = lang_code_map.get(primary_lang, 'en-US')
    else:
        primary_code = 'en-US'
    
    # Create combined language string (e.g., "hindi,english" or "kannada,gujarati")
    # Only include languages with >5% to avoid noise (exclude "other")
    significant_languages = [lang for lang, pct in sorted_languages if pct > 5 and lang != 'other']
    if not significant_languages:
        # If no significant language, use primary
        significant_languages = [primary_lang] if primary_languages else ['english']
    
    combined_languages = ','.join(significant_languages)
    
    # Return only actual languages in breakdown (exclude "other")
    language_breakdown_dict = {lang: pct for lang, pct in dict(sorted_languages).items() if lang != 'other'}
    
    return primary_code, {
        'primary_language': primary_code,
        'combined_languages': combined_languages,  # e.g., "hindi,english"
        'language_breakdown': language_breakdown_dict,  # Actual languages only
        'all_languages': language_percentages  # All including "other"
    }

def _detect_agent_speaker(temp_messages, interaction_type='pca'):
    """
    Detect agent/sales speaker(s) using LLM analysis of FULL transcript.
    
    For PCA: Returns single support_speaker
    For Instore: Returns speaker_roles mapping for all participants
    
    Args:
        temp_messages: List of message dictionaries
        interaction_type: 'pca' for customer support calls, 'instore' for sales interactions
    
    Returns: 
        - For PCA: speaker ID string of the agent
        - For Instore: dict with speaker_roles mapping and primary_sales_executive
    """
    if not temp_messages:
        if interaction_type == 'instore':
            return {'speaker_roles': {}, 'primary_sales_executive': 'speaker_0'}
        return 'speaker_0'
    
    if len(temp_messages) < 2:
        if interaction_type == 'instore':
            speaker_id = temp_messages[0].get('_raw_speaker', 'speaker_0')
            return {
                'speaker_roles': {speaker_id: 'sales_executive'},
                'primary_sales_executive': speaker_id
            }
        return temp_messages[0].get('_raw_speaker', 'speaker_0')
    
    # Use FULL transcript for accurate role detection
    raw_transcript = ""
    for msg in temp_messages:
        speaker = msg.get('_raw_speaker', 'unknown')
        text = msg.get('text', '')
        raw_transcript += f"{speaker}: {text}\n"
    
    # Call LLM for speaker detection on full transcript
    detection_result = _detect_speakers_with_llm(raw_transcript, interaction_type)
    
    if interaction_type == 'instore':
        # For instore, return mapping of all speakers and their roles
        if detection_result and 'speakers' in detection_result:
            return {
                'speaker_roles': detection_result['speakers'],
                'primary_sales_executive': detection_result.get('primary_sales_executive', 'speaker_0'),
                'confidence': detection_result.get('confidence', 'medium')
            }
        else:
            # NO FALLBACK - if LLM fails, we want to know about it
            print(f"[TRANSCRIBE] CRITICAL: Speaker detection failed for instore interaction!")
            print(f"[TRANSCRIBE] Detection result was: {detection_result}")
            # Return None to signal failure
            return None
    else:
        # For PCA, return single support speaker (backward compatible)
        if detection_result:
            support_key = 'support_speaker'
            if support_key in detection_result:
                return detection_result[support_key]
        
        # Fallback: assume second speaker is agent
        speakers = sorted(set(msg.get('_raw_speaker', '') for msg in temp_messages))
        return speakers[1] if len(speakers) > 1 else (speakers[0] if speakers else 'speaker_0')


def _parse_elevenlabs_transcript(transcription, interaction_type='pca'):
    """
    Parse ElevenLabs output into message format
    
    Args:
        transcription: ElevenLabs transcription object
        interaction_type: 'pca' for customer support calls, 'instore' for sales interactions
    
    Returns: [{'role': 'user'/'agent', 'text': '...', 'timestamp': '...', 'start_time': seconds}]
    
    ElevenLabs with diarize=True returns word-level data with speaker labels and timestamps
    """
    messages = []
    
    if not hasattr(transcription, 'words') or not transcription.words:
        full_text = getattr(transcription, 'text', '')
        if full_text:
            messages.append({
                'role': 'agent',
                'text': full_text,
                'start_time': 0.0,
                'timestamp': '00:00'
            })
        return messages
    
    # First pass: Group words by speaker
    temp_messages = []
    current_speaker = None
    current_words = []
    current_start_time = None
    current_end_time = None
    
    for word_obj in transcription.words:
        speaker = getattr(word_obj, 'speaker_id', None)
        word_text = getattr(word_obj, 'text', '')
        word_type = getattr(word_obj, 'type', 'word')
        start_time = getattr(word_obj, 'start', None)
        end_time = getattr(word_obj, 'end', None)
        
        # Skip spacing and audio events for word collection (but track timestamps)
        if word_type == 'spacing' or word_type == 'audio_event':
            if end_time is not None:
                current_end_time = end_time
            continue
        
        # If speaker changes, save the previous segment
        if speaker != current_speaker and current_words:
            # Format timestamp
            if current_start_time is not None:
                minutes = int(current_start_time // 60)
                seconds = int(current_start_time % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d}"
            else:
                timestamp_str = '00:00'
            
            temp_messages.append({
                '_raw_speaker': current_speaker,
                'text': ' '.join(current_words),
                'start_time': current_start_time if current_start_time is not None else 0.0,
                'end_time': current_end_time,
                'timestamp': timestamp_str
            })
            
            # Reset for new speaker
            current_words = []
            current_start_time = None
            current_end_time = None
        
        # Update current segment
        current_speaker = speaker
        if word_text and word_type == 'word':
            current_words.append(word_text)
        if current_start_time is None and start_time is not None:
            current_start_time = start_time
        if end_time is not None:
            current_end_time = end_time
    
    # Add the last segment
    if current_words:
        if current_start_time is not None:
            minutes = int(current_start_time // 60)
            seconds = int(current_start_time % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
        else:
            timestamp_str = '00:00'
        
        temp_messages.append({
            '_raw_speaker': current_speaker,
            'text': ' '.join(current_words),
            'start_time': current_start_time if current_start_time is not None else 0.0,
            'end_time': current_end_time,
            'timestamp': timestamp_str
        })
    
    # Second pass: Detect which speaker is the agent and assign roles
    agent_speaker_result = _detect_agent_speaker(temp_messages, interaction_type)
    
    if interaction_type == 'instore':
        # Instore: Check if detection succeeded
        if agent_speaker_result is None or not isinstance(agent_speaker_result, dict):
            # Detection failed completely - raise error
            raise ValueError(f"Speaker detection failed for instore interaction. LLM returned invalid response. Check logs for details.")
        
        # Use speaker_roles mapping
        speaker_roles = agent_speaker_result.get('speaker_roles', {})
        
        for msg in temp_messages:
            raw_speaker = msg.pop('_raw_speaker')
            role = speaker_roles.get(raw_speaker, 'other')
            msg['role'] = role
            msg['speaker_id'] = raw_speaker
            messages.append(msg)
    else:
        # PCA: Use single agent speaker (backward compatible)
        agent_speaker = agent_speaker_result if isinstance(agent_speaker_result, str) else 'speaker_0'
        
        for msg in temp_messages:
            raw_speaker = msg.pop('_raw_speaker')
            role = 'agent' if raw_speaker == agent_speaker else 'user'
            msg['role'] = role
            messages.append(msg)
    
    return messages


def save_transcript_to_s3(transcript_messages, call_id, started_at=None):
    """
    Save transcript to S3 in the format expected by pca_service
    Returns: S3 key and calculated duration in seconds
    """
    s3_client = _get_s3_client()
    
    # S3 key format: {call_id}/transcript.json
    s3_key = f"{call_id}/transcript.json"
    
    # Calculate actual call duration from transcript timestamps
    duration_seconds = 0
    if transcript_messages:
        # Get the last message's end_time or start_time
        last_msg = transcript_messages[-1]
        if 'end_time' in last_msg and last_msg['end_time']:
            duration_seconds = int(last_msg['end_time'])
        elif 'start_time' in last_msg:
            # Approximate: use start time + estimated message duration (5 seconds)
            duration_seconds = int(last_msg['start_time']) + 5
    
    # Build transcript object
    transcript = {
        'session_id': call_id,
        'started_at': started_at or datetime.now().isoformat(),
        'duration_seconds': duration_seconds,
        'messages': transcript_messages
    }
    
    # Upload to S3
    s3_client.put_object(
        Bucket=S3_RECORDINGS_BUCKET,
        Key=s3_key,
        Body=json.dumps(transcript),
        ContentType='application/json'
    )
    
    return s3_key, duration_seconds
