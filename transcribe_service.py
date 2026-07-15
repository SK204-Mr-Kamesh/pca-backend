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


def transcribe_audio(s3_key, language_code='en-US'):
    """
    Transcribe audio using ElevenLabs with enhanced language detection
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
        diarize=True
    )
    
    # Parse ElevenLabs response into messages
    messages = _parse_elevenlabs_transcript(transcription)
    
    # Enhanced language detection after transcription
    detected_language = detect_language_improved(messages)
    print(f"[TRANSCRIBE] Language detected: {detected_language}")
    
    # Apply word corrections
    for msg in messages:
        if 'text' in msg:
            msg['text'] = _word_format(msg['text'])
    
    return messages


def detect_language_improved(transcript_messages):
    """
    Enhanced language detection using multiple strategies:
    1. Character set analysis (Hindi unicode blocks)
    2. Word frequency (common Hindi/English words)  
    3. Mixed language detection (Hinglish)
    """
    if not transcript_messages:
        return 'en-US'
    
    english_count = 0
    hindi_count = 0
    total_chars = 0
    total_words = 0
    
    # Common Hindi words for detection
    hindi_words = {
        'हाँ', 'नहीं', 'क्या', 'कैसे', 'कब', 'कहाँ', 'जी', 'सर', 'मैम',
        'ठीक', 'अच्छा', 'बुरा', 'समस्या', 'आर्डर', 'डिलीवरी', 'पैसा', 'रुपया',
        'हेलो', 'नमस्ते', 'धन्यवाद', 'शुक्रिया', 'माफ', 'क्षमा'
    }
    
    # Common English words 
    english_words = {
        'hello', 'hi', 'yes', 'no', 'thank', 'thanks', 'sorry', 'please',
        'order', 'delivery', 'wakefit', 'mattress', 'customer', 'support',
        'help', 'problem', 'issue', 'money', 'refund', 'good', 'bad'
    }
    
    hindi_word_count = 0
    english_word_count = 0
    
    for msg in transcript_messages:
        text = msg.get('text', '')
        if not text:
            continue
            
        total_chars += len(text)
        words = text.lower().split()
        total_words += len(words)
        
        # Count Hindi characters (Devanagari script)
        hindi_count += sum(1 for char in text if '\u0900' <= char <= '\u097F')
        
        # Count English characters
        english_count += sum(1 for char in text if 'a' <= char.lower() <= 'z')
        
        # Count language-specific words
        for word in words:
            word_clean = word.strip('.,!?():;')
            if word_clean in hindi_words:
                hindi_word_count += 1
            elif word_clean in english_words:
                english_word_count += 1
    
    if total_chars == 0:
        return 'en-US'
    
    # Calculate percentages
    hindi_char_pct = (hindi_count / total_chars) * 100 if total_chars > 0 else 0
    english_char_pct = (english_count / total_chars) * 100 if total_chars > 0 else 0
    hindi_word_pct = (hindi_word_count / total_words) * 100 if total_words > 0 else 0
    english_word_pct = (english_word_count / total_words) * 100 if total_words > 0 else 0
    
    # Debug logging
    print(f"[LANG-DETECT] Hindi chars: {hindi_char_pct:.1f}%, English chars: {english_char_pct:.1f}%")
    print(f"[LANG-DETECT] Hindi words: {hindi_word_pct:.1f}%, English words: {english_word_pct:.1f}%")
    
    # Classification logic
    if hindi_char_pct > 30 or hindi_word_pct > 20:
        return 'hi-IN'  # Predominantly Hindi
    elif hindi_char_pct > 10 and english_char_pct > 30:
        return 'hi-IN'  # Hinglish (code-switching, treat as Hindi)
    elif hindi_word_pct > 5 and english_word_pct > 10:
        return 'hi-IN'  # Mixed conversation with Hindi elements
    else:
        return 'en-US'  # Predominantly English

def _detect_agent_speaker(temp_messages):
    """
    Enhanced agent detection using 15 messages with weighted scoring system
    Returns: 'speaker_0' or 'speaker_1' (whichever is the agent)
    """
    if not temp_messages or len(temp_messages) < 2:
        return 'speaker_0'  # Default fallback
    
    speaker_0_score = 0
    speaker_1_score = 0
    
    # Analyze first 15 messages (or all if fewer available)
    analysis_count = min(15, len(temp_messages))
    
    for i, msg in enumerate(temp_messages[:analysis_count]):
        text = msg.get('text', '').lower()
        speaker = msg.get('_raw_speaker', '')
        
        if not text or not speaker:
            continue
        
        # STRONG agent indicators (weight: 5 points each)
        strong_agent_keywords = [
            'wakefit', 'wake fit', 'good morning wakefit', 'hello wakefit',
            'thank you for calling wakefit', 'welcome to wakefit',
            'this is wakefit customer support', 'wakefit customer care',
            'let me check your order', 'i can see your order here',
            'let me look into this for you', 'i will help you with this'
        ]
        
        # MODERATE agent indicators (weight: 3 points each)
        moderate_agent_keywords = [
            'good morning', 'good evening', 'good afternoon', 'hello sir', 'hello ma\'am',
            'how can i help you', 'how may i assist you', 'how can i assist',
            'let me check', 'let me look into', 'just give me a moment',
            'sir', 'ma\'am', 'mister', 'madam',
            'thank you for calling', 'thanks for calling',
            'may i have your order number', 'can i get your order id',
            'let me transfer you', 'i will escalate this'
        ]
        
        # WEAK agent indicators (weight: 1 point each)
        weak_agent_keywords = [
            'our company', 'our product', 'our policy', 'our team',
            'we will', 'we can', 'i can help', 'i will help',
            'please hold', 'one moment please', 'bear with me'
        ]
        
        # STRONG customer indicators (weight: -5 points each)
        strong_customer_keywords = [
            'i ordered', 'i bought', 'i purchased', 'my order',
            'i am waiting for', 'i have been waiting', 'where is my order',
            'i want my refund', 'i need my money back', 'this is unacceptable',
            'i will complain', 'i want to speak to your manager'
        ]
        
        # MODERATE customer indicators (weight: -3 points each)
        moderate_customer_keywords = [
            'not delivered', 'didn\'t receive', 'haven\'t received', 'not got',
            'i need', 'i want', 'i require', 'my problem is',
            'i called yesterday', 'i have called before', 'nobody helped me'
        ]
        
        # Calculate weighted scores
        strong_agent = sum(5 for keyword in strong_agent_keywords if keyword in text)
        moderate_agent = sum(3 for keyword in moderate_agent_keywords if keyword in text)
        weak_agent = sum(1 for keyword in weak_agent_keywords if keyword in text)
        
        strong_customer = sum(-5 for keyword in strong_customer_keywords if keyword in text)
        moderate_customer = sum(-3 for keyword in moderate_customer_keywords if keyword in text)
        
        total_score = strong_agent + moderate_agent + weak_agent + strong_customer + moderate_customer
        
        # First speaker bonus (agents typically greet first)
        if i == 0:
            total_score += 3
            
        # Greeting pattern bonus (if first message contains greeting)
        if i == 0 and any(greet in text for greet in ['good morning', 'hello', 'hi', 'welcome']):
            total_score += 5
        
        # Question pattern analysis (agents ask more questions)
        question_count = text.count('?') + sum(1 for q in ['what is', 'when did', 'how can', 'do you', 'have you', 'can you', 'may i'] if q in text)
        if question_count > 0:
            total_score += question_count * 2
            
        # Apply score to appropriate speaker
        if speaker == 'speaker_0':
            speaker_0_score += total_score
        elif speaker == 'speaker_1':
            speaker_1_score += total_score
    
    # Final decision with confidence logging
    if speaker_0_score > speaker_1_score:
        confidence = speaker_0_score - speaker_1_score
        print(f"[TRANSCRIBE] Agent detected as speaker_0 (confidence: {confidence})")
        return 'speaker_0'
    elif speaker_1_score > speaker_0_score:
        confidence = speaker_1_score - speaker_0_score  
        print(f"[TRANSCRIBE] Agent detected as speaker_1 (confidence: {confidence})")
        return 'speaker_1'
    else:
        print(f"[TRANSCRIBE] Equal scores, defaulting to speaker_0")
        return 'speaker_0'  # Default fallback


def _parse_elevenlabs_transcript(transcription):
    """
    Parse ElevenLabs output into message format
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
    
    # Second pass: Detect which speaker is the agent
    agent_speaker = _detect_agent_speaker(temp_messages)
    
    # Third pass: Assign correct roles
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
