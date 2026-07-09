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
    """Get ElevenLabs client with regional support"""
    return ElevenLabs(
        api_key=ELEVENLABS_API_KEY,
        base_url=ELEVENLABS_BASE_URL
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
    if not text:
        return text
    
    import re
    
    patterns = [
        (r'\bwiprit\b', 'Wakefit'),
        (r'\brakefirt\b', 'Wakefit'),
        (r'\bwakefeet\b', 'Wakefit'),
        (r'\bwakefeat\b', 'Wakefit'),
        (r'\bwakepit\b', 'Wakefit'),
        (r'\bwikfit\b', 'Wakefit'),
        (r'\bwikfeet\b', 'Wakefit'),
        (r'\bvakefit\b', 'Wakefit'),
        (r'\bvakfit\b', 'Wakefit'),
        (r'\bweakfit\b', 'Wakefit'),
        (r'\bwekfit\b', 'Wakefit'),
        (r'\bwak\s*fit\b', 'Wakefit'),
        (r'\bwake\s*fit\b', 'Wakefit'),
        (r'\bwik\s*fit\b', 'Wakefit'),
        (r'\bvake\s*fit\b', 'Wakefit'),
        (r'\bwak\s*feet\b', 'Wakefit'),
        (r'\bwake\s*feet\b', 'Wakefit'),
        (r'\bwak\s*pit\b', 'Wakefit'),
        (r'\bwake\s*pit\b', 'Wakefit'),
        (r'\brake\s*fit\b', 'Wakefit'),
        (r'\brake\s*firt\b', 'Wakefit'),
        (r'\brak\s*fit\b', 'Wakefit'),
        (r'\brak\s*feet\b', 'Wakefit'),
        (r'\b[wrv][aei]k?[ec]?\s*(?:fit|pit|feet|firt|feit|feat|fert)\b', 'Wakefit'),
        (r'\bwakefit\b', 'Wakefit'),
    ]
    
    corrected_text = text
    for pattern, replacement in patterns:
        corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
    
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
        return result
    
    digit_pattern = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|double|triple)(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|double|triple))*\b'
    corrected_text = re.sub(digit_pattern, replace_spoken_digits, corrected_text, flags=re.IGNORECASE)
    
    email_pattern = r'\b(\w+(?:\s*dot\s*\w+)*)\s+at\s+(\w+)\s+dot\s+(\w+)\b'
    def format_email(match):
        local = match.group(1).replace(' dot ', '.')
        domain = match.group(2)
        tld = match.group(3)
        return f"{local}@{domain}.{tld}"
    
    corrected_text = re.sub(email_pattern, format_email, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text


def transcribe_audio(s3_key, language_code='en-US'):
    """
    Transcribe audio using ElevenLabs
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
    
    for msg in messages:
        if 'text' in msg:
            msg['text'] = _word_format(msg['text'])
    
    return messages

def _detect_agent_speaker(temp_messages):
    """
    Intelligently detect which speaker is the Customer Support agent
    Returns: 'speaker_0' or 'speaker_1' (whichever is the agent)
    """
    if not temp_messages or len(temp_messages) < 2:
        return 'speaker_0'  # Default fallback
    
    speaker_0_score = 0
    speaker_1_score = 0
    
    # Analyze first 5 messages for patterns
    for msg in temp_messages[:5]:
        text = msg.get('text', '').lower()
        speaker = msg.get('_raw_speaker', '')
        
        if not text or not speaker:
            continue
        
        # Agent indicators
        agent_keywords = [
            'wakefit', 'wake fit',
            'good morning', 'good evening', 'good afternoon',
            'how can i help', 'how may i help', 'how can i assist',
            'call back request', 'raised a request',
            'let me check', 'let me look into', 'just give me a moment',
            'sir', 'ma\'am', 'mister',
            'thank you for calling', 'thanks for calling'
        ]
        
        # Customer indicators
        customer_keywords = [
            'i am waiting', 'i\'m waiting', 'i have been waiting',
            'not delivered', 'didn\'t receive', 'haven\'t received',
            'my order', 'i ordered', 'i bought',
            'i need', 'i want', 'i require'
        ]
        
        agent_count = sum(1 for keyword in agent_keywords if keyword in text)
        customer_count = sum(1 for keyword in customer_keywords if keyword in text)
        
        if speaker == 'speaker_0':
            speaker_0_score += agent_count - customer_count
        elif speaker == 'speaker_1':
            speaker_1_score += agent_count - customer_count
    
    # Higher score = more likely to be agent
    if speaker_0_score > speaker_1_score:
        return 'speaker_0'
    elif speaker_1_score > speaker_0_score:
        return 'speaker_1'
    else:
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
