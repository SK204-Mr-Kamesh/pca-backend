"""
Audio transcription service using AWS Transcribe
"""
import os
import json
import time
import uuid
import boto3
from datetime import datetime

AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')
S3_RECORDINGS_BUCKET = os.environ.get('S3_RECORDINGS_BUCKET', 'sahaa-voiceai-recordings')


def _get_aws_clients():
    """Get AWS clients for Transcribe and S3"""
    return {
        's3': boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=AWS_REGION
        ),
        'transcribe': boto3.client(
            'transcribe',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=AWS_REGION
        )
    }


def upload_audio_to_s3(audio_file, call_id):
    """Upload audio file to S3 and return S3 key"""
    clients = _get_aws_clients()
    s3_client = clients['s3']
    
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


def transcribe_audio(s3_key, language_code='en-US'):
    """
    Transcribe audio using AWS Transcribe
    Returns transcript messages in format: [{'role': 'user'/'agent', 'text': '...', 'timestamp': '...'}]
    """
    clients = _get_aws_clients()
    transcribe_client = clients['transcribe']
    s3_client = clients['s3']
    
    # Create unique job name
    job_name = f"pca-{uuid.uuid4().hex[:8]}-{int(time.time())}"
    
    # S3 URI
    audio_uri = f"s3://{S3_RECORDINGS_BUCKET}/{s3_key}"
    
    # Start transcription job
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': audio_uri},
        MediaFormat='wav',
        LanguageCode=language_code,
        Settings={
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 2
        }
    )
    
    # Wait for completion (poll every 5 seconds, max 5 minutes)
    max_attempts = 60
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        time.sleep(5)
        
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']
        
        if job_status == 'COMPLETED':
            # Get transcript URI
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            
            # Download transcript directly from the URI (it's a presigned URL)
            import requests
            transcript_response = requests.get(transcript_uri)
            transcript_data = transcript_response.json()
            
            # Parse transcript into messages
            messages = _parse_transcript(transcript_data)
            
            # Clean up transcription job
            try:
                transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
            except:
                pass
            
            return messages
        
        elif job_status == 'FAILED':
            raise Exception(f"Transcription failed: {status['TranscriptionJob'].get('FailureReason', 'Unknown')}")
    
    raise Exception("Transcription timed out")


def _parse_transcript(transcript_data):
    """
    Parse AWS Transcribe output into message format
    Returns: [{'role': 'user'/'agent', 'text': '...', 'timestamp': '...'}]
    """
    messages = []
    
    # Get speaker segments
    segments = transcript_data.get('results', {}).get('speaker_labels', {}).get('segments', [])
    items = transcript_data.get('results', {}).get('items', [])
    
    if not segments:
        # No speaker labels, treat all as single speaker
        full_text = transcript_data.get('results', {}).get('transcripts', [{}])[0].get('transcript', '')
        if full_text:
            messages.append({
                'role': 'user',
                'text': full_text,
                'timestamp': datetime.now().isoformat()
            })
        return messages
    
    # Build word lookup
    word_lookup = {}
    for item in items:
        if item['type'] == 'pronunciation':
            start_time = float(item['start_time'])
            word_lookup[start_time] = item['alternatives'][0]['content']
    
    # Build messages from segments
    for segment in segments:
        speaker_label = segment.get('speaker_label', 'spk_0')
        # Assume spk_0 is customer (user), spk_1 is agent
        role = 'user' if speaker_label == 'spk_0' else 'agent'
        
        # Get words for this segment
        segment_items = segment.get('items', [])
        words = []
        start_time = None
        
        for seg_item in segment_items:
            item_start = float(seg_item['start_time'])
            if start_time is None:
                start_time = item_start
            word = word_lookup.get(item_start, '')
            if word:
                words.append(word)
        
        if words:
            messages.append({
                'role': role,
                'text': ' '.join(words),
                'timestamp': datetime.fromtimestamp(start_time).isoformat() if start_time else datetime.now().isoformat()
            })
    
    return messages


def save_transcript_to_s3(transcript_messages, call_id, started_at=None):
    """
    Save transcript to S3 in the format expected by pca_service
    Returns: S3 key
    """
    clients = _get_aws_clients()
    s3_client = clients['s3']
    
    # S3 key format: {call_id}/transcript.json
    s3_key = f"{call_id}/transcript.json"
    
    # Build transcript object
    transcript = {
        'session_id': call_id,
        'started_at': started_at or datetime.now().isoformat(),
        'messages': transcript_messages
    }
    
    # Upload to S3
    s3_client.put_object(
        Bucket=S3_RECORDINGS_BUCKET,
        Key=s3_key,
        Body=json.dumps(transcript),
        ContentType='application/json'
    )
    
    return s3_key
