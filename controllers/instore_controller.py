"""
In-Store Interaction API Routes
All endpoints for in-store interaction analysis
"""
import json
import os
import sys
import uuid
import threading
from datetime import datetime, timedelta, timezone
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import instore_service
from transcribe_service import upload_audio_to_s3, transcribe_audio, save_transcript_to_s3
import instore_clickhouse as ch
from controllers.controller_utils import success_response, error_response

instore_bp = Blueprint('instore', __name__)

# In-memory store for upload progress tracking
upload_progress = {}


def update_progress(interaction_id, step, status='processing', message='', details=None):
    """Update progress for an interaction upload"""
    if interaction_id not in upload_progress:
        upload_progress[interaction_id] = {
            'interaction_id': interaction_id,
            'steps': [],
            'current_step': '',
            'status': 'processing',
            'started_at': datetime.now().isoformat()
        }
    
    upload_progress[interaction_id]['current_step'] = step
    upload_progress[interaction_id]['status'] = status
    upload_progress[interaction_id]['steps'].append({
        'step': step,
        'status': status,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'details': details or {}
    })
    
    print(f"[InStore] {message}")


def process_upload_async(file_data, interaction_id, customer_name, store_id, sales_executive_id, 
                         notes, original_filename, audio_size):
    """Background task to process the in-store interaction upload"""
    try:
        upload_started_at = datetime.now()
        
        # Upload audio to S3
        update_progress(interaction_id, 'upload_audio', 'processing', 
                       'Uploading audio for in-store interaction')
        
        file_obj = BytesIO(file_data)
        file_obj.name = original_filename
        recording_s3_key = upload_audio_to_s3(file_obj, interaction_id)
        
        update_progress(interaction_id, 'upload_audio', 'completed', 
                       'Audio uploaded')
        
        # Transcribe audio
        update_progress(interaction_id, 'transcribe_audio', 'processing',
                       'Transcribing audio')
        
        transcript_messages = transcribe_audio(recording_s3_key, 'en-US')
        
        update_progress(interaction_id, 'transcribe_audio', 'completed',
                       'Transcription completed')
        
        # Save transcript to S3
        update_progress(interaction_id, 'save_transcript', 'processing',
                       'Saving transcript')
        
        transcript_s3_key, actual_duration = save_transcript_to_s3(
            transcript_messages, 
            interaction_id, 
            upload_started_at.isoformat()
        )
        
        update_progress(interaction_id, 'save_transcript', 'completed',
                       'Transcript saved')
        
        # Calculate timestamps
        started_at = upload_started_at
        ended_at = started_at + timedelta(seconds=actual_duration) if actual_duration > 0 else started_at
        
        # Detect language
        detected_language = 'en-US'
        transcript_text = ' '.join([msg.get('text', '') for msg in transcript_messages])
        if any('\u0900' <= char <= '\u097F' for char in transcript_text):
            detected_language = 'hi-IN'
        
        # Create interaction record
        update_progress(interaction_id, 'create_record', 'processing',
                       'Creating interaction record')
        
        record = ch.InstoreRecord(
            interaction_id=interaction_id,
            store_id=store_id or '',
            sales_executive_id=sales_executive_id or '',
            customer_name=customer_name or 'Unknown',
            status='completed',
            language=detected_language,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=actual_duration,
            transcript_s3_key=transcript_s3_key,
            recording_s3_key=recording_s3_key,
            audio_size=audio_size,
            uploaded_filename=original_filename,
            notes=notes or '',
            created_on=datetime.now(timezone.utc)
        )
        
        ch.upsert_record(record)
        
        update_progress(interaction_id, 'create_record', 'completed',
                       'Interaction record created')
        
        # Analyze interaction
        update_progress(interaction_id, 'analyze_interaction', 'processing',
                       'Analyzing interaction')
        
        analysis = instore_service.analyze_instore_interaction(transcript_messages, interaction_id)
        
        if analysis:
            # Extract matrices from analysis
            matrices = analysis.get('interaction_matrices', {})
            
            analytics = ch.InstoreAnalytics(
                interaction_id=interaction_id,
                overall_sentiment=instore_service._to_score(analysis.get('overall_sentiment')),
                customer_satisfaction=instore_service._to_score(analysis.get('customer_satisfaction')),
                sales_executive_performance=instore_service._to_score(analysis.get('sales_executive_performance')),
                summary=analysis.get('summary', ''),
                topics=analysis.get('topics', []),
                action_items=analysis.get('action_items', []),
                key_indicators=analysis.get('key_indicators', []),
                customer_name=analysis.get('customer_name') or customer_name,
                interaction_outcome=analysis.get('interaction_outcome'),
                interaction_code=matrices.get('interaction_code'),
                category=matrices.get('primary_category'),
                sub_category=matrices.get('sub_category'),
                product=matrices.get('product'),
                sales_outcome=matrices.get('overall_sales_outcome'),
                l1_pillow=matrices.get('l1_pillow'),
                l2_pillow=matrices.get('l2_pillow'),
                l3_pillow=matrices.get('l3_pillow'),
                raw_model_response=analysis,
                model_id=os.environ.get('PCA_MODEL_ID', 'claude-haiku-4.5'),
                analyzed_at=datetime.now(timezone.utc)
            )
            
            ch.upsert_analytics(analytics)
        
        update_progress(interaction_id, 'complete', 'completed',
                       'Processing complete')
        
    except Exception as e:
        update_progress(interaction_id, 'error', 'failed',
                       f'Upload failed: {str(e)}')
        import traceback
        traceback.print_exc()


# ── POST /api/instore/uploads ─────────────────────────────────────────────────

@instore_bp.route('/instore/uploads', methods=['POST'])
def upload_interaction():
    """
    Upload audio file + metadata for in-store interaction
    Frontend: POST /api/instore/uploads (multipart)
    """
    try:
        # Get file
        if 'file' not in request.files:
            return error_response('No file provided', 400)
        
        file = request.files['file']
        if file.filename == '':
            return error_response('Empty filename', 400)
        
        # Get metadata
        customer_name = request.form.get('customerName', '').strip()
        store_id = request.form.get('storeId', '').strip()
        sales_executive_id = request.form.get('salesExecutiveId', '').strip()
        notes = request.form.get('notes', '').strip()
        original_filename = secure_filename(file.filename)
        
        # Get file size and content
        file.seek(0, os.SEEK_END)
        audio_size = file.tell()
        file.seek(0)
        file_data = file.read()
        
        # Generate interaction ID
        interaction_id = f"instore-{uuid.uuid4().hex[:12]}"
        
        # Initialize progress
        update_progress(interaction_id, 'initiated', 'processing', 
                       'Upload started')
        
        # Start background processing
        thread = threading.Thread(
            target=process_upload_async,
            args=(file_data, interaction_id, customer_name, store_id, 
                  sales_executive_id, notes, original_filename, audio_size)
        )
        thread.daemon = True
        thread.start()
        
        # Return immediately
        return success_response('Upload started, processing in background', {
            'interactionId': interaction_id,
            'message': 'Poll /api/instore/interactions/{interactionId}/processing for status'
        })
        
    except Exception as e:
        print(f"[InStore] Upload initiation failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f'Upload failed: {str(e)}', 500)


# ── GET /api/instore/interactions ────────────────────────────────────────────

@instore_bp.route('/instore/interactions', methods=['GET'])
def list_interactions():
    """
    List all in-store interactions with pagination
    Frontend: GET /api/instore/interactions?page=0&limit=10&storeId=...
    """
    try:
        page = int(request.args.get('page', 0))
        limit = int(request.args.get('limit', 10))
        store_id = request.args.get('storeId') or request.args.get('store_id')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Parse dates if provided
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        
        records, total_count = ch.query_records(store_id, start_dt, end_dt, page, limit)
        
        # Get analytics for all records
        interaction_ids = [r.interaction_id for r in records]
        analytics_map = ch.get_analytics_map(interaction_ids)
        
        # Format response
        interactions = [r.to_log_dict(analytics_map.get(r.interaction_id)) for r in records]
        
        total_pages = (total_count + limit - 1) // limit if limit else 1
        
        return success_response('Interactions retrieved', {
            'interactions': interactions,
            'total_count': total_count,
            'total_pages': total_pages
        })
        
    except Exception as e:
        print(f"[InStore] List interactions failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f'Failed to list interactions: {str(e)}', 500)


# ── GET /api/instore/interactions/{interactionId} ───────────────────────────

@instore_bp.route('/instore/interactions/<string:interaction_id>', methods=['GET'])
def get_interaction(interaction_id):
    """
    Get full interaction details with analytics
    Frontend: GET /api/instore/interactions/{interactionId}
    """
    try:
        record = ch.get_record(interaction_id)
        if not record:
            return error_response('Interaction not found', 404)
        
        analytics = ch.get_analytics(interaction_id)
        
        # Load transcript from S3
        import boto3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'ap-south-1')
        )
        
        transcript_messages = []
        if record.transcript_s3_key:
            try:
                obj = s3_client.get_object(
                    Bucket=os.environ.get('S3_RECORDINGS_BUCKET'),
                    Key=record.transcript_s3_key
                )
                transcript_data = json.loads(obj['Body'].read())
                transcript_messages = transcript_data.get('messages', [])
            except Exception as e:
                print(f"[InStore] Failed to load transcript: {e}")
        
        # Format transcript for frontend
        transcript = []
        for msg in transcript_messages:
            role = msg.get('role')
            speaker_type = 'user' if role == 'user' else 'agent'
            
            if 'timestamp' in msg and isinstance(msg['timestamp'], str) and ':' in msg['timestamp']:
                stamp = msg['timestamp']
            elif 'start_time' in msg:
                start_seconds = float(msg['start_time'])
                minutes = int(start_seconds // 60)
                seconds = int(start_seconds % 60)
                stamp = f"{minutes:02d}:{seconds:02d}"
            else:
                stamp = "00:00"
            
            transcript.append({
                "speaker": "Customer" if speaker_type == "user" else "Sales Executive",
                "timestamp": stamp,
                "content": msg.get("text", ""),
                "type": speaker_type
            })
        
        # Presign recording URL
        recording_url = None
        if record.recording_s3_key:
            try:
                recording_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': os.environ.get('S3_RECORDINGS_BUCKET'),
                        'Key': record.recording_s3_key
                    },
                    ExpiresIn=3600
                )
            except Exception as e:
                print(f"[InStore] Failed to presign recording: {e}")
        
        # Convert times to IST
        def to_ist(dt):
            if not dt:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ist_offset = timedelta(hours=5, minutes=30)
            ist_dt = dt.astimezone(timezone.utc) + ist_offset
            return ist_dt.strftime("%d/%m/%Y, %H:%M:%S")
        
        audio_size_mb = None
        if record.audio_size:
            audio_size_mb = round(record.audio_size / (1024 * 1024), 2)
        
        data = {
            'interactionId': interaction_id,
            'customerName': record.customer_name or 'Unknown',
            'storeId': record.store_id,
            'salesExecutiveId': record.sales_executive_id,
            'uploadedFile': record.uploaded_filename or '—',
            'uploadedAt': to_ist(record.created_on),
            'audioSize': audio_size_mb,
            'notes': record.notes or '',
            'language': record.language or '—',
            'interactionDuration': record._format_duration(),
            'status': record.status,
            'interactionStart': to_ist(record.started_at),
            'interactionEnd': to_ist(record.ended_at),
            'transcript': transcript,
            'recordingUrl': recording_url,
            'summary': '',
            'topics': [],
            'actionItems': [],
            'productsDiscussed': [],
            'matrices': {},
            'sentiment': {
                'overallSentiment': 0,
                'customerSatisfaction': 0,
                'salesExecutivePerformance': 0,
                'keyIndicators': []
            }
        }
        
        if analytics:
            data['summary'] = analytics.summary or ''
            data['topics'] = analytics.topics or []
            data['actionItems'] = analytics.action_items or []
            data['matrices'] = analytics.to_dict()
            data['sentiment'] = {
                'overallSentiment': float(analytics.overall_sentiment) if analytics.overall_sentiment is not None else 0,
                'customerSatisfaction': float(analytics.customer_satisfaction) if analytics.customer_satisfaction is not None else 0,
                'salesExecutivePerformance': float(analytics.sales_executive_performance) if analytics.sales_executive_performance is not None else 0,
                'keyIndicators': analytics.key_indicators or []
            }
            if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
                data['productsDiscussed'] = analytics.raw_model_response.get('products_discussed', [])
                learning_suggestions = analytics.raw_model_response.get('learning_suggestions')
                data['learningSuggestions'] = learning_suggestions if learning_suggestions else ""
                competitor_intelligence = analytics.raw_model_response.get('competitor_intelligence', [])
                data['competitorIntelligence'] = competitor_intelligence if competitor_intelligence else []
                interaction_matrices = analytics.raw_model_response.get('interaction_matrices', {})
                data['interactionMatrices'] = interaction_matrices if interaction_matrices else {}
        
        return success_response('Interaction details retrieved', data)
        
    except Exception as e:
        print(f"[InStore] Get interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f'Failed to get interaction: {str(e)}', 500)


# ── DELETE /api/instore/interactions/{interactionId} ────────────────────────

@instore_bp.route('/instore/interactions/<string:interaction_id>', methods=['DELETE'])
def delete_interaction(interaction_id):
    """
    Delete interaction record, analytics, and S3 files
    Frontend: DELETE /api/instore/interactions/{interactionId}
    """
    try:
        record = ch.get_record(interaction_id)
        if not record:
            return error_response('Interaction not found', 404)
        
        # Delete from S3
        import boto3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'ap-south-1')
        )
        
        s3_keys_to_delete = []
        if record.transcript_s3_key:
            s3_keys_to_delete.append({'Key': record.transcript_s3_key})
        if record.recording_s3_key:
            s3_keys_to_delete.append({'Key': record.recording_s3_key})
        
        if s3_keys_to_delete:
            try:
                s3_client.delete_objects(
                    Bucket=os.environ.get('S3_RECORDINGS_BUCKET'),
                    Delete={'Objects': s3_keys_to_delete}
                )
                print(f"[InStore] Deleted {len(s3_keys_to_delete)} S3 objects")
            except Exception as e:
                print(f"[InStore] Failed to delete S3 objects: {e}")
        
        # Delete from ClickHouse
        success = ch.delete_interaction(interaction_id)
        
        if not success:
            return error_response('Failed to delete interaction', 500)
        
        return success_response('Interaction deleted successfully', {
            'interactionId': interaction_id,
            'deleted': True
        })
        
    except Exception as e:
        print(f"[InStore] Delete interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f'Failed to delete interaction: {str(e)}', 500)


# ── GET /api/instore/interactions/{interactionId}/processing ────────────────

@instore_bp.route('/instore/interactions/<string:interaction_id>/processing', methods=['GET'])
def get_processing_status(interaction_id):
    """
    Get processing status for an in-store interaction
    Frontend: GET /api/instore/interactions/{interactionId}/processing
    """
    try:
        if interaction_id in upload_progress:
            progress_data = upload_progress[interaction_id]
            
            step_map = {
                'upload_audio': {'key': 'upload', 'label': 'Uploading audio'},
                'transcribe_audio': {'key': 'transcribe', 'label': 'Transcribing audio'},
                'save_transcript': {'key': 'diarize', 'label': 'Diarizing speakers'},
                'analyze_interaction': {'key': 'analyze', 'label': 'Analyzing interaction'},
                'complete': {'key': 'complete', 'label': 'Complete'}
            }
            
            steps = []
            current_step_index = 0
            all_steps = ['upload_audio', 'transcribe_audio', 'save_transcript', 'analyze_interaction', 'complete']
            
            for idx, step_name in enumerate(all_steps):
                step_info = step_map.get(step_name, {'key': step_name, 'label': step_name})
                step_data = next((s for s in progress_data['steps'] if s['step'] == step_name), None)
                
                if step_data:
                    if step_data['status'] == 'completed':
                        step_info['state'] = 'done'
                        step_info['message'] = step_data.get('message', '')
                        current_step_index = idx + 1
                    elif step_data['status'] == 'processing':
                        step_info['state'] = 'running'
                        step_info['message'] = step_data.get('message', '')
                        current_step_index = idx
                    elif step_data['status'] == 'failed':
                        step_info['state'] = 'error'
                        step_info['message'] = step_data.get('message', '')
                else:
                    step_info['state'] = 'pending'
                
                steps.append(step_info)
            
            return success_response('Processing status retrieved', {
                'status': progress_data['status'],
                'currentStepIndex': current_step_index,
                'currentStep': progress_data['current_step'],
                'steps': steps,
                'logs': progress_data['steps']
            })
        
        # Fallback: check database
        record = ch.get_record(interaction_id)
        if not record:
            return error_response('Interaction not found', 404)
        
        analytics = ch.get_analytics(interaction_id)
        
        status = 'completed' if analytics else 'processing'
        current_step = 5 if analytics else 3
        
        steps = [
            {'key': 'upload', 'label': 'Uploading audio', 'state': 'done'},
            {'key': 'transcribe', 'label': 'Transcribing audio', 'state': 'done'},
            {'key': 'diarize', 'label': 'Diarizing speakers', 'state': 'done'},
            {'key': 'analyze', 'label': 'Analyzing interaction', 'state': 'done' if analytics else 'running'},
            {'key': 'complete', 'label': 'Complete', 'state': 'done' if analytics else 'pending'}
        ]
        
        return success_response('Processing status retrieved', {
            'status': status,
            'currentStepIndex': current_step,
            'steps': steps
        })
        
    except Exception as e:
        print(f"[InStore] Get processing status failed: {e}")
        return error_response(f'Failed to get status: {str(e)}', 500)
