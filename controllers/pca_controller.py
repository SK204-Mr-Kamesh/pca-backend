"""
PCA API Routes
All endpoints for standalone PCA backend
"""
import os
import sys
import uuid
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import pca_service
from transcribe_service import upload_audio_to_s3, transcribe_audio, save_transcript_to_s3
import pca_clickhouse as ch
from controllers.controller_utils import success_response, error_response

pca_bp = Blueprint('pca', __name__)

PCA_INGEST_SECRET = os.environ.get('PCA_INGEST_SECRET', '')

# In-memory store for upload progress tracking
upload_progress = {}


def update_progress(call_id, step, status='processing', message='', details=None):
    """Update progress for a call upload"""
    if call_id not in upload_progress:
        upload_progress[call_id] = {
            'call_id': call_id,
            'steps': [],
            'current_step': '',
            'status': 'processing',
            'started_at': datetime.now().isoformat()
        }
    
    upload_progress[call_id]['current_step'] = step
    upload_progress[call_id]['status'] = status
    upload_progress[call_id]['steps'].append({
        'step': step,
        'status': status,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'details': details or {}
    })
    
    print(f"[PCA] {message}")


def process_upload_async(file_data, call_id, caller_name, notes, original_filename, audio_size):
    """Background task to process the upload"""
    try:
        from datetime import timedelta
        from io import BytesIO
        
        upload_started_at = datetime.now()
        
        # Upload audio to S3
        update_progress(call_id, 'upload_audio', 'processing', 
                       f"Uploading audio")
        
        # Create file object from bytes
        file_obj = BytesIO(file_data)
        file_obj.name = original_filename
        recording_s3_key = upload_audio_to_s3(file_obj, call_id)
        
        update_progress(call_id, 'upload_audio', 'completed', 
                       f"Audio uploaded")
        
        # Transcribe audio with automatic language detection (default to English)
        update_progress(call_id, 'transcribe_audio', 'processing',
                       f"Transcribing audio")
        
        # Use English as default, transcription will detect Hindi if present
        transcript_messages = transcribe_audio(recording_s3_key, 'en-US')
        
        update_progress(call_id, 'transcribe_audio', 'completed',
                       f"Transcription completed")
        
        # Save transcript to S3
        update_progress(call_id, 'save_transcript', 'processing',
                       f"Saving transcript")
        
        transcript_s3_key, actual_duration = save_transcript_to_s3(
            transcript_messages, 
            call_id, 
            upload_started_at.isoformat()
        )
        
        update_progress(call_id, 'save_transcript', 'completed',
                       f"Transcript saved")
        
        # Calculate timestamps
        started_at = upload_started_at
        ended_at = upload_started_at
        if actual_duration > 0:
            ended_at = started_at + timedelta(seconds=actual_duration)
        
        # Create call record and analyze
        update_progress(call_id, 'analyze_call', 'processing',
                       f"Analyzing call")
        
        # Detect language from transcript
        detected_language = 'en-US'  # Default
        # Simple detection: if transcript has Hindi unicode, mark as hi-IN
        transcript_text = ' '.join([msg.get('text', '') for msg in transcript_messages])
        if any('\u0900' <= char <= '\u097F' for char in transcript_text):
            detected_language = 'hi-IN'
        
        payload = {
            'call_id': call_id,
            'session_id': call_id,
            'from_phone': caller_name or 'Unknown',
            'to_phone': 'Unknown',
            'language': detected_language,
            'started_at': started_at.isoformat(),
            'ended_at': ended_at.isoformat(),
            'duration_seconds': actual_duration,
            'status': 'answered',
            'call_source': 'upload',
            'transcript_key': transcript_s3_key,
            'recording_key': recording_s3_key,
            'audio_size': audio_size,
            'uploaded_filename': original_filename,
            'notes': notes
        }
        
        record, analytics = pca_service.ingest_call(payload)
        
        update_progress(call_id, 'complete', 'completed',
                       f"Processing complete")
        
    except Exception as e:
        update_progress(call_id, 'error', 'failed',
                       f"Upload failed: {str(e)}")
        import traceback
        traceback.print_exc()


# ── POST /api/pca/uploads ─────────────────────────────────────────────────────

@pca_bp.route('/pca/uploads', methods=['POST'])
def upload_call():
    """
    Upload audio file + metadata → return call_id immediately → process in background
    Frontend: POST /api/pca/uploads (multipart)
    Language is auto-detected from transcript
    """
    try:
        # Get file
        if 'file' not in request.files:
            return error_response('No file provided', 400)
        
        file = request.files['file']
        if file.filename == '':
            return error_response('Empty filename', 400)
        
        # Get metadata from form data (no language field needed)
        caller_name = request.form.get('callerName', '').strip()
        notes = request.form.get('notes', '').strip()
        original_filename = secure_filename(file.filename)
        
        # Get file size and read file data
        file.seek(0, os.SEEK_END)
        audio_size = file.tell()
        file.seek(0)
        file_data = file.read()  # Read file content into memory
        
        # Generate call ID
        call_id = f"call-{uuid.uuid4().hex[:12]}"
        
        # Initialize progress
        update_progress(call_id, 'initiated', 'processing', 
                       f"Upload started")
        
        # Start background processing
        thread = threading.Thread(
            target=process_upload_async,
            args=(file_data, call_id, caller_name, notes, original_filename, audio_size)
        )
        thread.daemon = True
        thread.start()
        
        # Return immediately with call_id
        return success_response('Upload started, processing in background', {
            'callId': call_id,
            'message': 'Poll /api/pca/calls/{callId}/processing for status'
        })
        
    except Exception as e:
        print(f"[PCA] Upload initiation failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f'Upload failed: {str(e)}', 500)


# ── GET /api/pca/calls ────────────────────────────────────────────────────────

@pca_bp.route('/pca/calls', methods=['GET'])
def list_calls():
    """
    List all calls with pagination
    Frontend: GET /api/pca/calls?page=0&limit=10
    """
    try:
        page = int(request.args.get('page', 0))
        limit = int(request.args.get('limit', 10))
        agent_id = request.args.get('agent_id')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        data = pca_service.get_call_logs(agent_id, start_date, end_date, page, limit)
        
        return success_response('Call logs retrieved', {
            'calls': data['call_logs'],
            'total_count': data['total_count'],
            'total_pages': data['total_pages']
        })
        
    except Exception as e:
        print(f"[PCA] List calls failed: {e}")
        return error_response(f'Failed to list calls: {str(e)}', 500)


# ── GET /api/pca/calls/{callId} ───────────────────────────────────────────────

@pca_bp.route('/pca/calls/<string:call_id>', methods=['GET'])
def get_call(call_id):
    """
    Get full call details with analytics
    Frontend: GET /api/pca/calls/{callId}
    """
    try:
        data = pca_service.get_call_details(call_id)
        if data is None:
            return error_response('Call not found', 404)
        
        return success_response('Call details retrieved', data)
        
    except Exception as e:
        print(f"[PCA] Get call failed: {e}")
        return error_response(f'Failed to get call: {str(e)}', 500)


# ── DELETE /api/pca/calls/{callId} ────────────────────────────────────────────

@pca_bp.route('/pca/calls/<string:call_id>', methods=['DELETE'])
def delete_call(call_id):
    """
    Delete call record, analytics, and S3 files
    Frontend: DELETE /api/pca/calls/{callId}
    """
    try:
        success = pca_service.delete_call(call_id)
        
        if not success:
            return error_response('Failed to delete call', 500)
        
        return success_response('Call deleted successfully', {
            'callId': call_id,
            'deleted': True
        })
        
    except Exception as e:
        print(f"[PCA] Delete call failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f'Failed to delete call: {str(e)}', 500)


# ── GET /api/pca/calls/{callId}/processing ────────────────────────────────────

@pca_bp.route('/pca/calls/<string:call_id>/processing', methods=['GET'])
def get_processing_status(call_id):
    """
    Get processing status for a call
    Frontend: GET /api/pca/calls/{callId}/processing
    
    Returns real-time status and step progress for upload tracking
    """
    try:
        # Check if we have progress data for this call
        if call_id in upload_progress:
            progress_data = upload_progress[call_id]
            
            # Map internal steps to frontend-friendly format
            step_map = {
                'upload_audio': {'key': 'upload', 'label': 'Uploading audio'},
                'transcribe_audio': {'key': 'transcribe', 'label': 'Transcribing audio'},
                'save_transcript': {'key': 'diarize', 'label': 'Diarizing speakers'},
                'analyze_call': {'key': 'sentiment', 'label': 'Analyzing sentiment'},
                'complete': {'key': 'summarize', 'label': 'Generating summary'}
            }
            
            # Build steps array for frontend
            steps = []
            current_step_index = 0
            all_steps = ['upload_audio', 'transcribe_audio', 'save_transcript', 'analyze_call', 'complete']
            
            for idx, step_name in enumerate(all_steps):
                step_info = step_map.get(step_name, {'key': step_name, 'label': step_name})
                
                # Find if this step has been executed
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
                'logs': progress_data['steps']  # Full log for debugging
            })
        
        # Fallback: check database if no progress tracking
        record = ch.get_record(call_id)
        if not record:
            return error_response('Call not found', 404)
        
        analytics = ch.get_analytics(call_id)
        
        # Determine status and current step
        if analytics:
            status = 'completed'
            current_step = 5  # All steps complete
        else:
            status = 'processing'
            current_step = 3  # Analyzing
        
        # Fallback step structure
        steps = [
            {'key': 'upload', 'label': 'Uploading audio', 'state': 'done'},
            {'key': 'transcribe', 'label': 'Transcribing audio', 'state': 'done'},
            {'key': 'diarize', 'label': 'Diarizing speakers', 'state': 'done'},
            {'key': 'sentiment', 'label': 'Analyzing sentiment', 'state': 'done' if analytics else 'running'},
            {'key': 'summarize', 'label': 'Generating summary', 'state': 'done' if analytics else 'pending'},
        ]
        
        return success_response('Processing status retrieved', {
            'status': status,
            'currentStepIndex': current_step,
            'steps': steps
        })
        
    except Exception as e:
        print(f"[PCA] Get processing status failed: {e}")
        return error_response(f'Failed to get status: {str(e)}', 500)
        
        return success_response('Processing status retrieved', {
            'status': status,
            'currentStepIndex': current_step,
            'steps': steps
        })
        
    except Exception as e:
        print(f"[PCA] Get processing status failed: {e}")
        return error_response(f'Failed to get status: {str(e)}', 500)


# ── GET /api/pca/analytics/aggregate ──────────────────────────────────────────

@pca_bp.route('/pca/analytics/aggregate', methods=['GET'])
def get_aggregate_analytics():
    """
    Get aggregate analytics across all calls
    Frontend: GET /api/pca/analytics/aggregate?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
    """
    try:
        agent_id = request.args.get('agent_id')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        data = pca_service.get_agent_analytics(agent_id, start_date, end_date)
        
        return success_response('Aggregate analytics retrieved', data)
        
    except Exception as e:
        print(f"[PCA] Get aggregate analytics failed: {e}")
        return error_response(f'Failed to get analytics: {str(e)}', 500)


# ── GET /api/pca/calls/{callId}/export ────────────────────────────────────────

@pca_bp.route('/pca/calls/<string:call_id>/export', methods=['GET'])
def export_call_report(call_id):
    """
    Export call report in various formats
    Frontend: GET /api/pca/calls/{callId}/export?format=csv|pdf|json|xlsx
    
    Currently returns JSON format (CSV/PDF/XLSX can be added later)
    """
    try:
        format_type = request.args.get('format', 'json')
        
        data = pca_service.get_call_details(call_id)
        if data is None:
            return error_response('Call not found', 404)
        
        if format_type == 'json':
            return jsonify(data)
        else:
            # TODO: Implement CSV/PDF/XLSX export
            return error_response(f'Format {format_type} not yet supported', 400)
        
    except Exception as e:
        print(f"[PCA] Export call failed: {e}")
        return error_response(f'Failed to export call: {str(e)}', 500)
