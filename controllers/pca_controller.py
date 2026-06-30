"""
PCA API Routes
All endpoints for standalone PCA backend
"""
import os
import sys
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import pca_service
from transcribe_service import upload_audio_to_s3, transcribe_audio, save_transcript_to_s3
from clickhouse_integration import CallRecord
import clickhouse_integration as ch

pca_bp = Blueprint('pca', __name__)

PCA_INGEST_SECRET = os.environ.get('PCA_INGEST_SECRET', '')


# ── Helper functions ──────────────────────────────────────────────────────────

def success_response(message, data, status_code=200):
    """Standard success response"""
    return jsonify({
        'status': 'success',
        'message': message,
        'data': data,
        'status_code': status_code
    }), status_code


def error_response(message, status_code=500, data=None):
    """Standard error response"""
    return jsonify({
        'status': 'error',
        'message': message,
        'data': data or {},
        'status_code': status_code
    }), status_code


# ── POST /api/pca/uploads ─────────────────────────────────────────────────────

@pca_bp.route('/pca/uploads', methods=['POST'])
def upload_call():
    """
    Upload audio file + metadata → transcribe → save to S3 → analyze
    Frontend: POST /api/pca/uploads (multipart)
    """
    try:
        # Get file
        if 'file' not in request.files:
            return error_response('No file provided', 400)
        
        file = request.files['file']
        if file.filename == '':
            return error_response('Empty filename', 400)
        
        # Get metadata from form data
        caller_name = request.form.get('callerName', '')
        language = request.form.get('language', 'en-US')
        notes = request.form.get('notes', '')
        
        # Generate call ID
        call_id = f"call-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now()
        
        # 1. Upload audio to S3
        print(f"[PCA] Uploading audio for {call_id}")
        recording_s3_key = upload_audio_to_s3(file, call_id)
        
        # 2. Transcribe audio
        print(f"[PCA] Transcribing audio for {call_id}")
        language_code = 'en-US' if language == 'English' else 'hi-IN' if language == 'Hindi' else 'en-US'
        transcript_messages = transcribe_audio(recording_s3_key, language_code)
        
        # 3. Save transcript to S3
        print(f"[PCA] Saving transcript for {call_id}")
        transcript_s3_key = save_transcript_to_s3(transcript_messages, call_id, started_at.isoformat())
        
        # 4. Calculate duration from transcript
        ended_at = datetime.now()
        duration = int((ended_at - started_at).total_seconds())
        
        # 5. Create call record and analyze
        print(f"[PCA] Creating record and analyzing {call_id}")
        payload = {
            'call_id': call_id,
            'session_id': call_id,
            'from_phone': caller_name or 'Unknown',
            'language': language,
            'started_at': started_at.isoformat(),
            'ended_at': ended_at.isoformat(),
            'status': 'answered',
            'call_source': 'upload',
            'transcript_key': transcript_s3_key,
            'recording_key': recording_s3_key
        }
        
        record, analytics = pca_service.ingest_call(payload)
        
        print(f"[PCA] Upload complete for {call_id}")
        return success_response('Call uploaded successfully', {
            'callId': call_id,
            'analyzed': analytics is not None
        })
        
    except Exception as e:
        print(f"[PCA] Upload failed: {e}")
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


# ── GET /api/pca/calls/{callId}/processing ────────────────────────────────────

@pca_bp.route('/pca/calls/<string:call_id>/processing', methods=['GET'])
def get_processing_status(call_id):
    """
    Get processing status for a call
    Frontend: GET /api/pca/calls/{callId}/processing
    
    Returns status and step progress for real-time updates
    """
    try:
        record = ch.get_record(call_id)
        if not record:
            return error_response('Call not found', 404)
        
        analytics = ch.get_analytics(call_id)
        
        # Determine status and current step
        if analytics:
            status = 'ready'
            current_step = 4  # All steps complete
        else:
            status = 'processing'
            current_step = 2  # Analyzing
        
        # Mock step structure for frontend
        steps = [
            {'key': 'upload', 'label': 'Upload', 'state': 'done'},
            {'key': 'transcribe', 'label': 'Transcribe audio', 'state': 'done'},
            {'key': 'diarize', 'label': 'Diarize speakers', 'state': 'done'},
            {'key': 'sentiment', 'label': 'Score sentiment', 'state': 'done' if analytics else 'running'},
            {'key': 'summarize', 'label': 'Topic & summary', 'state': 'done' if analytics else 'pending'},
        ]
        
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


# ── POST /api/pca/calls/{callId}/chat ─────────────────────────────────────────

@pca_bp.route('/pca/calls/<string:call_id>/chat', methods=['POST'])
def chat_about_call(call_id):
    """
    Gen AI Assistant: ask questions about a call
    Frontend: POST /api/pca/calls/{callId}/chat
    Body: {"question": "..."}
    """
    try:
        body = request.get_json(silent=True) or {}
        question = (body.get('question') or body.get('query') or '').strip()
        
        if not question:
            return error_response('question is required', 400)
        
        answer = pca_service.chat_about_call(call_id, question)
        
        return success_response('Answer generated', {'answer': answer})
        
    except Exception as e:
        print(f"[PCA] Chat failed: {e}")
        return error_response(f'Failed to answer: {str(e)}', 500)


# ── POST /api/pca/ingest (internal) ───────────────────────────────────────────

@pca_bp.route('/pca/ingest', methods=['POST'])
def pca_ingest():
    """
    Internal endpoint for worker to push call data
    Requires X-PCA-Secret header
    """
    # Check secret if configured
    if PCA_INGEST_SECRET:
        if request.headers.get('X-PCA-Secret', '') != PCA_INGEST_SECRET:
            return error_response('Unauthorized', 401)
    
    try:
        payload = request.get_json(silent=True) or {}
        record, analytics = pca_service.ingest_call(payload)
        
        return success_response('Call ingested', {
            'call_id': record.call_id,
            'analyzed': analytics is not None
        })
        
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        print(f"[PCA] Ingest failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f'Ingest failed: {str(e)}', 500)
