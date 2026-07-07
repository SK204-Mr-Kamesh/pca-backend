"""
Data models for PCA
CallRecord and CallAnalytics classes
"""
from datetime import datetime


class CallRecord:
    """Call record model"""
    
    def __init__(self, call_id):
        self.call_id = call_id
        self.agent_id = None
        self.account_id = None
        self.room_name = None
        self.from_phone = None
        self.to_phone = None
        self.call_source = 'upload'
        self.status = 'answered'
        self.language = None
        self.started_at = None
        self.ended_at = None
        self.duration_seconds = 0
        self.transcript_s3_key = None
        self.recording_s3_key = None
        self.created_at = datetime.now()
    
    def _format_duration(self):
        """Format duration as MM:SS"""
        if not self.duration_seconds:
            return "00:00"
        minutes = self.duration_seconds // 60
        seconds = self.duration_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def to_dict(self):
        """Convert to dict"""
        return {
            'call_id': self.call_id,
            'agent_id': self.agent_id,
            'account_id': self.account_id,
            'room_name': self.room_name,
            'from_phone': self.from_phone,
            'to_phone': self.to_phone,
            'call_source': self.call_source,
            'status': self.status,
            'language': self.language,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'duration_seconds': self.duration_seconds,
            'transcript_s3_key': self.transcript_s3_key,
            'recording_s3_key': self.recording_s3_key
        }
    
    def to_log_dict(self, analytics=None):
        """Convert to call log format for frontend"""
        return {
            'callId': self.call_id,
            'customerName': (analytics.customer_name if analytics else None) or self.from_phone or 'Unknown',
            'phoneNumber': self.from_phone or self.to_phone or '—',
            'hangupReason': (analytics.hangup_reason if analytics else None) or '—',
            'language': self.language or '—',
            'callDuration': self._format_duration(),
            'status': self.status,
            'callStart': self.started_at.strftime('%d/%m/%Y, %H:%M:%S') if self.started_at else '—',
            'sentiment': float(analytics.overall_sentiment) if analytics and analytics.overall_sentiment else 0,
            'customerSatisfaction': float(analytics.customer_satisfaction) if analytics and analytics.customer_satisfaction else 0,
            'agentPerformance': float(analytics.agent_performance) if analytics and analytics.agent_performance else 0
        }


class CallAnalytics:
    """Call analytics model"""
    
    def __init__(self, call_id):
        self.call_id = call_id
        self.overall_sentiment = None
        self.customer_satisfaction = None
        self.agent_performance = None
        self.summary = ''
        self.topics = []
        self.action_items = []
        self.key_indicators = []
        self.customer_name = None
        self.hangup_reason = None
        self.call_matrices = {}
        self.raw_model_response = None
        self.model_id = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Validation fields (Phase 2)
        self.validation_results = None  # Full validation JSON
        self.validation_score = None  # Overall weighted score
        self.validation_percentage = None  # Percentage score
        self.skill_level = None  # Expert/Intermediate/Novice
    
    def to_dict(self):
        """Convert to dict for frontend"""
        base_dict = {
            'sentiment': float(self.overall_sentiment) if self.overall_sentiment else 0,
            'customerSatisfaction': float(self.customer_satisfaction) if self.customer_satisfaction else 0,
            'agentPerformance': float(self.agent_performance) if self.agent_performance else 0,
            'summary': self.summary or '',
            'topics': self.topics or [],
            'actionItems': self.action_items or [],
            'keyIndicators': self.key_indicators or [],
            'matrices': self.call_matrices or {}
        }
        
        # Add validation data if available
        if self.validation_results:
            base_dict['validation'] = self.validation_results
            base_dict['validationScore'] = float(self.validation_score) if self.validation_score else 0
            base_dict['validationPercentage'] = float(self.validation_percentage) if self.validation_percentage else 0
            base_dict['skillLevel'] = self.skill_level or 'Novice'
        
        return base_dict
