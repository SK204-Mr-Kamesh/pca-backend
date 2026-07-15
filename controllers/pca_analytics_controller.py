"""
PCA Analytics API Routes
Analytics and metrics endpoints for PCA backend
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint
from services import pca_analytics_service
from controllers.controller_utils import success_response, error_response

analytics_bp = Blueprint('pca_analytics', __name__)


@analytics_bp.route('/pca_analytics', methods=['GET'])
def get_analytics():
    """
    Get comprehensive PCA analytics with 17 metrics
    Frontend: GET /api/pca_analytics
    
    Returns all analytics fields:
    {
        "total_uploads": <int: total call records>,
        "ready": <int: calls with completed analysis>,
        "failed": <int: calls pending/failed analysis>,
        "average_call_duration_minutes": <float: avg duration>,
        "average_sentiment": <float: 0-10 scale>,
        "average_customer_satisfaction": <float: 0-10 scale>,
        "average_wait_time_seconds": <float: avg hold time>,
        "average_sla_compliance": <float: 0-100%>,
        "average_abandonment_rate": <float: 0-100%>,
        "agent_effectiveness": <float: 0-10 scale>,
        "upload_volume": [{"date": "YYYY-MM-DD", "count": <int>}, ...],
        "sentiment_distribution": {"positive": <float>, "neutral": <float>, "negative": <float>},
        "language_distribution": {<language>: <float %>, ...},
        "top_topics": [{"topic": "<topic>", "count": <int>}, ...],
        "agent_leaderboard": [{"agent_id": "<id>", "calls": <int>, "score": <float>, ...}, ...],
        "executive_summary": ["<insight1>", "<insight2>", ...],
        "coaching_priorities": [{"rank": <int>, "priority": "<text>", "details": "<text>", "severity": "HIGH|MED|LOW"}, ...]
    }
    """
    try:
        analytics = pca_analytics_service.get_pca_analytics()
        return success_response('PCA analytics retrieved', analytics)
    except Exception as e:
        print(f"[Analytics] Get analytics failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f'Failed to get analytics: {str(e)}', 500)
