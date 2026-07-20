"""
In-Store Analytics API Routes
Analytics and metrics endpoints for in-store interactions
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint
from services import instore_analytics_service
from controllers.controller_utils import success_response, error_response

instore_analytics_bp = Blueprint('instore_analytics', __name__)


@instore_analytics_bp.route('/instore_analytics', methods=['GET'])
def get_analytics():
    """
    Get comprehensive in-store analytics
    Frontend: GET /api/instore_analytics
    
    Returns all analytics fields:
    {
        "total_uploads": <int: total interaction records>,
        "ready": <int: interactions with completed analysis>,
        "failed": <int: interactions pending/failed analysis>,
        "average_interaction_duration_minutes": <float: avg duration>,
        "average_overall_sentiment": <float: 0-10 scale>,
        "average_customer_satisfaction": <float: 0-10 scale>,
        "average_sales_executive_performance": <float: 0-10 scale>,
        "average_sla_compliance": <float: 0-100%>,
        "upload_volume": [{"date": "YYYY-MM-DD", "count": <int>}, ...],
        "sentiment_distribution": {"positive": <float>, "neutral": <float>, "negative": <float>},
        "language_distribution": {<language>: <float %>, ...},
        "top_topics": [{"topic": "<topic>", "count": <int>}, ...],
        "sales_executive_leaderboard": [{"sales_executive_id": "<id>", "interactions": <int>, "score": <float>, ...}, ...],
        "executive_summary": ["<insight1>", "<insight2>", ...],
        "coaching_priorities": [{"rank": <int>, "priority": "<text>", "details": "<text>", "severity": "HIGH|MED|LOW"}, ...],
        "quality_scorecard": [{"sales_executive_id": "<id>", "interactions": <int>, "communication": <float 0-100>, "discovery": <float 0-100>, "solution_fit": <float 0-100>, "sales_execution": <float 0-100>, "customer_experience": <float 0-100>}, ...]
    }
    """
    try:
        analytics = instore_analytics_service.get_instore_analytics()
        return success_response('In-store analytics retrieved', analytics)
    except Exception as e:
        print(f"[InStore Analytics] Get analytics failed: {e}")
        import traceback
        traceback.print_exc()
        return error_response(f'Failed to get analytics: {str(e)}', 500)
