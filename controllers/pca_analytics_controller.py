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
    Get aggregate PCA analytics
    Frontend: GET /api/analytics
    
    Returns:
    {
        "total_uploads": <total call records>,
        "ready": <calls with completed analysis>,
        "failed": <calls pending/failed analysis>,
        "average_call_duration_minutes": <avg duration>,
        "average_sentiment": <avg sentiment 0-10>
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
