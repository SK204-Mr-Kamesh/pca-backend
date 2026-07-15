"""
PCA Analytics Service
Provides aggregate analytics and metrics for PCA call records
"""
import json
import pca_clickhouse as ch


def get_pca_analytics():
    """
    Get aggregate analytics for all PCA calls
    
    Returns:
        Dictionary with:
        - total_uploads: Total call records in wakefit_call_records
        - ready: Call records that have analytics (present in wakefit_call_analytics)
        - failed: Call records missing analytics (absent in wakefit_call_analytics)
        - average_call_duration_minutes: Average duration across all calls
        - average_sentiment: Average overall_sentiment from all analytics
        - average_wait_time_seconds: Average hold/wait time
        - average_sla_compliance: Average SLA compliance percentage
        - average_abandonment_rate: Average abandonment rate percentage
        - agent_effectiveness: Average agent_performance score
    """
    try:
        # Get all records from wakefit_call_records
        all_records, total_count = ch.query_records(page=0, limit=999999)
        
        total_uploads = total_count
        
        if total_uploads == 0:
            return {
                'total_uploads': 0,
                'ready': 0,
                'failed': 0,
                'average_call_duration_minutes': 0,
                'average_sentiment': 0,
                'average_customer_satisfaction': 0,
                'average_wait_time_seconds': 0,
                'average_sla_compliance': 0,
                'average_abandonment_rate': 0,
                'agent_effectiveness': 0
            }
        
        # Get analytics for all records
        call_ids = [r.call_id for r in all_records]
        analytics_map = ch.get_analytics_map(call_ids)
        
        # Count ready (present in analytics) and failed (missing in analytics)
        ready = len(analytics_map)
        failed = total_uploads - ready
        
        # Calculate average call duration
        durations = [r.duration_seconds or 0 for r in all_records if r.duration_seconds]
        average_duration_seconds = sum(durations) / len(durations) if durations else 0
        average_call_duration_minutes = round(average_duration_seconds / 60, 2)
        
        # Calculate average sentiment
        sentiments = []
        # Calculate average customer satisfaction
        customer_satisfactions = []
        # Calculate average wait time
        wait_times = []
        # Calculate average SLA compliance
        sla_compliances = []
        # Calculate average abandonment rate
        abandonment_rates = []
        # Calculate agent effectiveness
        agent_performances = []
        
        for analytics in analytics_map.values():
            # Average sentiment
            if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
                overall_sentiment = analytics.raw_model_response.get('overall_sentiment')
                if overall_sentiment is not None:
                    try:
                        sentiments.append(float(overall_sentiment))
                    except (TypeError, ValueError):
                        pass
                
                # Average wait time
                avg_wait_time = analytics.raw_model_response.get('avg_wait_time')
                if avg_wait_time is not None:
                    try:
                        wait_times.append(float(avg_wait_time))
                    except (TypeError, ValueError):
                        pass
                
                # Average SLA compliance
                sla_compliance = analytics.raw_model_response.get('sla_compliance')
                if sla_compliance is not None:
                    try:
                        sla_compliances.append(float(sla_compliance))
                    except (TypeError, ValueError):
                        pass
                
                # Average abandonment rate
                abandonment_rate = analytics.raw_model_response.get('abandonment_rate')
                if abandonment_rate is not None:
                    try:
                        abandonment_rates.append(float(abandonment_rate))
                    except (TypeError, ValueError):
                        pass
            
            # Customer satisfaction
            if analytics.customer_satisfaction is not None:
                try:
                    customer_satisfactions.append(float(analytics.customer_satisfaction))
                except (TypeError, ValueError):
                    pass
            
            # Agent effectiveness (agent_performance)
            if analytics.agent_performance is not None:
                try:
                    agent_performances.append(float(analytics.agent_performance))
                except (TypeError, ValueError):
                    pass
        
        average_sentiment = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0
        average_customer_satisfaction = round(sum(customer_satisfactions) / len(customer_satisfactions), 2) if customer_satisfactions else 0
        average_wait_time_seconds = round(sum(wait_times) / len(wait_times), 2) if wait_times else 0
        average_sla_compliance = round(sum(sla_compliances) / len(sla_compliances), 2) if sla_compliances else 0
        average_abandonment_rate = round(sum(abandonment_rates) / len(abandonment_rates), 2) if abandonment_rates else 0
        agent_effectiveness = round(sum(agent_performances) / len(agent_performances), 2) if agent_performances else 0
        
        return {
            'total_uploads': total_uploads,
            'ready': ready,
            'failed': failed,
            'average_call_duration_minutes': average_call_duration_minutes,
            'average_sentiment': average_sentiment,
            'average_customer_satisfaction': average_customer_satisfaction,
            'average_wait_time_seconds': average_wait_time_seconds,
            'average_sla_compliance': average_sla_compliance,
            'average_abandonment_rate': average_abandonment_rate,
            'agent_effectiveness': agent_effectiveness
        }
        
    except Exception as e:
        print(f"[PCA Analytics] Error getting analytics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'total_uploads': 0,
            'ready': 0,
            'failed': 0,
            'average_call_duration_minutes': 0,
            'average_sentiment': 0,
            'average_customer_satisfaction': 0,
            'average_wait_time_seconds': 0,
            'average_sla_compliance': 0,
            'average_abandonment_rate': 0,
            'agent_effectiveness': 0,
            'error': str(e)
        }
