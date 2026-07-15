"""
PCA Analytics Service
Provides aggregate analytics and metrics for PCA call records
"""
import json
import os
import boto3
from datetime import datetime, timedelta, timezone
from collections import Counter
import pca_clickhouse as ch

AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
PCA_MODEL_ID = os.environ.get("PCA_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")


def _round_float(value, decimals=2):
    """Round float values to specified decimal places"""
    if isinstance(value, (int, float)):
        return round(float(value), decimals)
    return value


def _get_bedrock_client():
    """Get AWS Bedrock client"""
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=AWS_REGION,
    )


def get_pca_analytics():
    """
    Get comprehensive PCA analytics including all dashboard metrics
    """
    try:
        # Get all records (pass None for agent_id to get all records)
        all_records, total_count = ch.query_records(agent_id=None, page=0, limit=999999)
        
        total_uploads = total_count
        
        if total_uploads == 0:
            return _get_empty_analytics()
        
        # Get analytics for all records
        call_ids = [r.call_id for r in all_records]
        analytics_map = ch.get_analytics_map(call_ids)
        
        ready = len(analytics_map)
        failed = total_uploads - ready
        
        # Create a map for quick record lookup
        records_map = {r.call_id: r for r in all_records}
        
        # Basic metrics
        durations = [r.duration_seconds or 0 for r in all_records if r.duration_seconds]
        average_duration_seconds = sum(durations) / len(durations) if durations else 0.0
        average_call_duration_minutes = average_duration_seconds / 60
        
        # Initialize collections
        sentiments = []
        customer_satisfactions = []
        wait_times = []
        sla_compliances = []
        abandonment_rates = []
        agent_performances = []
        topics_list = []
        languages = []
        agent_scores = {}  # For leaderboard
        agent_performance_details = {}  # For coaching priorities
        coaching_priorities_all = []  # Collect all coaching priorities
        
        for call_id, analytics in analytics_map.items():
            # Find corresponding record for agent_id
            record = records_map.get(call_id)
            agent_id = record.agent_id if record else "Unknown"
            
            # Initialize agent tracking
            if agent_id not in agent_performance_details:
                agent_performance_details[agent_id] = {
                    'calls': [],
                    'low_performers': False,
                    'issues': []
                }
            
            if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
                # Sentiment metrics
                overall_sentiment = analytics.raw_model_response.get('overall_sentiment')
                if overall_sentiment is not None:
                    try:
                        sentiment_val = float(overall_sentiment)
                        sentiments.append(sentiment_val)
                        agent_performance_details[agent_id]['calls'].append({
                            'call_id': call_id,
                            'sentiment': sentiment_val
                        })
                    except (TypeError, ValueError):
                        pass
                
                # Wait time
                avg_wait_time = analytics.raw_model_response.get('avg_wait_time')
                if avg_wait_time is not None:
                    try:
                        wait_times.append(float(avg_wait_time))
                    except (TypeError, ValueError):
                        pass
                
                # SLA compliance
                sla_compliance = analytics.raw_model_response.get('sla_compliance')
                if sla_compliance is not None:
                    try:
                        sla_val = float(sla_compliance)
                        sla_compliances.append(sla_val)
                        if sla_val < 60:
                            agent_performance_details[agent_id]['issues'].append('Low SLA compliance')
                    except (TypeError, ValueError):
                        pass
                
                # Abandonment rate
                abandonment_rate = analytics.raw_model_response.get('abandonment_rate')
                if abandonment_rate is not None:
                    try:
                        abandonment_rates.append(float(abandonment_rate))
                    except (TypeError, ValueError):
                        pass
                
                # Topics
                topics = analytics.raw_model_response.get('topics', [])
                if topics:
                    topics_list.extend(topics)
                
                # Learning suggestions
                learning_suggestions = analytics.raw_model_response.get('learning_suggestions')
                if learning_suggestions:
                    agent_performance_details[agent_id]['issues'].append(learning_suggestions)
                
                # Coaching priorities
                coaching_priorities = analytics.raw_model_response.get('coaching_priorities', [])
                if coaching_priorities and isinstance(coaching_priorities, list):
                    coaching_priorities_all.extend(coaching_priorities)
            
            # Customer satisfaction
            csat = None
            if analytics.customer_satisfaction is not None:
                try:
                    csat = float(analytics.customer_satisfaction)
                    customer_satisfactions.append(csat)
                except (TypeError, ValueError):
                    pass
            
            # Agent performance
            agent_perf = None
            if analytics.agent_performance is not None:
                try:
                    agent_perf = float(analytics.agent_performance)
                    agent_performances.append(agent_perf)
                    if agent_perf < 6:
                        agent_performance_details[agent_id]['low_performers'] = True
                except (TypeError, ValueError):
                    pass
            
            # Language
            if record and record.language:
                languages.append(record.language)
            
            # Build agent leaderboard data
            if agent_id not in agent_scores:
                agent_scores[agent_id] = {
                    'calls': 0,
                    'performance_scores': [],
                    'satisfaction_scores': [],
                    'sla_scores': [],
                    'compliance_count': 0,
                    'sentiment_scores': []
                }
            
            agent_scores[agent_id]['calls'] += 1
            if agent_perf is not None:
                agent_scores[agent_id]['performance_scores'].append(agent_perf)
            if csat is not None:
                agent_scores[agent_id]['satisfaction_scores'].append(csat)
            if sla_compliance is not None:
                agent_scores[agent_id]['sla_scores'].append(float(sla_compliance))
            if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
                compliance_flags = analytics.raw_model_response.get('compliance_flags', [])
                agent_scores[agent_id]['compliance_count'] += len(compliance_flags)
            if overall_sentiment is not None:
                try:
                    agent_scores[agent_id]['sentiment_scores'].append(float(overall_sentiment))
                except (TypeError, ValueError):
                    pass
        
        # Calculate averages (keep as precise floats)
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        average_customer_satisfaction = sum(customer_satisfactions) / len(customer_satisfactions) if customer_satisfactions else 0.0
        average_wait_time_seconds = sum(wait_times) / len(wait_times) if wait_times else 0.0
        average_sla_compliance = sum(sla_compliances) / len(sla_compliances) if sla_compliances else 0.0
        average_abandonment_rate = sum(abandonment_rates) / len(abandonment_rates) if abandonment_rates else 0.0
        agent_effectiveness = sum(agent_performances) / len(agent_performances) if agent_performances else 0.0
        
        # Upload volume (per day for last 7 days)
        upload_volume = _get_upload_volume_trend(all_records)
        
        # Sentiment distribution (positive/neutral/negative)
        sentiment_distribution = _get_sentiment_distribution(sentiments)
        
        # Language distribution
        language_distribution = _get_language_distribution(languages)
        
        # Top topics
        top_topics = _get_top_topics(topics_list)
        
        # Agent effectiveness leaderboard
        leaderboard = _get_agent_leaderboard(agent_scores)
        
        # Executive AI summary (top insights)
        executive_summary = _get_executive_summary(
            sentiments, 
            sla_compliances, 
            abandonment_rates,
            len(analytics_map)
        )
        
        # Top coaching priorities (try raw data first, fallback to analysis)
        coaching_priorities = _get_coaching_priorities_from_raw_data(coaching_priorities_all)
        if not coaching_priorities:
            coaching_priorities = _get_coaching_priorities(agent_performance_details, agent_scores)
        
        # AI Quality Scorecard (8 metrics per agent)
        quality_scorecard = _get_quality_scorecard(analytics_map, records_map)
        
        return {
            'total_uploads': total_uploads,
            'ready': ready,
            'failed': failed,
            'average_call_duration_minutes': _round_float(average_call_duration_minutes),
            'average_sentiment': _round_float(average_sentiment),
            'average_customer_satisfaction': _round_float(average_customer_satisfaction),
            'average_wait_time_seconds': _round_float(average_wait_time_seconds),
            'average_sla_compliance': _round_float(average_sla_compliance),
            'average_abandonment_rate': _round_float(average_abandonment_rate),
            'agent_effectiveness': _round_float(agent_effectiveness),
            'upload_volume': upload_volume,
            'sentiment_distribution': {
                'positive': _round_float(sentiment_distribution['positive']),
                'neutral': _round_float(sentiment_distribution['neutral']),
                'negative': _round_float(sentiment_distribution['negative'])
            },
            'language_distribution': {lang: _round_float(pct) for lang, pct in language_distribution.items()},
            'top_topics': top_topics,
            'agent_leaderboard': [{
                **agent,
                'score': _round_float(agent['score']),
                'csat': _round_float(agent['csat']),
                'fcr': _round_float(agent['fcr']),
                'compliance': _round_float(agent['compliance']),
                'sentiment': _round_float(agent['sentiment'])
            } for agent in leaderboard],
            'executive_summary': executive_summary,
            'coaching_priorities': coaching_priorities,
            'quality_scorecard': [{
                **scorecard,
                'greeting': _round_float(scorecard['greeting']),
                'understanding': _round_float(scorecard['understanding']),
                'crm_validation': _round_float(scorecard['crm_validation']),
                'communication': _round_float(scorecard['communication']),
                'soft_skills': _round_float(scorecard['soft_skills']),
                'compliance': _round_float(scorecard['compliance']),
                'resolution': _round_float(scorecard['resolution']),
                'closing': _round_float(scorecard['closing'])
            } for scorecard in quality_scorecard]
        }
        
    except Exception as e:
        print(f"[PCA Analytics] Error getting analytics: {e}")
        import traceback
        traceback.print_exc()
        return _get_empty_analytics()


def _get_empty_analytics():
    """Return empty analytics structure"""
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
        'upload_volume': [],
        'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
        'language_distribution': {},
        'top_topics': [],
        'agent_leaderboard': [],
        'executive_summary': [],
        'coaching_priorities': [],
        'quality_scorecard': []
    }


def _get_upload_volume_trend(records):
    """Get upload volume per day for last 7 days"""
    trend = {}
    today = datetime.now(timezone.utc).date()
    
    for i in range(7):
        date = (today - timedelta(days=i)).isoformat()
        trend[date] = 0
    
    for record in records:
        if record.created_on:
            date = record.created_on.date().isoformat()
            if date in trend:
                trend[date] += 1
    
    return [{'date': k, 'count': v} for k, v in sorted(trend.items())]


def _get_sentiment_distribution(sentiments):
    """Categorize sentiments into positive/neutral/negative with corrected thresholds"""
    if not sentiments:
        return {'positive': 0, 'neutral': 0, 'negative': 0}
    
    # Fixed thresholds: positive ≥7, neutral 3-7, negative <3
    positive = sum(1 for s in sentiments if s >= 7)
    negative = sum(1 for s in sentiments if s < 3)  
    neutral = sum(1 for s in sentiments if 3 <= s < 7)
    
    total = len(sentiments)
    
    # Calculate percentages (keep as precise floats)
    pos_pct = (positive / total) * 100 if total > 0 else 0.0
    neg_pct = (negative / total) * 100 if total > 0 else 0.0
    neu_pct = (neutral / total) * 100 if total > 0 else 0.0
    
    # Adjust for rounding to ensure total = 100% (keep precise floats)
    total_pct = pos_pct + neg_pct + neu_pct
    if total_pct != 100.0 and total > 0:
        # Adjust the largest percentage to make total = 100%
        if pos_pct >= neg_pct and pos_pct >= neu_pct:
            pos_pct += (100.0 - total_pct)
        elif neg_pct >= neu_pct:
            neg_pct += (100.0 - total_pct)
        else:
            neu_pct += (100.0 - total_pct)
    
    return {
        'positive': pos_pct,
        'neutral': neu_pct,
        'negative': neg_pct
    }


def _get_language_distribution(languages):
    """Get distribution of detected languages"""
    if not languages:
        return {}
    
    language_counts = Counter(languages)
    total = len(languages)
    
    return {
        lang: (count / total) * 100 
        for lang, count in language_counts.most_common()
    }


def _normalize_topics(topics_list):
    """
    Normalize topic names to standard taxonomy for consistent analytics
    """
    if not topics_list:
        return []
    
    # Define topic normalization mapping
    topic_mapping = {
        # Delivery Issues
        'delivery delay': 'Delivery Delay',
        'late delivery': 'Delivery Delay',
        'delayed delivery': 'Delivery Delay', 
        'shipment delay': 'Delivery Delay',
        'order delay': 'Delivery Delay',
        'delayed order': 'Delivery Delay',
        
        'delivery damage': 'Delivery Damaged',
        'damaged delivery': 'Delivery Damaged',
        'broken product': 'Delivery Damaged',
        'damaged goods': 'Delivery Damaged',
        
        'wrong address': 'Delivery Wrong Address',
        'address issue': 'Delivery Wrong Address',
        'delivery address': 'Delivery Wrong Address',
        
        'missed delivery': 'Delivery Missed',
        'delivery attempt': 'Delivery Missed',
        'failed delivery': 'Delivery Missed',
        
        'delivery reschedule': 'Delivery Rescheduling',
        'reschedule delivery': 'Delivery Rescheduling',
        'change delivery': 'Delivery Rescheduling',
        
        # Payment & Refunds
        'refund': 'Refund Request',
        'money back': 'Refund Request', 
        'return money': 'Refund Request',
        'get refund': 'Refund Request',
        'refund query': 'Refund Request',
        
        'payment problem': 'Payment Issue',
        'payment failed': 'Payment Issue',
        'payment error': 'Payment Issue',
        'transaction issue': 'Payment Issue',
        
        'billing question': 'Billing Query',
        'invoice query': 'Billing Query',
        'bill inquiry': 'Billing Query',
        
        'price match': 'Price Match Request',
        'lower price': 'Price Match Request',
        
        # Product Issues  
        'product quality': 'Product Quality Issue',
        'quality issue': 'Product Quality Issue',
        'defective product': 'Product Quality Issue',
        'product defect': 'Product Quality Issue',
        'comfort issue': 'Product Quality Issue',
        'product problem': 'Product Quality Issue',
        
        'size issue': 'Product Size Issue',
        'wrong size': 'Product Size Issue',
        'size problem': 'Product Size Issue',
        'dimension issue': 'Product Size Issue',
        
        'product exchange': 'Product Exchange',
        'exchange product': 'Product Exchange',
        'change product': 'Product Exchange',
        
        'product info': 'Product Information',
        'product details': 'Product Information',
        'specification': 'Product Information',
        'product question': 'Product Information',
        
        'assembly': 'Assembly Issue',
        'installation': 'Assembly Issue',
        'setup issue': 'Assembly Issue',
        
        # Order Management
        'order status': 'Order Status Inquiry',
        'track order': 'Order Status Inquiry',
        'order tracking': 'Order Status Inquiry',
        'where is order': 'Order Status Inquiry',
        'order inquiry': 'Order Status Inquiry',
        
        'cancel order': 'Order Cancellation',
        'order cancel': 'Order Cancellation',
        'cancellation': 'Order Cancellation',
        
        'modify order': 'Order Modification',
        'change order': 'Order Modification',
        'order change': 'Order Modification',
        'update order': 'Order Modification',
        
        'new order': 'New Order Placement',
        'place order': 'New Order Placement',
        'order placement': 'New Order Placement',
        
        # Technical Support
        'website problem': 'Website Issue',
        'app issue': 'Website Issue',
        'site not working': 'Website Issue',
        'technical issue': 'Website Issue',
        
        'login issue': 'Account Access',
        'account problem': 'Account Access',
        'password issue': 'Account Access',
        'cant login': 'Account Access',
        
        'warranty': 'Warranty Claim',
        'warranty claim': 'Warranty Claim',
        'warranty issue': 'Warranty Claim',
        
        'installation help': 'Installation Support',
        'setup help': 'Installation Support',
        'how to install': 'Installation Support',
        
        # General
        'general question': 'General Inquiry',
        'inquiry': 'General Inquiry',
        'information': 'General Inquiry',
        
        'complaint': 'Complaint Escalation',
        'escalation': 'Complaint Escalation',
        'manager request': 'Complaint Escalation',
        
        'feedback': 'Feedback/Review',
        'review': 'Feedback/Review',
        'suggestion': 'Feedback/Review',
    }
    
    normalized_topics = []
    
    for topic in topics_list:
        if not topic:
            continue
            
        # Keep order IDs and preserve case for them
        order_id_part = ""
        topic_clean = topic
        if " - #" in topic or " #" in topic:
            parts = topic.split(" - #") if " - #" in topic else topic.split(" #")
            if len(parts) == 2:
                topic_clean = parts[0]
                order_id_part = f" - #{parts[1]}"
        
        # Normalize the topic part (case-insensitive matching)
        topic_lower = topic_clean.lower().strip()
        
        # Find best match in mapping
        normalized_topic = None
        for key, value in topic_mapping.items():
            if key in topic_lower or topic_lower in key:
                normalized_topic = value
                break
        
        # If no match found, try to find partial matches for standard topics
        if not normalized_topic:
            standard_topics = [
                'Delivery Delay', 'Delivery Damaged', 'Delivery Wrong Address', 'Delivery Missed', 'Delivery Rescheduling',
                'Refund Request', 'Payment Issue', 'Billing Query', 'Price Match Request',
                'Product Quality Issue', 'Product Size Issue', 'Product Exchange', 'Product Information', 'Assembly Issue',
                'Order Status Inquiry', 'Order Cancellation', 'Order Modification', 'New Order Placement',
                'Website Issue', 'Account Access', 'Warranty Claim', 'Installation Support',
                'General Inquiry', 'Complaint Escalation', 'Feedback/Review'
            ]
            
            for standard_topic in standard_topics:
                if any(word in topic_lower for word in standard_topic.lower().split()):
                    normalized_topic = standard_topic
                    break
        
        # Use original topic if no normalization found
        if not normalized_topic:
            normalized_topic = topic_clean
        
        # Add back order ID if present
        final_topic = normalized_topic + order_id_part
        
        # Keep ALL instances (including duplicates) for proper counting
        normalized_topics.append(final_topic)
    
    return normalized_topics


def _get_all_topics_from_database():
    """
    Get all topics from all call records in the database
    Returns normalized topic counts from all calls
    """
    try:
        # Get all analytics records from database
        all_records, _ = ch.query_records(agent_id=None, page=0, limit=999999)
        call_ids = [r.call_id for r in all_records]
        analytics_map = ch.get_analytics_map(call_ids)
        
        # Collect all topics from all calls
        all_topics = []
        
        for call_id, analytics in analytics_map.items():
            if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
                topics = analytics.raw_model_response.get('topics', [])
                if topics and isinstance(topics, list):
                    all_topics.extend(topics)
        
        # Apply normalization
        normalized_topics = _normalize_topics(all_topics)
        
        # Count occurrences
        topic_counts = Counter(normalized_topics)
        
        # Return top 5 most common
        top_topics = [
            {'topic': topic, 'count': count}
            for topic, count in topic_counts.most_common(5)
        ]
        
        return top_topics
        
    except Exception as e:
        return []


def _get_top_topics(topics_list):
    """
    Get top 5 most common topics from database (ignores topics_list parameter)
    Now queries database directly for accurate counts
    """
    return _get_all_topics_from_database()


def _get_agent_leaderboard(agent_scores):
    """Build agent effectiveness leaderboard"""
    leaderboard = []
    
    for agent_id, scores in agent_scores.items():
        avg_score = sum(scores['performance_scores']) / len(scores['performance_scores']) if scores['performance_scores'] else 0.0
        avg_csat = sum(scores['satisfaction_scores']) / len(scores['satisfaction_scores']) if scores['satisfaction_scores'] else 0.0
        avg_sla = sum(scores['sla_scores']) / len(scores['sla_scores']) if scores['sla_scores'] else 0.0
        avg_sentiment = sum(scores['sentiment_scores']) / len(scores['sentiment_scores']) if scores['sentiment_scores'] else 0.0
        
        # Fix compliance rate calculation - should be percentage of calls WITHOUT violations
        total_calls = scores['calls']
        violation_calls = min(scores['compliance_count'], total_calls)  # Can't exceed total calls
        clean_calls = total_calls - violation_calls
        compliance_rate = (clean_calls / total_calls) * 100 if total_calls > 0 else 100.0
        
        leaderboard.append({
            'agent_id': agent_id,
            'calls': scores['calls'],
            'score': _round_float(avg_score),
            'csat': _round_float(avg_csat),
            'fcr': _round_float(avg_sla),  # First Call Resolution
            'compliance': _round_float(max(0.0, min(100.0, compliance_rate))),  # Ensure 0-100 range
            'sentiment': _round_float(avg_sentiment)
        })
    
    # Sort by score descending (keep as float)
    return sorted(leaderboard, key=lambda x: x['score'], reverse=True)[:10]


def _get_executive_summary(sentiments, sla_compliances, abandonment_rates, total_calls):
    """Generate executive summary insights using Claude LLM for intelligent analysis"""
    
    if not sentiments and not sla_compliances:
        return []
    
    try:
        # Prepare data summary for LLM
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        negative_count = sum(1 for s in sentiments if s <= 3) if sentiments else 0
        negative_pct = (negative_count / len(sentiments)) * 100 if sentiments else 0.0
        positive_count = sum(1 for s in sentiments if s >= 7) if sentiments else 0
        positive_pct = (positive_count / len(sentiments)) * 100 if sentiments else 0.0
        
        avg_sla = sum(sla_compliances) / len(sla_compliances) if sla_compliances else 0.0
        high_sla_count = sum(1 for s in sla_compliances if s >= 80) if sla_compliances else 0
        high_sla_pct = (high_sla_count / len(sla_compliances)) * 100 if sla_compliances else 0.0
        low_sla_count = sum(1 for s in sla_compliances if s < 60) if sla_compliances else 0
        low_sla_pct = (low_sla_count / len(sla_compliances)) * 100 if sla_compliances else 0.0
        
        avg_abandonment = sum(abandonment_rates) / len(abandonment_rates) if abandonment_rates else 0.0
        abandoned_count = sum(1 for a in abandonment_rates if a > 0) if abandonment_rates else 0
        abandoned_pct = (abandoned_count / len(abandonment_rates)) * 100 if abandonment_rates else 0.0
        
        # Create context for LLM
        data_context = f"""
Call Center Analytics Summary (Last Period):
- Total Calls Analyzed: {total_calls}
- Average Call Sentiment: {avg_sentiment}/10
  * Positive sentiment (≥7): {positive_pct}% of calls
  * Negative sentiment (≤3): {negative_pct}% of calls
- Average SLA Compliance: {avg_sla}%
  * High compliance (≥80%): {high_sla_pct}% of calls
  * Low compliance (<60%): {low_sla_pct}% of calls
- Abandonment Rate: {abandoned_pct}% (avg: {avg_abandonment}%)

Please analyze this data and generate 3-5 concise, actionable business insights for the executive team.
Each insight should:
1. Be data-driven and specific (include numbers)
2. Highlight trends or concerns
3. Suggest action if needed
4. Be formatted as a single sentence or short phrase
5. Focus on business impact (customer satisfaction, operational efficiency, team performance)

Return ONLY a JSON array of strings (no markdown, no extra text):
["insight 1", "insight 2", "insight 3", ...]
"""
        
        bedrock_client = _get_bedrock_client()
        response = bedrock_client.converse(
            modelId=PCA_MODEL_ID,
            system=[{"text": "You are a call center analytics expert. Analyze metrics and provide executive-level business insights."}],
            messages=[{"role": "user", "content": [{"text": data_context}]}],
            inferenceConfig={"maxTokens": 1024},
        )
        
        output_text = response["output"]["message"]["content"][0]["text"].strip()
        
        # Parse JSON array from response
        try:
            # Try direct parse
            insights = json.loads(output_text)
            if isinstance(insights, list):
                return insights[:5]  # Return max 5 insights
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON array from text
        start = output_text.find("[")
        end = output_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                insights = json.loads(output_text[start:end + 1])
                if isinstance(insights, list):
                    return insights[:5]
            except json.JSONDecodeError:
                pass
        
        return []
        
    except Exception as e:
        return []


def _get_coaching_priorities_from_raw_data(coaching_priorities_all):
    """
    Aggregate coaching priorities from all calls
    Returns top 5 priorities with count and average score
    """
    if not coaching_priorities_all:
        return []
    
    priority_map = {}
    
    for priority_item in coaching_priorities_all:
        if isinstance(priority_item, dict):
            priority_name = priority_item.get('priority', 'Unknown')
            score = priority_item.get('score', 0)
            
            if priority_name not in priority_map:
                priority_map[priority_name] = {
                    'priority': priority_name,
                    'count': 0,
                    'total_score': 0,
                    'examples': []
                }
            
            priority_map[priority_name]['count'] += 1
            priority_map[priority_name]['total_score'] += float(score) if score else 0
            if 'evidence' in priority_item:
                priority_map[priority_name]['examples'].append(priority_item['evidence'])
    
    # Calculate average scores and sort by count (most frequent first)
    result = []
    for priority_data in priority_map.values():
        avg_score = priority_data['total_score'] / priority_data['count'] if priority_data['count'] > 0 else 0.0
        result.append({
            'rank': 0,  # Will be set later
            'priority': priority_data['priority'],
            'count': priority_data['count'],
            'avg_score': avg_score,
            'details': f"Found in {priority_data['count']} calls, avg score: {avg_score}/10",
            'severity': 'HIGH' if avg_score < 5 else 'MED' if avg_score < 7 else 'LOW'
        })
    
    # Sort by count (descending) and take top 5
    result.sort(key=lambda x: x['count'], reverse=True)
    
    # Set ranks
    for i, item in enumerate(result[:5]):
        item['rank'] = i + 1
    
    return result[:5]


def _get_coaching_priorities(agent_performance_details, agent_scores):
    """
    Generate top coaching priorities based on actual agent performance (Fallback method)
    """
    priorities = []
    
    # Find low performers (agents with performance < 6)
    low_performers = []
    for agent_id, details in agent_performance_details.items():
        if details['low_performers']:
            scores = agent_scores.get(agent_id, {})
            perf_scores = scores.get('performance_scores', [])
            avg_perf = sum(perf_scores) / len(perf_scores) if perf_scores else 0
            low_performers.append({
                'agent_id': agent_id,
                'performance': avg_perf,
                'issues': details['issues'],
                'calls': scores.get('calls', 0)
            })
    
    # Sort by performance score
    low_performers.sort(key=lambda x: x['performance'])
    
    # Generate priority 1: Low performers
    if low_performers:
        agent_count = len(low_performers)
        avg_perf = sum(a['performance'] for a in low_performers) / len(low_performers)
        details_text = f"{agent_count} agent(s), avg score {avg_perf}/10"
        priorities.append({
            'rank': 1,
            'priority': 'Low performance agents',
            'details': details_text,
            'severity': 'HIGH'
        })
    
    # Generate priority 2: SLA compliance issues
    sla_issues = []
    for agent_id, scores in agent_scores.items():
        sla_scores = scores.get('sla_scores', [])
        if sla_scores:
            avg_sla = sum(sla_scores) / len(sla_scores)
            if avg_sla < 60:
                sla_issues.append(agent_id)
    
    if sla_issues:
        priorities.append({
            'rank': 2,
            'priority': 'Weak SLA compliance',
            'details': f"{len(sla_issues)} agent(s) below 60% SLA threshold — review handling efficiency",
            'severity': 'HIGH'
        })
    
    # Generate priority 3: Sentiment/empathy issues
    low_sentiment_agents = []
    for agent_id, details in agent_performance_details.items():
        for call in details['calls']:
            if call.get('sentiment', 0) < 5:
                low_sentiment_agents.append(agent_id)
                break
    
    if low_sentiment_agents:
        unique_agents = len(set(low_sentiment_agents))
        priorities.append({
            'rank': 3,
            'priority': 'Low customer sentiment',
            'details': f"{unique_agents} agent(s) with sentiment issues — improve empathy and active listening",
            'severity': 'MED'
        })
    
    # Generate priority 4: Compliance violations
    compliance_issues = 0
    for agent_id, details in agent_performance_details.items():
        compliance_issues += sum(1 for issue in details['issues'] if 'compliance' in issue.lower())
    
    if compliance_issues > 0:
        priorities.append({
            'rank': 4,
            'priority': 'Compliance violations detected',
            'details': f"{compliance_issues} compliance issue(s) — reinforce policy adherence",
            'severity': 'HIGH'
        })
    
    # Generate priority 5: Development opportunities
    if len(priorities) < 5:
        high_call_agents = sorted(
            agent_scores.items(),
            key=lambda x: x[1]['calls'],
            reverse=True
        )
        
        if high_call_agents:
            top_agent = high_call_agents[0]
            top_agent_id = top_agent[0]
            top_agent_csat = top_agent[1]['satisfaction_scores']
            avg_csat = sum(top_agent_csat) / len(top_agent_csat) if top_agent_csat else 0
            
            if avg_csat > 7:
                priorities.append({
                    'rank': 5,
                    'priority': 'Peer mentoring program',
                    'details': f"Agent {top_agent_id} (CSAT {avg_csat}/10) — excellent candidate for mentoring role",
                    'severity': 'MED'
                })
    
    # Add fallback priorities if needed
    if len(priorities) < 5:
        fallbacks = [
            {
                'rank': 5,
                'priority': 'Knowledge base updates',
                'details': 'Product knowledge varies across team — schedule refresher training',
                'severity': 'MED'
            }
        ]
        for fallback in fallbacks:
            if len(priorities) < 5:
                priorities.append(fallback)
    
    return priorities[:5]


def _get_quality_scorecard(analytics_map, records_map=None):
    """
    Generate AI Quality Scorecard with 8 key performance metrics PER AGENT
    
    Scorecard metrics (0-100 scale):
    1. Greeting - From validation greetings score (0-5 → 0-100)
    2. Understanding - From validation CRM paraphrase (0-5 → 0-100)
    3. CRM Validation - From validation probing accuracy (0-12 → 0-100)
    4. Communication - From validation energy/grammar average (0-5 → 0-100)
    5. Soft Skills - From validation empathy/listening average (0-5 → 0-100)
    6. Compliance - From validation hold process (0-6 → 0-100)
    7. Resolution - From validation percentage (0-100)
    8. Closing - From validation correct_closing (0-6 → 0-100)
    
    Returns: Array of per-agent scorecards
    """
    
    if not analytics_map:
        return []
    
    # Group metrics by agent_id
    agent_metrics = {}
    
    for call_id, analytics in analytics_map.items():
        # Get agent_id from records_map if available
        agent_id = "Unknown"
        if records_map and call_id in records_map:
            agent_id = records_map[call_id].agent_id or "Unknown"
        
        # Initialize agent metrics if not exists
        if agent_id not in agent_metrics:
            agent_metrics[agent_id] = {
                'greeting': [],
                'understanding': [],
                'crm_validation': [],
                'communication': [],
                'soft_skills': [],
                'compliance': [],
                'resolution': [],
                'closing': [],
                'calls': 0
            }
        
        agent_metrics[agent_id]['calls'] += 1
        
        # Extract validation data if available
        if analytics.validation_results and isinstance(analytics.validation_results, dict):
            validation = analytics.validation_results.get('validation', {})
            
            # 1. Greeting (0-5 → 0-100)
            greetings = validation.get('greetings', {})
            if 'score' in greetings and greetings['score'] is not None:
                score = float(greetings['score'])
                normalized = (score / 5) * 100
                agent_metrics[agent_id]['greeting'].append(min(100.0, normalized))
            
            # 2. Understanding - From CRM query paraphrase (0-5 → 0-100)
            crm_query = validation.get('crm_query_paraphrase', {})
            if 'score' in crm_query and crm_query['score'] is not None:
                score = float(crm_query['score'])
                normalized = (score / 5) * 100
                agent_metrics[agent_id]['understanding'].append(min(100.0, normalized))
            
            # 3. CRM Validation - From good_right_probing (0-12 → 0-100)
            probing = validation.get('good_right_probing', {})
            if 'score' in probing and probing['score'] is not None:
                score = float(probing['score'])
                normalized = (score / 12) * 100
                agent_metrics[agent_id]['crm_validation'].append(min(100.0, normalized))
            
            # 4. Communication - Average of energy and grammar (0-5 → 0-100)
            comm_scores = []
            for key in ['energy_enthusiasm_pace', 'grammar_vocabulary']:
                metric = validation.get(key, {})
                if 'score' in metric and metric['score'] is not None:
                    score = float(metric['score'])
                    normalized = (score / 5) * 100
                    comm_scores.append(normalized)
            
            if comm_scores:
                avg_comm = sum(comm_scores) / len(comm_scores)
                agent_metrics[agent_id]['communication'].append(min(100.0, avg_comm))
            
            # 5. Soft Skills - Average of listening and empathy (0-5 → 0-100)
            soft_skill_scores = []
            for key in ['listening_acknowledgment', 'apology_empathy']:
                metric = validation.get(key, {})
                if 'score' in metric and metric['score'] is not None:
                    score = float(metric['score'])
                    normalized = (score / 5) * 100
                    soft_skill_scores.append(normalized)
            
            if soft_skill_scores:
                avg_soft_skill = sum(soft_skill_scores) / len(soft_skill_scores)
                agent_metrics[agent_id]['soft_skills'].append(min(100.0, avg_soft_skill))
            
            # 6. Compliance - From hold process (0-6 → 0-100)
            hold_process = validation.get('dead_air_hold_process', {})
            if 'score' in hold_process and hold_process['score'] is not None:
                score = float(hold_process['score'])
                normalized = (score / 6) * 100
                agent_metrics[agent_id]['compliance'].append(min(100.0, normalized))
            
            # 7. Resolution - From overall validation percentage (0-100)
            if 'percentage' in validation and validation['percentage'] is not None:
                pct = float(validation['percentage'])
                agent_metrics[agent_id]['resolution'].append(min(100.0, pct))
            
            # 8. Closing - From correct_closing (0-6 → 0-100)
            closing = validation.get('correct_closing', {})
            if 'score' in closing and closing['score'] is not None:
                score = float(closing['score'])
                normalized = (score / 6) * 100
                agent_metrics[agent_id]['closing'].append(min(100.0, normalized))
    
    # Calculate per-agent averages
    agent_scorecards = []
    for agent_id, metrics in agent_metrics.items():
        scorecard = {
            'agent_id': agent_id,
            'calls': metrics['calls']
        }
        
        # Calculate averages for each metric (keep as precise floats)
        for metric_name in ['greeting', 'understanding', 'crm_validation', 'communication', 
                           'soft_skills', 'compliance', 'resolution', 'closing']:
            scores = metrics[metric_name]
            if scores:
                avg_score = sum(scores) / len(scores)
                scorecard[metric_name] = _round_float(max(0.0, min(100.0, avg_score)))  # Ensure 0-100 range
            else:
                scorecard[metric_name] = 0.0
        
        agent_scorecards.append(scorecard)
    
    # Sort by agent_id for consistent ordering
    return sorted(agent_scorecards, key=lambda x: x['agent_id'])
