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
        average_duration_seconds = sum(durations) / len(durations) if durations else 0
        average_call_duration_minutes = round(average_duration_seconds / 60, 2)
        
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
        
        # Calculate averages
        average_sentiment = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0
        average_customer_satisfaction = round(sum(customer_satisfactions) / len(customer_satisfactions), 2) if customer_satisfactions else 0
        average_wait_time_seconds = round(sum(wait_times) / len(wait_times), 2) if wait_times else 0
        average_sla_compliance = round(sum(sla_compliances) / len(sla_compliances), 2) if sla_compliances else 0
        average_abandonment_rate = round(sum(abandonment_rates) / len(abandonment_rates), 2) if abandonment_rates else 0
        agent_effectiveness = round(sum(agent_performances) / len(agent_performances), 2) if agent_performances else 0
        
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
        
        # Top coaching priorities
        coaching_priorities = _get_coaching_priorities(agent_performance_details, agent_scores)
        
        # AI Quality Scorecard (8 metrics across key areas)
        quality_scorecard = _get_quality_scorecard(analytics_map)
        
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
            'agent_effectiveness': agent_effectiveness,
            'upload_volume': upload_volume,
            'sentiment_distribution': sentiment_distribution,
            'language_distribution': language_distribution,
            'top_topics': top_topics,
            'agent_leaderboard': leaderboard,
            'executive_summary': executive_summary,
            'coaching_priorities': coaching_priorities,
            'quality_scorecard': quality_scorecard
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
        'quality_scorecard': {
            'greeting': 0.0,
            'understanding': 0.0,
            'crm_validation': 0.0,
            'communication': 0.0,
            'soft_skills': 0.0,
            'compliance': 0.0,
            'resolution': 0.0,
            'closing': 0.0
        }
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
    """Categorize sentiments into positive/neutral/negative"""
    if not sentiments:
        return {'positive': 0, 'neutral': 0, 'negative': 0}
    
    positive = sum(1 for s in sentiments if s >= 7)
    negative = sum(1 for s in sentiments if s <= 3)
    neutral = len(sentiments) - positive - negative
    
    total = len(sentiments)
    return {
        'positive': round((positive / total) * 100, 1) if total > 0 else 0,
        'neutral': round((neutral / total) * 100, 1) if total > 0 else 0,
        'negative': round((negative / total) * 100, 1) if total > 0 else 0
    }


def _get_language_distribution(languages):
    """Get distribution of detected languages"""
    if not languages:
        return {}
    
    language_counts = Counter(languages)
    total = len(languages)
    
    return {
        lang: round((count / total) * 100, 1) 
        for lang, count in language_counts.most_common()
    }


def _get_top_topics(topics_list):
    """Get top 5 most common topics"""
    if not topics_list:
        return []
    
    topic_counts = Counter(topics_list)
    return [
        {'topic': topic, 'count': count}
        for topic, count in topic_counts.most_common(5)
    ]


def _get_agent_leaderboard(agent_scores):
    """Build agent effectiveness leaderboard"""
    leaderboard = []
    
    for agent_id, scores in agent_scores.items():
        avg_score = round(sum(scores['performance_scores']) / len(scores['performance_scores']), 1) if scores['performance_scores'] else 0
        avg_csat = round(sum(scores['satisfaction_scores']) / len(scores['satisfaction_scores']), 1) if scores['satisfaction_scores'] else 0
        avg_sla = round(sum(scores['sla_scores']) / len(scores['sla_scores']), 1) if scores['sla_scores'] else 0
        avg_sentiment = round(sum(scores['sentiment_scores']) / len(scores['sentiment_scores']), 1) if scores['sentiment_scores'] else 0
        compliance_rate = round(((scores['calls'] - scores['compliance_count']) / scores['calls']) * 100, 1) if scores['calls'] > 0 else 0
        
        leaderboard.append({
            'agent_id': agent_id,
            'calls': scores['calls'],
            'score': avg_score,
            'csat': avg_csat,
            'fcr': avg_sla,  # First Call Resolution
            'compliance': compliance_rate,
            'sentiment': avg_sentiment
        })
    
    # Sort by score descending
    return sorted(leaderboard, key=lambda x: x['score'], reverse=True)[:10]


def _get_executive_summary(sentiments, sla_compliances, abandonment_rates, total_calls):
    """Generate executive summary insights using Claude LLM for intelligent analysis"""
    
    if not sentiments and not sla_compliances:
        return []
    
    try:
        # Prepare data summary for LLM
        avg_sentiment = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0
        negative_count = sum(1 for s in sentiments if s <= 3) if sentiments else 0
        negative_pct = round((negative_count / len(sentiments)) * 100, 1) if sentiments else 0
        positive_count = sum(1 for s in sentiments if s >= 7) if sentiments else 0
        positive_pct = round((positive_count / len(sentiments)) * 100, 1) if sentiments else 0
        
        avg_sla = round(sum(sla_compliances) / len(sla_compliances), 2) if sla_compliances else 0
        high_sla_count = sum(1 for s in sla_compliances if s >= 80) if sla_compliances else 0
        high_sla_pct = round((high_sla_count / len(sla_compliances)) * 100, 1) if sla_compliances else 0
        low_sla_count = sum(1 for s in sla_compliances if s < 60) if sla_compliances else 0
        low_sla_pct = round((low_sla_count / len(sla_compliances)) * 100, 1) if sla_compliances else 0
        
        avg_abandonment = round(sum(abandonment_rates) / len(abandonment_rates), 2) if abandonment_rates else 0
        abandoned_count = sum(1 for a in abandonment_rates if a > 0) if abandonment_rates else 0
        abandoned_pct = round((abandoned_count / len(abandonment_rates)) * 100, 1) if abandonment_rates else 0
        
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
        
        print("[Analytics] Failed to parse LLM executive summary, returning empty")
        return []
        
    except Exception as e:
        print(f"[Analytics] Executive summary generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def _get_coaching_priorities(agent_performance_details, agent_scores):
    """Generate top coaching priorities based on actual agent performance"""
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
                'performance': round(avg_perf, 1),
                'issues': details['issues'],
                'calls': scores.get('calls', 0)
            })
    
    # Sort by performance score
    low_performers.sort(key=lambda x: x['performance'])
    
    # Generate priority 1: Low performers
    if low_performers:
        worst_agent = low_performers[0]
        agent_count = len(low_performers)
        details_text = f"{agent_count} agent(s), avg score {round(sum(a['performance'] for a in low_performers) / len(low_performers), 1)}/10"
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
                    'details': f"Agent {top_agent_id} (CSAT {round(avg_csat, 1)}/10) — excellent candidate for mentoring role",
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


def _get_quality_scorecard(analytics_map):
    """
    Generate AI Quality Scorecard with 8 key performance metrics
    
    Scorecard metrics (0-10 scale):
    1. Greeting - From validation greetings score
    2. Understanding - From validation discovery/CRM paraphrase
    3. CRM Validation - From validation probing accuracy
    4. Communication - From validation communication score
    5. Soft Skills - Aggregate soft skills compliance
    6. Compliance - From compliance flags violations
    7. Resolution - From validation closing/resolution
    8. Closing - From validation correct_closing
    """
    
    if not analytics_map:
        return {}
    
    # Initialize metric collections
    metrics = {
        'greeting': [],
        'understanding': [],
        'crm_validation': [],
        'communication': [],
        'soft_skills': [],
        'compliance': [],
        'resolution': [],
        'closing': []
    }
    
    for call_id, analytics in analytics_map.items():
        # Extract validation data if available
        if analytics.validation_results and isinstance(analytics.validation_results, dict):
            validation = analytics.validation_results.get('validation', {})
            
            # 1. Greeting - From validation greetings marking
            greetings = validation.get('greetings', {})
            if 'score' in greetings:
                score = greetings['score']
                if isinstance(score, (int, float)):
                    # Normalize 0-5 to 0-10
                    normalized = (score / 5) * 10 if score > 0 else 0
                    metrics['greeting'].append(min(10, round(normalized, 1)))
            
            # 2. Understanding - From CRM query paraphrase
            crm_query = validation.get('crm_query_paraphrase', {})
            if 'score' in crm_query:
                score = crm_query['score']
                if isinstance(score, (int, float)):
                    normalized = (score / 5) * 10 if score > 0 else 0
                    metrics['understanding'].append(min(10, round(normalized, 1)))
            
            # 3. CRM Validation - From good_right_probing
            probing = validation.get('good_right_probing', {})
            if 'score' in probing:
                score = probing['score']
                if isinstance(score, (int, float)):
                    # good_right_probing is 0-12 scale, normalize to 0-10
                    normalized = (score / 12) * 10 if score > 0 else 0
                    metrics['crm_validation'].append(min(10, round(normalized, 1)))
            
            # 4. Communication - From communication score
            comm = validation.get('communication', {})
            if 'score' in comm:
                score = comm['score']
                if isinstance(score, (int, float)):
                    normalized = (score / 5) * 10 if score > 0 else 0
                    metrics['communication'].append(min(10, round(normalized, 1)))
            
            # 5. Soft Skills - Average of energy, listening, grammar, empathy
            soft_skill_scores = []
            for metric_key in ['energy_enthusiasm_pace', 'listening_acknowledgment', 'grammar_vocabulary', 'apology_empathy']:
                metric = validation.get(metric_key, {})
                if 'score' in metric:
                    score = metric['score']
                    if isinstance(score, (int, float)):
                        normalized = (score / 5) * 10 if score > 0 else 0
                        soft_skill_scores.append(min(10, normalized))
            
            if soft_skill_scores:
                avg_soft_skill = round(sum(soft_skill_scores) / len(soft_skill_scores), 1)
                metrics['soft_skills'].append(avg_soft_skill)
            
            # 6. Compliance - From dead_air_hold_process
            hold_process = validation.get('dead_air_hold_process', {})
            if 'score' in hold_process:
                score = hold_process['score']
                if isinstance(score, (int, float)):
                    # dead_air is 0-6 scale, normalize to 0-10
                    normalized = (score / 6) * 10 if score > 0 else 0
                    metrics['compliance'].append(min(10, round(normalized, 1)))
            
            # 7. Resolution - We'll use overall validation percentage as resolution score
            if 'percentage' in validation:
                pct = validation['percentage']
                if isinstance(pct, (int, float)):
                    # percentage is 0-100, convert to 0-10
                    score = (pct / 100) * 10
                    metrics['resolution'].append(round(score, 1))
            
            # 8. Closing - From correct_closing
            closing = validation.get('correct_closing', {})
            if 'score' in closing:
                score = closing['score']
                if isinstance(score, (int, float)):
                    # correct_closing is 0-6 scale, normalize to 0-10
                    normalized = (score / 6) * 10 if score > 0 else 0
                    metrics['closing'].append(min(10, round(normalized, 1)))
    
    # Calculate averages for each metric
    scorecard = {}
    for metric_name, scores in metrics.items():
        if scores:
            avg_score = round(sum(scores) / len(scores), 1)
            scorecard[metric_name] = avg_score
        else:
            scorecard[metric_name] = 0.0
    
    return scorecard
