"""
In-Store Analytics Service
Provides aggregate analytics and metrics for in-store interaction records
"""
import json
import os
import boto3
from datetime import datetime, timedelta, timezone
from collections import Counter
import instore_clickhouse as ch

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


def get_instore_analytics():
    """
    Get comprehensive in-store analytics including all dashboard metrics
    """
    try:
        # Get all records (pass None for store_id to get all records)
        all_records, total_count = ch.query_records(store_id=None, start_date=None, end_date=None, page=0, limit=999999)
        
        total_uploads = total_count
        
        if total_uploads == 0:
            return _get_empty_analytics()
        
        # Get analytics for all records
        interaction_ids = [r.interaction_id for r in all_records]
        analytics_map = ch.get_analytics_map(interaction_ids)
        
        ready = len(analytics_map)
        failed = total_uploads - ready
        
        # Create a map for quick record lookup
        records_map = {r.interaction_id: r for r in all_records}
        
        # Basic metrics
        durations = [r.duration_seconds or 0 for r in all_records if r.duration_seconds]
        average_duration_seconds = sum(durations) / len(durations) if durations else 0.0
        average_interaction_duration_minutes = average_duration_seconds / 60
        
        # Initialize collections
        customer_satisfactions = []
        sales_executive_performances = []
        overall_sentiments = []
        sla_compliances = []
        topics_list = []
        languages = []
        sales_executive_scores = {}  # For leaderboard
        sales_executive_details = {}  # For coaching priorities
        coaching_priorities_all = []  # Collect all coaching priorities
        
        for interaction_id, analytics in analytics_map.items():
            # Find corresponding record for sales_executive_id
            record = records_map.get(interaction_id)
            sales_executive_id = record.sales_executive_id if record else "Unknown"
            
            # Initialize sales executive tracking
            if sales_executive_id not in sales_executive_details:
                sales_executive_details[sales_executive_id] = {
                    'interactions': [],
                    'low_performers': False,
                    'issues': []
                }
            
            # Customer satisfaction
            csat = None
            if analytics.customer_satisfaction is not None:
                try:
                    csat = float(analytics.customer_satisfaction)
                    customer_satisfactions.append(csat)
                except (TypeError, ValueError):
                    pass
            
            # Sales executive performance
            sales_perf = None
            if analytics.sales_executive_performance is not None:
                try:
                    sales_perf = float(analytics.sales_executive_performance)
                    sales_executive_performances.append(sales_perf)
                    if sales_perf < 6:
                        sales_executive_details[sales_executive_id]['low_performers'] = True
                except (TypeError, ValueError):
                    pass
            
            # Overall sentiment
            sentiment = None
            if analytics.overall_sentiment is not None:
                try:
                    sentiment = float(analytics.overall_sentiment)
                    overall_sentiments.append(sentiment)
                except (TypeError, ValueError):
                    pass
            
            # Products discussed
            if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
                products = analytics.raw_model_response.get('products_discussed', [])
                if products and isinstance(products, list):
                    for product in products:
                        if isinstance(product, dict) and 'product_name' in product:
                            topics_list.append(product.get('product_name'))
                
                # SLA compliance
                sla_compliance = analytics.raw_model_response.get('sla_compliance')
                if sla_compliance is not None:
                    try:
                        sla_val = float(sla_compliance)
                        sla_compliances.append(sla_val)
                        if sla_val < 60:
                            sales_executive_details[sales_executive_id]['issues'].append('Low SLA compliance')
                    except (TypeError, ValueError):
                        pass
                
                # Learning suggestions
                learning_suggestions = analytics.raw_model_response.get('learning_suggestions')
                if learning_suggestions:
                    sales_executive_details[sales_executive_id]['issues'].append(learning_suggestions)
                
                # Collect coaching priorities from LLM response
                coaching_priorities = analytics.raw_model_response.get('coaching_priorities', [])
                if coaching_priorities and isinstance(coaching_priorities, list):
                    coaching_priorities_all.extend(coaching_priorities)
            
            # Language
            if record and record.language:
                languages.append(record.language)
            
            # Build sales executive leaderboard data
            if sales_executive_id not in sales_executive_scores:
                sales_executive_scores[sales_executive_id] = {
                    'interactions': 0,
                    'performance_scores': [],
                    'satisfaction_scores': [],
                    'sentiment_scores': [],
                    'sales_outcomes': []
                }
            
            sales_executive_scores[sales_executive_id]['interactions'] += 1
            if sales_perf is not None:
                sales_executive_scores[sales_executive_id]['performance_scores'].append(sales_perf)
            if csat is not None:
                sales_executive_scores[sales_executive_id]['satisfaction_scores'].append(csat)
            if sentiment is not None:
                sales_executive_scores[sales_executive_id]['sentiment_scores'].append(sentiment)
            
            # Track SLA compliance
            if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
                sla_val = analytics.raw_model_response.get('sla_compliance')
                if sla_val is not None:
                    sales_executive_scores[sales_executive_id].setdefault('sla_scores', []).append(float(sla_val))
            
            # Track sales outcomes
            if analytics.sales_outcome:
                sales_executive_scores[sales_executive_id]['sales_outcomes'].append(analytics.sales_outcome)
        
        # Calculate averages
        average_customer_satisfaction = sum(customer_satisfactions) / len(customer_satisfactions) if customer_satisfactions else 0.0
        average_sales_executive_performance = sum(sales_executive_performances) / len(sales_executive_performances) if sales_executive_performances else 0.0
        average_overall_sentiment = sum(overall_sentiments) / len(overall_sentiments) if overall_sentiments else 0.0
        average_sla_compliance = sum(sla_compliances) / len(sla_compliances) if sla_compliances else 0.0
        
        # Upload volume (per day for last 7 days)
        upload_volume = _get_upload_volume_trend(all_records)
        
        # Sentiment distribution (positive/neutral/negative) - based on overall_sentiment
        sentiment_distribution = _get_sentiment_distribution(overall_sentiments)
        
        # Language distribution
        language_distribution = _get_language_distribution(languages)
        
        # Top products discussed
        top_products = _get_top_topics(topics_list)
        
        # Sales executive effectiveness leaderboard
        leaderboard = _get_sales_executive_leaderboard(sales_executive_scores)
        
        # Executive AI summary (top insights)
        executive_summary = _get_executive_summary(
            customer_satisfactions,
            sales_executive_performances,
            overall_sentiments,
            len(analytics_map)
        )
        
        # Coaching priorities (try raw data first, fallback to analysis)
        coaching_priorities = _get_coaching_priorities_from_raw_data(coaching_priorities_all)
        if not coaching_priorities:
            coaching_priorities = _get_coaching_priorities(sales_executive_details, sales_executive_scores)
        
        # AI Quality Scorecard (5 metrics per sales executive)
        quality_scorecard = _get_quality_scorecard(analytics_map, records_map)
        
        return {
            'total_uploads': total_uploads,
            'ready': ready,
            'failed': failed,
            'average_interaction_duration_minutes': _round_float(average_interaction_duration_minutes),
            'average_overall_sentiment': _round_float(average_overall_sentiment),
            'average_customer_satisfaction': _round_float(average_customer_satisfaction),
            'average_sales_executive_performance': _round_float(average_sales_executive_performance),
            'average_sla_compliance': _round_float(average_sla_compliance),
            'upload_volume': upload_volume,
            'sentiment_distribution': {
                'positive': _round_float(sentiment_distribution['positive']),
                'neutral': _round_float(sentiment_distribution['neutral']),
                'negative': _round_float(sentiment_distribution['negative'])
            },
            'language_distribution': {lang: _round_float(pct) for lang, pct in language_distribution.items()},
            'top_products': top_products,
            'sales_executive_leaderboard': [{
                **exec,
                'score': _round_float(exec['score']),
                'csat': _round_float(exec['csat']),
                'sla_compliance': _round_float(exec['sla_compliance']),
                'conversion_rate': _round_float(exec['conversion_rate']),
                'sentiment': _round_float(exec['sentiment'])
            } for exec in leaderboard],
            'executive_summary': executive_summary,
            'coaching_priorities': coaching_priorities,
            'quality_scorecard': [{
                **scorecard,
                'communication': _round_float(scorecard['communication']),
                'discovery': _round_float(scorecard['discovery']),
                'solution_fit': _round_float(scorecard['solution_fit']),
                'sales_execution': _round_float(scorecard['sales_execution']),
                'customer_experience': _round_float(scorecard['customer_experience'])
            } for scorecard in quality_scorecard]
        }
        
    except Exception as e:
        print(f"[InStore Analytics] Error getting analytics: {e}")
        import traceback
        traceback.print_exc()
        return _get_empty_analytics()


def _get_empty_analytics():
    """Return empty analytics structure"""
    return {
        'total_uploads': 0,
        'ready': 0,
        'failed': 0,
        'average_interaction_duration_minutes': 0,
        'average_overall_sentiment': 0,
        'average_customer_satisfaction': 0,
        'average_sales_executive_performance': 0,
        'average_sla_compliance': 0,
        'upload_volume': [],
        'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
        'language_distribution': {},
        'top_topics': [],
        'sales_executive_leaderboard': [],
        'executive_summary': [],
        'coaching_priorities': [],
        'quality_scorecard': []
    }


def _get_upload_volume_trend(records):
    """Get upload volume per day for last 7 days in IST"""
    trend = {}
    
    # IST is UTC+5:30
    ist_offset = timezone(timedelta(hours=5, minutes=30))
    today_ist = datetime.now(ist_offset).date()
    
    for i in range(7):
        date = (today_ist - timedelta(days=i)).isoformat()
        trend[date] = 0
    
    for record in records:
        if record.created_on:
            # Convert UTC timestamp to IST
            if record.created_on.tzinfo is None:
                # If naive, assume UTC
                utc_time = record.created_on.replace(tzinfo=timezone.utc)
            else:
                utc_time = record.created_on
            
            ist_time = utc_time.astimezone(ist_offset)
            date = ist_time.date().isoformat()
            if date in trend:
                trend[date] += 1
    
    return [{'date': k, 'count': v} for k, v in sorted(trend.items())]


def _get_sentiment_distribution(sentiments):
    """Categorize sentiments into positive/neutral/negative"""
    if not sentiments:
        return {'positive': 0, 'neutral': 0, 'negative': 0}
    
    # Thresholds: positive ≥7, neutral 3-7, negative <3
    positive = sum(1 for s in sentiments if s >= 7)
    negative = sum(1 for s in sentiments if s < 3)
    neutral = sum(1 for s in sentiments if 3 <= s < 7)
    
    total = len(sentiments)
    
    # Calculate percentages
    pos_pct = (positive / total) * 100 if total > 0 else 0.0
    neg_pct = (negative / total) * 100 if total > 0 else 0.0
    neu_pct = (neutral / total) * 100 if total > 0 else 0.0
    
    # Adjust for rounding to ensure total = 100%
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


def _get_top_topics(topics_list):
    """Get top 5 most discussed products"""
    if not topics_list:
        return []
    
    topic_counts = Counter(topics_list)
    
    return [
        {
            'product': product,
            'count': count,
        }
        for product, count in topic_counts.most_common(5)
    ]


def _get_sales_executive_leaderboard(sales_executive_scores):
    """Build sales executive effectiveness leaderboard"""
    leaderboard = []
    
    for exec_id, scores in sales_executive_scores.items():
        avg_score = sum(scores['performance_scores']) / len(scores['performance_scores']) if scores['performance_scores'] else 0.0
        avg_csat = sum(scores['satisfaction_scores']) / len(scores['satisfaction_scores']) if scores['satisfaction_scores'] else 0.0
        avg_sentiment = sum(scores['sentiment_scores']) / len(scores['sentiment_scores']) if scores['sentiment_scores'] else 0.0
        avg_sla = sum(scores.get('sla_scores', [])) / len(scores.get('sla_scores', [1])) if scores.get('sla_scores') else 0.0
        
        # Calculate conversion rate (successful sales outcomes)
        successful_outcomes = sum(1 for outcome in scores['sales_outcomes'] if outcome in ['successful', 'purchased'])
        total_interactions = scores['interactions']
        conversion_rate = (successful_outcomes / total_interactions) * 100 if total_interactions > 0 else 0.0
        
        leaderboard.append({
            'sales_executive_id': exec_id,
            'interactions': scores['interactions'],
            'score': _round_float(avg_score),
            'csat': _round_float(avg_csat),
            'sla_compliance': _round_float(avg_sla),
            'conversion_rate': _round_float(conversion_rate),
            'sentiment': _round_float(avg_sentiment)
        })
    
    # Sort by score descending
    return sorted(leaderboard, key=lambda x: x['score'], reverse=True)[:10]


def _get_executive_summary(customer_satisfactions, sales_performances, sentiments, total_interactions):
    """Generate executive summary insights using Claude LLM"""
    
    if not sentiments and not customer_satisfactions:
        return []
    
    try:
        # Prepare data summary for LLM
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        negative_count = sum(1 for s in sentiments if s <= 3) if sentiments else 0
        negative_pct = (negative_count / len(sentiments)) * 100 if sentiments else 0.0
        positive_count = sum(1 for s in sentiments if s >= 7) if sentiments else 0
        positive_pct = (positive_count / len(sentiments)) * 100 if sentiments else 0.0
        
        avg_csat = sum(customer_satisfactions) / len(customer_satisfactions) if customer_satisfactions else 0.0
        avg_sales_perf = sum(sales_performances) / len(sales_performances) if sales_performances else 0.0
        
        # Create context for LLM
        data_context = f"""
In-Store Sales Interaction Analytics Summary (Last Period):
- Total Interactions Analyzed: {total_interactions}
- Average Overall Sentiment: {avg_sentiment}/10
  * Positive sentiment (≥7): {positive_pct}% of interactions
  * Negative sentiment (≤3): {negative_pct}% of interactions
- Average Customer Satisfaction: {avg_csat}/10
- Average Sales Executive Performance: {avg_sales_perf}/10

Please analyze this data and generate 3-5 concise, actionable business insights for the executive team.
Each insight should:
1. Be data-driven and specific (include numbers)
2. Highlight trends or concerns
3. Suggest action if needed
4. Be formatted as a single sentence or short phrase
5. Focus on business impact (customer satisfaction, sales performance, team effectiveness)

Return ONLY a JSON array of strings (no markdown, no extra text):
["insight 1", "insight 2", "insight 3", ...]
"""
        
        bedrock_client = _get_bedrock_client()
        response = bedrock_client.converse(
            modelId=PCA_MODEL_ID,
            system=[{"text": "You are an in-store sales analytics expert. Analyze metrics and provide executive-level business insights for retail operations."}],
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
        print(f"[InStore Analytics] Executive summary generation failed: {e}")
        return []


def _get_coaching_priorities_from_raw_data(coaching_priorities_all):
    """
    Aggregate coaching priorities from all interactions
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
            'avg_score': round(avg_score, 2),
            'details': f"Found in {priority_data['count']} interactions, avg score: {avg_score:.1f}/10",
            'severity': 'HIGH' if avg_score < 5 else 'MED' if avg_score < 7 else 'LOW'
        })
    
    # Sort by count (descending) and take top 5
    result.sort(key=lambda x: x['count'], reverse=True)
    
    # Set ranks
    for i, item in enumerate(result[:5]):
        item['rank'] = i + 1
    
    return result[:5]


def _get_coaching_priorities(sales_executive_details, sales_executive_scores):
    """
    Generate top coaching priorities based on actual sales executive performance (Fallback method)
    """
    priorities = []
    
    # Find low performers (executives with performance < 6)
    low_performers = []
    for exec_id, details in sales_executive_details.items():
        if details['low_performers']:
            scores = sales_executive_scores.get(exec_id, {})
            perf_scores = scores.get('performance_scores', [])
            avg_perf = sum(perf_scores) / len(perf_scores) if perf_scores else 0
            low_performers.append({
                'exec_id': exec_id,
                'performance': avg_perf,
                'issues': details['issues'],
                'interactions': scores.get('interactions', 0)
            })
    
    # Sort by performance score
    low_performers.sort(key=lambda x: x['performance'])
    
    # Generate priority 1: Low performers
    if low_performers:
        exec_count = len(low_performers)
        avg_perf = sum(e['performance'] for e in low_performers) / len(low_performers)
        details_text = f"{exec_count} sales executive(s), avg score {avg_perf:.1f}/10"
        priorities.append({
            'rank': 1,
            'priority': 'Low performance sales executives',
            'details': details_text,
            'severity': 'HIGH'
        })
    
    # Generate priority 2: SLA compliance issues
    sla_issues = []
    for exec_id, scores in sales_executive_scores.items():
        sla_scores = scores.get('sla_scores', [])
        if sla_scores:
            avg_sla = sum(sla_scores) / len(sla_scores)
            if avg_sla < 60:
                sla_issues.append(exec_id)
    
    if sla_issues:
        priorities.append({
            'rank': 2,
            'priority': 'Weak SLA compliance',
            'details': f"{len(sla_issues)} sales executive(s) below 60% SLA threshold — review service standards",
            'severity': 'HIGH'
        })
    
    # Generate priority 3: Sentiment/customer satisfaction issues
    low_sentiment_execs = []
    for exec_id, scores in sales_executive_scores.items():
        satisfaction_scores = scores.get('satisfaction_scores', [])
        if satisfaction_scores:
            avg_csat = sum(satisfaction_scores) / len(satisfaction_scores)
            if avg_csat < 5:
                low_sentiment_execs.append(exec_id)
    
    if low_sentiment_execs:
        unique_execs = len(set(low_sentiment_execs))
        priorities.append({
            'rank': 3,
            'priority': 'Low customer satisfaction',
            'details': f"{unique_execs} sales executive(s) with low CSAT — improve empathy and customer engagement",
            'severity': 'MED'
        })
    
    # Generate priority 4: Compliance violations
    compliance_issues = 0
    for exec_id, details in sales_executive_details.items():
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
        high_interaction_execs = sorted(
            sales_executive_scores.items(),
            key=lambda x: x[1]['interactions'],
            reverse=True
        )
        
        if high_interaction_execs:
            top_exec = high_interaction_execs[0]
            top_exec_id = top_exec[0]
            top_exec_csat = top_exec[1]['satisfaction_scores']
            avg_csat = sum(top_exec_csat) / len(top_exec_csat) if top_exec_csat else 0
            
            if avg_csat > 7:
                priorities.append({
                    'rank': 5,
                    'priority': 'Peer mentoring program',
                    'details': f"Sales Executive {top_exec_id} (CSAT {avg_csat:.1f}/10) — excellent candidate for mentoring role",
                    'severity': 'MED'
                })
    
    # Add fallback priorities if needed
    if len(priorities) < 5:
        fallbacks = [
            {
                'rank': len(priorities) + 1,
                'priority': 'Product knowledge updates',
                'details': 'Product knowledge varies across team — schedule refresher training on new collections',
                'severity': 'MED'
            },
            {
                'rank': len(priorities) + 2,
                'priority': 'Sales process optimization',
                'details': 'Review discovery and closing techniques — standardize best practices across team',
                'severity': 'LOW'
            }
        ]
        for fallback in fallbacks:
            if len(priorities) < 5:
                fallback['rank'] = len(priorities) + 1
                priorities.append(fallback)
    
    return priorities[:5]


def _get_quality_scorecard(analytics_map, records_map):
    """
    Generate AI Quality Scorecard with 5 key performance metrics PER SALES EXECUTIVE
    
    Scorecard metrics (0-100 scale):
    1. Communication - Clarity, tone, listening, professionalism (0-10 → 0-100)
    2. Discovery - Understanding customer needs and asking right questions (0-10 → 0-100)
    3. Solution Fit - How well product recommendations matched customer needs (0-10 → 0-100)
    4. Sales Execution - Handling objections, closing, next steps (0-10 → 0-100)
    5. Customer Experience - Overall impression, respect, value, satisfaction (0-10 → 0-100)
    
    Returns: Array of per-sales-executive scorecards
    """
    
    if not analytics_map:
        return []
    
    # Group metrics by sales_executive_id
    executive_metrics = {}
    
    for interaction_id, analytics in analytics_map.items():
        # Get sales_executive_id from records_map if available
        sales_executive_id = "Unknown"
        if records_map and interaction_id in records_map:
            sales_executive_id = records_map[interaction_id].sales_executive_id or "Unknown"
        
        # Initialize executive metrics if not exists
        if sales_executive_id not in executive_metrics:
            executive_metrics[sales_executive_id] = {
                'communication': [],
                'discovery': [],
                'solution_fit': [],
                'sales_execution': [],
                'customer_experience': [],
                'interactions': 0
            }
        
        executive_metrics[sales_executive_id]['interactions'] += 1
        
        # Extract interaction_matrices data if available
        if analytics.raw_model_response and isinstance(analytics.raw_model_response, dict):
            interaction_matrices = analytics.raw_model_response.get('interaction_matrices', {})
            
            # Extract each of the 5 metrics (0-10 scale → 0-100 scale)
            for metric_name in ['communication', 'discovery', 'solution_fit', 'sales_execution', 'customer_experience']:
                metric_data = interaction_matrices.get(metric_name, {})
                if isinstance(metric_data, dict) and 'score' in metric_data:
                    try:
                        score = float(metric_data['score'])
                        # Convert from 0-10 to 0-100 scale
                        normalized = (score / 10) * 100
                        executive_metrics[sales_executive_id][metric_name].append(min(100.0, max(0.0, normalized)))
                    except (TypeError, ValueError):
                        pass
    
    # Calculate per-executive averages
    executive_scorecards = []
    for exec_id, metrics in executive_metrics.items():
        scorecard = {
            'sales_executive_id': exec_id,
            'interactions': metrics['interactions']
        }
        
        # Calculate averages for each metric (0-100 scale)
        for metric_name in ['communication', 'discovery', 'solution_fit', 'sales_execution', 'customer_experience']:
            scores = metrics[metric_name]
            if scores:
                avg_score = sum(scores) / len(scores)
                scorecard[metric_name] = _round_float(max(0.0, min(100.0, avg_score)))  # Ensure 0-100 range
            else:
                scorecard[metric_name] = 0.0
        
        executive_scorecards.append(scorecard)
    
    # Sort by sales_executive_id for consistent ordering
    return sorted(executive_scorecards, key=lambda x: x['sales_executive_id'])
