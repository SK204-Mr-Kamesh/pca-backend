"""
ClickHouse persistence for In-Store Interaction Analysis
Separate tables for in-store interactions
"""
import json
import os
import threading
from datetime import datetime, timezone
import clickhouse_connect


INSTORE_RECORDS_TABLE = "wakefit_instore_records"
INSTORE_ANALYTICS_TABLE = "wakefit_instore_analytics"

_thread_local = threading.local()


def _conn_settings():
    return {
        "host": os.getenv("CLICKHOUSE_HOST", ""),
        "port": int(os.getenv("CLICKHOUSE_PORT", "0") or 0),
        "username": os.getenv("CLICKHOUSE_USER", ""),
        "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
    }


def _get_client():
    """Return this thread's ClickHouse client"""
    client = getattr(_thread_local, "client", None)
    if client is None:
        cfg = _conn_settings()
        client = clickhouse_connect.get_client(
            host=cfg["host"],
            port=cfg["port"],
            username=cfg["username"],
            password=cfg["password"],
        )
        _thread_local.client = client
    return client


def ensure_tables():
    """Tables already exist - just verify connection"""
    client = _get_client()
    client.query("SELECT 1")


class InstoreRecord:
    """Represents a single in-store interaction record"""
    
    _FIELDS = [
        'interaction_id', 'store_id', 'sales_executive_id', 'customer_name',
        'status', 'language', 'started_at', 'ended_at', 'duration_seconds',
        'transcript_s3_key', 'recording_s3_key', 'audio_size', 'uploaded_filename',
        'notes', 'created_on'
    ]
    
    def __init__(self, interaction_id, **kwargs):
        self.interaction_id = interaction_id
        self.store_id = kwargs.get('store_id', '')
        self.sales_executive_id = kwargs.get('sales_executive_id', '')
        self.customer_name = kwargs.get('customer_name', '')
        self.status = kwargs.get('status', 'completed')
        self.language = kwargs.get('language', 'english')
        self.started_at = kwargs.get('started_at') or datetime.now(timezone.utc)
        self.ended_at = kwargs.get('ended_at') or datetime.now(timezone.utc)
        self.duration_seconds = kwargs.get('duration_seconds', 0)
        self.transcript_s3_key = kwargs.get('transcript_s3_key', '')
        self.recording_s3_key = kwargs.get('recording_s3_key', '')
        self.audio_size = kwargs.get('audio_size')
        self.uploaded_filename = kwargs.get('uploaded_filename', '')
        self.notes = kwargs.get('notes', '')
        self.created_on = kwargs.get('created_on') or datetime.now(timezone.utc)
    
    def _format_duration(self):
        """Format duration as MM:SS"""
        if not self.duration_seconds:
            return "0m 0s"
        minutes = self.duration_seconds // 60
        seconds = self.duration_seconds % 60
        return f"{minutes}m {seconds}s"
    
    def to_log_dict(self, analytics=None):
        """Convert to frontend log format"""
        from datetime import timedelta, timezone
        
        def to_ist(dt):
            if not dt:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ist_offset = timedelta(hours=5, minutes=30)
            ist_dt = dt.astimezone(timezone.utc) + ist_offset
            return ist_dt.strftime("%d/%m/%Y, %H:%M:%S")
        
        audio_size_mb = None
        if self.audio_size:
            audio_size_mb = round(self.audio_size / (1024 * 1024), 2)
        
        result = {
            'interactionId': self.interaction_id,
            'customerName': self.customer_name or 'Unknown',
            'storeId': self.store_id,
            'salesExecutiveId': self.sales_executive_id,
            'interactionStart': to_ist(self.started_at),
            'interactionDuration': self._format_duration(),
            'status': self.status,
            'language': self.language,
            'uploadedFilename': self.uploaded_filename or '—',
            'audioSize': audio_size_mb,
            'notes': self.notes
        }
        
        if analytics:
            result['sentiment'] = float(analytics.overall_sentiment) if analytics.overall_sentiment else 0
            result['customerSatisfaction'] = float(analytics.customer_satisfaction) if analytics.customer_satisfaction else 0
            result['salesPerformance'] = float(analytics.sales_executive_performance) if analytics.sales_executive_performance else 0
            result['salesOutcome'] = analytics.sales_outcome or '—'
            result['category'] = analytics.category or '—'
        
        return result


class InstoreAnalytics:
    """Represents analytics for an in-store interaction"""
    
    _FIELDS = [
        'interaction_id', 'overall_sentiment', 'customer_satisfaction',
        'sales_executive_performance', 'summary', 'topics', 'action_items',
        'key_indicators', 'customer_name', 'interaction_outcome',
        'interaction_code', 'category', 'sub_category', 'product',
        'sales_outcome', 'l1_pillow', 'l2_pillow', 'l3_pillow',
        'raw_model_response', 'model_id', 'analyzed_at'
    ]
    
    def __init__(self, interaction_id, **kwargs):
        self.interaction_id = interaction_id
        self.overall_sentiment = kwargs.get('overall_sentiment')
        self.customer_satisfaction = kwargs.get('customer_satisfaction')
        self.sales_executive_performance = kwargs.get('sales_executive_performance')
        self.summary = kwargs.get('summary', '')
        self.topics = kwargs.get('topics', [])
        self.action_items = kwargs.get('action_items', [])
        self.key_indicators = kwargs.get('key_indicators', [])
        self.customer_name = kwargs.get('customer_name')
        self.interaction_outcome = kwargs.get('interaction_outcome')
        self.interaction_code = kwargs.get('interaction_code')
        self.category = kwargs.get('category')
        self.sub_category = kwargs.get('sub_category')
        self.product = kwargs.get('product')
        self.sales_outcome = kwargs.get('sales_outcome')
        self.l1_pillow = kwargs.get('l1_pillow')
        self.l2_pillow = kwargs.get('l2_pillow')
        self.l3_pillow = kwargs.get('l3_pillow')
        self.raw_model_response = kwargs.get('raw_model_response')
        self.model_id = kwargs.get('model_id')
        self.analyzed_at = kwargs.get('analyzed_at') or datetime.now(timezone.utc)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'interaction_id': self.interaction_id,
            'overall_sentiment': self.overall_sentiment,
            'customer_satisfaction': self.customer_satisfaction,
            'sales_executive_performance': self.sales_executive_performance,
            'summary': self.summary,
            'topics': self.topics,
            'action_items': self.action_items,
            'key_indicators': self.key_indicators,
            'customer_name': self.customer_name,
            'interaction_outcome': self.interaction_outcome,
            'interaction_code': self.interaction_code,
            'category': self.category,
            'sub_category': self.sub_category,
            'product': self.product,
            'sales_outcome': self.sales_outcome,
            'l1_pillow': self.l1_pillow,
            'l2_pillow': self.l2_pillow,
            'l3_pillow': self.l3_pillow
        }


def upsert_record(record):
    """Insert or update an interaction record"""
    try:
        client = _get_client()
        
        data = [[
            record.interaction_id,
            record.store_id,
            record.sales_executive_id,
            record.customer_name,
            record.status,
            record.language,
            record.started_at,
            record.ended_at,
            record.duration_seconds,
            record.transcript_s3_key,
            record.recording_s3_key,
            record.audio_size or 0,
            record.uploaded_filename,
            record.notes,
            record.created_on
        ]]
        
        client.insert(INSTORE_RECORDS_TABLE, data, column_names=InstoreRecord._FIELDS)
        
    except Exception as e:
        print(f"[InStore-CH] upsert_record failed: {e}")
        raise


def upsert_analytics(analytics):
    """Insert or update interaction analytics"""
    try:
        client = _get_client()
        
        data = [[
            analytics.interaction_id,
            analytics.overall_sentiment,
            analytics.customer_satisfaction,
            analytics.sales_executive_performance,
            analytics.summary,
            analytics.topics,
            analytics.action_items,
            analytics.key_indicators,
            analytics.customer_name,
            analytics.interaction_outcome,
            analytics.interaction_code,
            analytics.category,
            analytics.sub_category,
            analytics.product,
            analytics.sales_outcome,
            analytics.l1_pillow,
            analytics.l2_pillow,
            analytics.l3_pillow,
            json.dumps(analytics.raw_model_response) if analytics.raw_model_response else '{}',
            analytics.model_id,
            analytics.analyzed_at
        ]]
        
        client.insert(INSTORE_ANALYTICS_TABLE, data, column_names=InstoreAnalytics._FIELDS)
        
    except Exception as e:
        print(f"[InStore-CH] upsert_analytics failed: {e}")
        raise


def get_record(interaction_id):
    """Get interaction record by ID"""
    try:
        client = _get_client()
        query = f"""
            SELECT {', '.join(InstoreRecord._FIELDS)}
            FROM {INSTORE_RECORDS_TABLE}
            WHERE interaction_id = %(interaction_id)s
            ORDER BY created_on DESC
            LIMIT 1
        """
        result = client.query(query, parameters={'interaction_id': interaction_id})
        
        if not result.result_rows:
            return None
        
        row = result.result_rows[0]
        return InstoreRecord(
            interaction_id=row[0],
            store_id=row[1],
            sales_executive_id=row[2],
            customer_name=row[3],
            status=row[4],
            language=row[5],
            started_at=row[6],
            ended_at=row[7],
            duration_seconds=row[8],
            transcript_s3_key=row[9],
            recording_s3_key=row[10],
            audio_size=row[11],
            uploaded_filename=row[12],
            notes=row[13],
            created_on=row[14]
        )
    except Exception as e:
        print(f"[InStore-CH] get_record failed: {e}")
        return None


def get_analytics(interaction_id):
    """Get analytics for interaction"""
    try:
        client = _get_client()
        query = f"""
            SELECT {', '.join(InstoreAnalytics._FIELDS)}
            FROM {INSTORE_ANALYTICS_TABLE}
            WHERE interaction_id = %(interaction_id)s
            ORDER BY analyzed_at DESC
            LIMIT 1
        """
        result = client.query(query, parameters={'interaction_id': interaction_id})
        
        if not result.result_rows:
            return None
        
        row = result.result_rows[0]
        raw_response = json.loads(row[18]) if row[18] else {}
        
        return InstoreAnalytics(
            interaction_id=row[0],
            overall_sentiment=row[1],
            customer_satisfaction=row[2],
            sales_executive_performance=row[3],
            summary=row[4],
            topics=row[5],
            action_items=row[6],
            key_indicators=row[7],
            customer_name=row[8],
            interaction_outcome=row[9],
            interaction_code=row[10],
            category=row[11],
            sub_category=row[12],
            product=row[13],
            sales_outcome=row[14],
            l1_pillow=row[15],
            l2_pillow=row[16],
            l3_pillow=row[17],
            raw_model_response=raw_response,
            model_id=row[19],
            analyzed_at=row[20]
        )
    except Exception as e:
        print(f"[InStore-CH] get_analytics failed: {e}")
        return None


def query_records(store_id=None, start_date=None, end_date=None, page=0, limit=10):
    """Query interaction records with pagination"""
    try:
        client = _get_client()
        
        where_clauses = []
        params = {}
        
        if store_id:
            where_clauses.append("store_id = %(store_id)s")
            params['store_id'] = store_id
        
        if start_date:
            where_clauses.append("started_at >= %(start_date)s")
            params['start_date'] = start_date
        
        if end_date:
            where_clauses.append("started_at <= %(end_date)s")
            params['end_date'] = end_date
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Get total count
        count_query = f"SELECT count() FROM {INSTORE_RECORDS_TABLE} WHERE {where_sql}"
        count_result = client.query(count_query, parameters=params)
        total_count = count_result.result_rows[0][0]
        
        # Get paginated results
        offset = page * limit
        query = f"""
            SELECT {', '.join(InstoreRecord._FIELDS)}
            FROM {INSTORE_RECORDS_TABLE}
            WHERE {where_sql}
            ORDER BY created_on DESC
            LIMIT %(limit)s OFFSET %(offset)s
        """
        params['limit'] = limit
        params['offset'] = offset
        
        result = client.query(query, parameters=params)
        
        records = []
        for row in result.result_rows:
            records.append(InstoreRecord(
                interaction_id=row[0],
                store_id=row[1],
                sales_executive_id=row[2],
                customer_name=row[3],
                status=row[4],
                language=row[5],
                started_at=row[6],
                ended_at=row[7],
                duration_seconds=row[8],
                transcript_s3_key=row[9],
                recording_s3_key=row[10],
                audio_size=row[11],
                uploaded_filename=row[12],
                notes=row[13],
                created_on=row[14]
            ))
        
        return records, total_count
        
    except Exception as e:
        print(f"[InStore-CH] query_records failed: {e}")
        return [], 0


def get_analytics_map(interaction_ids):
    """Get analytics for multiple interactions"""
    if not interaction_ids:
        return {}
    
    try:
        client = _get_client()
        
        ids_str = ', '.join(f"'{iid}'" for iid in interaction_ids)
        query = f"""
            SELECT {', '.join(InstoreAnalytics._FIELDS)}
            FROM {INSTORE_ANALYTICS_TABLE}
            WHERE interaction_id IN ({ids_str})
        """
        
        result = client.query(query)
        
        analytics_map = {}
        for row in result.result_rows:
            raw_response = json.loads(row[18]) if row[18] else {}
            analytics_map[row[0]] = InstoreAnalytics(
                interaction_id=row[0],
                overall_sentiment=row[1],
                customer_satisfaction=row[2],
                sales_executive_performance=row[3],
                summary=row[4],
                topics=row[5],
                action_items=row[6],
                key_indicators=row[7],
                customer_name=row[8],
                interaction_outcome=row[9],
                interaction_code=row[10],
                category=row[11],
                sub_category=row[12],
                product=row[13],
                sales_outcome=row[14],
                l1_pillow=row[15],
                l2_pillow=row[16],
                l3_pillow=row[17],
                raw_model_response=raw_response,
                model_id=row[19],
                analyzed_at=row[20]
            )
        
        return analytics_map
        
    except Exception as e:
        print(f"[InStore-CH] get_analytics_map failed: {e}")
        return {}


def delete_interaction(interaction_id):
    """Delete interaction record and analytics"""
    try:
        client = _get_client()
        
        # Delete from both tables
        client.command(
            f"ALTER TABLE {INSTORE_RECORDS_TABLE} DELETE WHERE interaction_id = %(interaction_id)s",
            parameters={'interaction_id': interaction_id}
        )
        
        client.command(
            f"ALTER TABLE {INSTORE_ANALYTICS_TABLE} DELETE WHERE interaction_id = %(interaction_id)s",
            parameters={'interaction_id': interaction_id}
        )
        
        print(f"[InStore-CH] Deleted interaction {interaction_id}")
        return True
        
    except Exception as e:
        print(f"[InStore-CH] delete_interaction failed: {e}")
        return False
