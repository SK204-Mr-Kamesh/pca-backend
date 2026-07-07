"""
ClickHouse persistence for Post-Call Analytics (PCA) - Standalone version
Copied from main platform pca_clickhouse.py
"""
import json
import os
import threading
from datetime import datetime, timezone
import clickhouse_connect


RECORDS_TABLE = "voice_call_records"
ANALYTICS_TABLE = "voice_call_analytics"

_thread_local = threading.local()


def _conn_settings():
    return {
        "host": os.getenv("CLICKHOUSE_HOST", ""),
        "port": int(os.getenv("CLICKHOUSE_PORT", "0") or 0),
        "username": os.getenv("CLICKHOUSE_USER", ""),
        "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
    }


def _get_client():
    """Return this thread's ClickHouse client, creating it on first use."""
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


def _reset_client():
    """Drop this thread's client so the next call reconnects cleanly."""
    _thread_local.client = None


def ensure_tables():
    """Tables already exist - just verify connection"""
    client = _get_client()
    client.query("SELECT 1")
    print("[PCA-CH] ClickHouse connection verified")


# ── Row wrappers ─────────────────────────────────────────────────────────────
class CallRecord:
    """Call record matching main platform interface"""

    _FIELDS = [
        "call_id", "agent_id", "account_id", "room_name", "from_phone", "to_phone",
        "call_source", "status", "language", "started_at", "ended_at",
        "duration_seconds", "transcript_s3_key", "recording_s3_key", "created_on",
    ]

    def __init__(self, call_id, agent_id=None, account_id=None, room_name=None,
                 from_phone=None, to_phone=None, call_source="livekit",
                 status="answered", language=None, started_at=None, ended_at=None,
                 duration_seconds=0, transcript_s3_key=None, recording_s3_key=None,
                 created_on=None):
        self.call_id = call_id
        self.agent_id = agent_id
        self.account_id = account_id
        self.room_name = room_name
        self.from_phone = from_phone
        self.to_phone = to_phone
        self.call_source = call_source
        self.status = status
        self.language = language
        self.started_at = started_at
        self.ended_at = ended_at
        self.duration_seconds = duration_seconds or 0
        self.transcript_s3_key = transcript_s3_key
        self.recording_s3_key = recording_s3_key
        self.created_on = created_on

    def _format_duration(self):
        secs = self.duration_seconds or 0
        return f"{secs // 60}m {secs % 60}s"

    def to_log_dict(self, analytics=None):
        """Frontend call-logs table row format"""
        return {
            "callId": self.call_id,
            "customerName": (analytics.customer_name if analytics and analytics.customer_name else None) or self.from_phone or "Unknown",
            "phoneNumber": self.from_phone or self.to_phone or "—",
            "hangupReason": (analytics.hangup_reason if analytics and analytics.hangup_reason else None) or "—",
            "language": self.language or "—",
            "callDuration": self._format_duration(),
            "status": self.status,
            "callStart": self.created_on.strftime('%d/%m/%Y, %H:%M:%S') if self.created_on else "—",
            "sentiment": float(analytics.overall_sentiment) if analytics and analytics.overall_sentiment else 0,
            "customerSatisfaction": float(analytics.customer_satisfaction) if analytics and analytics.customer_satisfaction else 0,
            "agentPerformance": float(analytics.agent_performance) if analytics and analytics.agent_performance else 0,
        }

    @staticmethod
    def _format_sentiment(analytics):
        if not analytics or analytics.overall_sentiment is None:
            return "—"
        score = float(analytics.overall_sentiment)
        label = "Positive" if score >= 7 else "Negative" if score < 4 else "Neutral"
        return f"{label} ({score:.1f}/10)"


class CallAnalytics:
    """Analytics row matching main platform interface"""

    def __init__(self, call_id, overall_sentiment=None, customer_satisfaction=None,
                 agent_performance=None, summary="", topics=None, action_items=None,
                 key_indicators=None, call_matrices=None, customer_name=None,
                 hangup_reason=None, raw_model_response=None, model_id=None,
                 validation_results=None, validation_score=None, validation_percentage=None,
                 skill_level=None, created_on=None):
        self.call_id = call_id
        self.overall_sentiment = overall_sentiment
        self.customer_satisfaction = customer_satisfaction
        self.agent_performance = agent_performance
        self.summary = summary or ""
        self.topics = topics or []
        self.action_items = action_items or []
        self.key_indicators = key_indicators or []
        self.call_matrices = call_matrices or {}
        self.customer_name = customer_name
        self.hangup_reason = hangup_reason
        self.raw_model_response = raw_model_response
        self.model_id = model_id
        self.validation_results = validation_results
        self.validation_score = validation_score
        self.validation_percentage = validation_percentage
        self.skill_level = skill_level
        self.created_on = created_on

    def to_dict(self):
        base_dict = {
            "overallSentiment": float(self.overall_sentiment) if self.overall_sentiment is not None else None,
            "customerSatisfaction": float(self.customer_satisfaction) if self.customer_satisfaction is not None else None,
            "agentPerformance": float(self.agent_performance) if self.agent_performance is not None else None,
            "summary": self.summary or "",
            "topics": self.topics or [],
            "actionItems": self.action_items or [],
            "keyIndicators": self.key_indicators or [],
            "callMatrices": self.call_matrices or {},
            "customerName": self.customer_name,
            "hangupReason": self.hangup_reason,
        }
        
        if self.validation_results:
            base_dict["validation"] = self.validation_results
            base_dict["validationScore"] = float(self.validation_score) if self.validation_score else 0
            base_dict["validationPercentage"] = float(self.validation_percentage) if self.validation_percentage else 0
            base_dict["skillLevel"] = self.skill_level or "Novice"
        
        return base_dict


# ── CRUD ─────────────────────────────────────────────────────────────────────
_RECORD_COLUMNS = [
    "call_id", "agent_id", "account_id", "room_name", "from_phone", "to_phone",
    "call_source", "status", "language", "started_at", "ended_at",
    "duration_seconds", "transcript_s3_key", "recording_s3_key", "created_on",
    "updated_at",
]

_ANALYTICS_COLUMNS = [
    "call_id", "overall_sentiment", "customer_satisfaction", "agent_performance",
    "summary", "topics", "action_items", "key_indicators", "call_matrices",
    "customer_name", "hangup_reason", "raw_model_response", "model_id",
    "created_on", "updated_at",
]

_ANALYTICS_SELECT = [
    "call_id", "overall_sentiment", "customer_satisfaction", "agent_performance",
    "summary", "topics", "action_items", "key_indicators", "call_matrices",
    "customer_name", "hangup_reason", "raw_model_response", "model_id", "created_on",
]


def _now():
    return datetime.now(timezone.utc)


def get_record(call_id):
    """Return the latest CallRecord for call_id, or None"""
    try:
        rows = _get_client().query(
            f"SELECT {', '.join(CallRecord._FIELDS)} FROM {RECORDS_TABLE} FINAL "
            f"WHERE call_id = %(cid)s LIMIT 1",
            parameters={"cid": call_id},
        ).result_rows
    except Exception as e:
        print(f"[PCA-CH] get_record failed for {call_id}: {e}")
        _reset_client()
        return None
    if not rows:
        return None
    return CallRecord(**dict(zip(CallRecord._FIELDS, rows[0])))


def upsert_record(record: CallRecord):
    """Insert (or replace) a call record"""
    row = [
        record.call_id,
        record.agent_id or "",
        record.account_id or "",
        record.room_name or "",
        record.from_phone or "",
        record.to_phone or "",
        record.call_source or "livekit",
        record.status or "answered",
        record.language or "",
        record.started_at,
        record.ended_at,
        int(record.duration_seconds or 0),
        record.transcript_s3_key or "",
        record.recording_s3_key or "",
        record.created_on or _now(),
        _now(),
    ]
    try:
        _get_client().insert(RECORDS_TABLE, [row], column_names=_RECORD_COLUMNS)
    except Exception as e:
        print(f"[PCA-CH] upsert_record failed for {record.call_id}: {e}")
        _reset_client()
        raise


def _row_to_analytics(row):
    d = dict(zip(_ANALYTICS_SELECT, row))
    
    # Parse raw_model_response which may contain validation data
    raw_response = json.loads(d["raw_model_response"]) if d["raw_model_response"] else {}
    
    # Extract validation data if present
    validation_data = raw_response.get("validation", {})
    validation_results = validation_data.get("results")
    validation_score = validation_data.get("score")
    validation_percentage = validation_data.get("percentage")
    skill_level = validation_data.get("skill_level")
    
    # Remove validation from raw_response to keep it clean for original sentiment analysis
    if "validation" in raw_response:
        raw_response_without_validation = {k: v for k, v in raw_response.items() if k != "validation"}
    else:
        raw_response_without_validation = raw_response
    
    return CallAnalytics(
        call_id=d["call_id"],
        overall_sentiment=d["overall_sentiment"],
        customer_satisfaction=d["customer_satisfaction"],
        agent_performance=d["agent_performance"],
        summary=d["summary"],
        topics=list(d["topics"]) if d["topics"] else [],
        action_items=list(d["action_items"]) if d["action_items"] else [],
        key_indicators=list(d["key_indicators"]) if d["key_indicators"] else [],
        call_matrices=json.loads(d["call_matrices"]) if d["call_matrices"] else {},
        customer_name=d["customer_name"],
        hangup_reason=d["hangup_reason"],
        raw_model_response=raw_response_without_validation if raw_response_without_validation else None,
        model_id=d["model_id"],
        validation_results=validation_results,
        validation_score=validation_score,
        validation_percentage=validation_percentage,
        skill_level=skill_level,
        created_on=d["created_on"],
    )


def get_analytics(call_id):
    """Return the latest CallAnalytics for call_id, or None"""
    try:
        rows = _get_client().query(
            f"SELECT {', '.join(_ANALYTICS_SELECT)} FROM {ANALYTICS_TABLE} FINAL "
            f"WHERE call_id = %(cid)s LIMIT 1",
            parameters={"cid": call_id},
        ).result_rows
    except Exception as e:
        print(f"[PCA-CH] get_analytics failed for {call_id}: {e}")
        _reset_client()
        return None
    if not rows:
        return None
    return _row_to_analytics(rows[0])


def get_analytics_map(call_ids):
    """Return {call_id: CallAnalytics} for the given ids"""
    if not call_ids:
        return {}
    try:
        rows = _get_client().query(
            f"SELECT {', '.join(_ANALYTICS_SELECT)} FROM {ANALYTICS_TABLE} FINAL "
            f"WHERE call_id IN %(ids)s",
            parameters={"ids": list(call_ids)},
        ).result_rows
    except Exception as e:
        print(f"[PCA-CH] get_analytics_map failed: {e}")
        _reset_client()
        return {}
    out = {}
    for r in rows:
        a = _row_to_analytics(r)
        out[a.call_id] = a
    return out


def upsert_analytics(a: CallAnalytics):
    """Insert (or replace) an analytics row"""
    # Combine raw_model_response with validation in a single JSON
    combined_response = {}
    if a.raw_model_response:
        combined_response = a.raw_model_response.copy() if isinstance(a.raw_model_response, dict) else {}
    
    # Add validation data to the combined response
    if a.validation_results:
        combined_response["validation"] = {
            "results": a.validation_results,
            "score": a.validation_score,
            "percentage": a.validation_percentage,
            "skill_level": a.skill_level
        }
    
    row = [
        a.call_id,
        a.overall_sentiment,
        a.customer_satisfaction,
        a.agent_performance,
        a.summary or "",
        list(a.topics or []),
        list(a.action_items or []),
        list(a.key_indicators or []),
        json.dumps(a.call_matrices or {}),
        a.customer_name,
        a.hangup_reason,
        json.dumps(combined_response) if combined_response else "",
        a.model_id or "",
        a.created_on or _now(),
        _now(),
    ]
    try:
        _get_client().insert(ANALYTICS_TABLE, [row], column_names=_ANALYTICS_COLUMNS)
    except Exception as e:
        print(f"[PCA-CH] upsert_analytics failed for {a.call_id}: {e}")
        _reset_client()
        raise


def query_records(agent_id, start_date=None, end_date=None, page=None, limit=None):
    """Return (records, total_count) for an agent, newest first"""
    where = ["agent_id = %(agent_id)s"] if agent_id else ["1=1"]
    params = {"agent_id": agent_id} if agent_id else {}
    
    if start_date:
        where.append("created_on >= %(start)s")
        params["start"] = start_date
    if end_date:
        where.append("created_on <= %(end)s")
        params["end"] = end_date
    where_sql = " AND ".join(where)

    try:
        client = _get_client()
        total = client.query(
            f"SELECT count() FROM (SELECT call_id FROM {RECORDS_TABLE} FINAL WHERE {where_sql})",
            parameters=params,
        ).result_rows[0][0]

        sql = (f"SELECT {', '.join(CallRecord._FIELDS)} FROM {RECORDS_TABLE} FINAL "
               f"WHERE {where_sql} ORDER BY created_on DESC")
        if limit is not None:
            sql += " LIMIT %(limit)s OFFSET %(offset)s"
            params["limit"] = int(limit)
            params["offset"] = int(page or 0) * int(limit)
        rows = client.query(sql, parameters=params).result_rows
    except Exception as e:
        print(f"[PCA-CH] query_records failed: {e}")
        _reset_client()
        return [], 0

    records = [CallRecord(**dict(zip(CallRecord._FIELDS, r))) for r in rows]
    return records, total
