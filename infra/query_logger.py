"""
查询日志模块
使用SQLite存储所有查询记录，支持统计分析
"""

import sqlite3
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List


class QueryLogger:
    """查询日志记录器"""

    def __init__(self, db_path: str = "./logs/query_logs.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                datetime_str TEXT NOT NULL,
                session_id TEXT,
                user_id TEXT,
                question TEXT NOT NULL,
                answer TEXT,
                success INTEGER DEFAULT 1,
                error_msg TEXT,
                processing_time REAL,
                tool_steps INTEGER DEFAULT 0,
                source TEXT DEFAULT 'api'
            );
            CREATE INDEX IF NOT EXISTS idx_timestamp ON query_logs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_session ON query_logs(session_id);
            CREATE INDEX IF NOT EXISTS idx_success ON query_logs(success);
        """)
        conn.commit()

    def log(self, question: str, answer: str, success: bool = True,
            session_id: str = "", user_id: str = "",
            error_msg: str = "", processing_time: float = 0,
            tool_steps: int = 0, source: str = "api"):
        """记录一条查询"""
        conn = self._get_conn()
        now = time.time()
        dt_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("""
            INSERT INTO query_logs (timestamp, datetime_str, session_id, user_id,
                question, answer, success, error_msg, processing_time, tool_steps, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, dt_str, session_id, user_id, question, answer,
              1 if success else 0, error_msg, processing_time, tool_steps, source))
        conn.commit()

    def get_logs(self, page: int = 1, page_size: int = 50,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None,
                 success_only: Optional[bool] = None) -> Dict[str, Any]:
        """分页查询日志"""
        conn = self._get_conn()
        conditions = []
        params = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if success_only is not None:
            conditions.append("success = ?")
            params.append(1 if success_only else 0)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        # 总数
        total = conn.execute(f"SELECT COUNT(*) FROM query_logs {where}", params).fetchone()[0]

        # 分页数据
        offset = (page - 1) * page_size
        rows = conn.execute(f"""
            SELECT * FROM query_logs {where}
            ORDER BY timestamp DESC LIMIT ? OFFSET ?
        """, params + [page_size, offset]).fetchall()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "logs": [dict(r) for r in rows]
        }

    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """获取统计数据"""
        conn = self._get_conn()
        cutoff = time.time() - days * 86400

        # 基础统计
        row = conn.execute("""
            SELECT
                COUNT(*) as total_queries,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count,
                AVG(processing_time) as avg_latency,
                MIN(processing_time) as min_latency,
                MAX(processing_time) as max_latency
            FROM query_logs WHERE timestamp >= ?
        """, (cutoff,)).fetchone()

        total = row["total_queries"] or 0
        success = row["success_count"] or 0
        errors = row["error_count"] or 0
        error_rate = (errors / total * 100) if total > 0 else 0

        # 每日统计
        daily = conn.execute("""
            SELECT
                date(datetime_str) as day,
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as errors,
                AVG(processing_time) as avg_latency
            FROM query_logs WHERE timestamp >= ?
            GROUP BY date(datetime_str)
            ORDER BY day
        """, (cutoff,)).fetchall()

        # 每小时分布
        hourly = conn.execute("""
            SELECT
                CAST(strftime('%H', datetime_str) AS INTEGER) as hour,
                COUNT(*) as count
            FROM query_logs WHERE timestamp >= ?
            GROUP BY hour ORDER BY hour
        """, (cutoff,)).fetchall()

        return {
            "period_days": days,
            "total_queries": total,
            "success_count": success,
            "error_count": errors,
            "error_rate": round(error_rate, 2),
            "avg_latency": round(row["avg_latency"] or 0, 2),
            "min_latency": round(row["min_latency"] or 0, 2),
            "max_latency": round(row["max_latency"] or 0, 2),
            "daily": [dict(r) for r in daily],
            "hourly": [dict(r) for r in hourly],
        }

    def get_hot_questions(self, days: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        """获取热门问题"""
        conn = self._get_conn()
        cutoff = time.time() - days * 86400
        rows = conn.execute("""
            SELECT
                question,
                COUNT(*) as query_count,
                AVG(processing_time) as avg_latency,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                MAX(timestamp) as last_query_time
            FROM query_logs
            WHERE timestamp >= ?
            GROUP BY question
            ORDER BY query_count DESC
            LIMIT ?
        """, (cutoff, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_error_logs(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """获取错误日志"""
        conn = self._get_conn()
        cutoff = time.time() - days * 86400
        rows = conn.execute("""
            SELECT * FROM query_logs
            WHERE success = 0 AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (cutoff, limit)).fetchall()
        return [dict(r) for r in rows]

    def clear_old_logs(self, days: int = 30):
        """清理旧日志"""
        conn = self._get_conn()
        cutoff = time.time() - days * 86400
        deleted = conn.execute("DELETE FROM query_logs WHERE timestamp < ?", (cutoff,)).rowcount
        conn.commit()
        return deleted
