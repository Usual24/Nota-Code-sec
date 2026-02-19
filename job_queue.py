from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from config import settings


@dataclass(slots=True)
class JobStatus:
    job_id: str
    user_id: int
    thread_id: int
    command: str
    state: str
    created_at: float
    updated_at: float
    detail: str = ""
    result: str = ""


class JobQueueService:
    """Local async queue with optional RQ-compatibility extension point.

    For production, this can be replaced by an RQ worker implementation while
    preserving the same persistence schema and status API.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nota-job")
        self._db_lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    thread_id INTEGER NOT NULL,
                    command TEXT NOT NULL,
                    state TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    result TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def enqueue(
        self,
        *,
        user_id: int,
        thread_id: int,
        command: str,
        handler: Callable[[], str],
    ) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._db_lock:
            conn = sqlite3.connect(settings.sqlite_path)
            try:
                conn.execute(
                    """
                    INSERT INTO jobs(job_id, user_id, thread_id, command, state, detail, result, created_at, updated_at)
                    VALUES(?, ?, ?, ?, 'queued', '', '', ?, ?)
                    """,
                    (job_id, user_id, thread_id, command, now, now),
                )
                conn.commit()
            finally:
                conn.close()

        self._executor.submit(self._run_job, job_id, handler)
        return job_id

    def _run_job(self, job_id: str, handler: Callable[[], str]) -> None:
        self._update(job_id, state="running", detail="?묒뾽 ?ㅽ뻾 以?)
        try:
            result = handler()
            self._update(job_id, state="completed", detail="?묒뾽 ?꾨즺", result=result)
        except Exception as exc:  # noqa: BLE001
            self._update(job_id, state="failed", detail=f"?묒뾽 ?ㅽ뙣: {type(exc).__name__}", result=self._redact_paths(str(exc)))

    @staticmethod
    def _redact_paths(message: str) -> str:
        if not message:
            return message
        # Redact absolute filesystem paths to avoid path disclosure/injection.
        message = re.sub(r\"[A-Za-z]:\\\\[^\r\n\\\"']+\", \"<redacted-path>\", message)
        message = re.sub(r\"/[^\\s\\\"'\r\n]+\", \"<redacted-path>\", message)
        return message
    def _update(self, job_id: str, *, state: str, detail: str, result: str | None = None) -> None:
        now = time.time()
        with self._db_lock:
            conn = sqlite3.connect(settings.sqlite_path)
            try:
                if result is None:
                    conn.execute(
                        "UPDATE jobs SET state=?, detail=?, updated_at=? WHERE job_id=?",
                        (state, detail, now, job_id),
                    )
                else:
                    conn.execute(
                        "UPDATE jobs SET state=?, detail=?, result=?, updated_at=? WHERE job_id=?",
                        (state, detail, result, now, job_id),
                    )
                conn.commit()
            finally:
                conn.close()

    def get_status(self, job_id: str, user_id: int, thread_id: int) -> JobStatus | None:
        conn = sqlite3.connect(settings.sqlite_path)
        try:
            row = conn.execute(
                """
                SELECT job_id, user_id, thread_id, command, state, created_at, updated_at, detail, result
                FROM jobs
                WHERE job_id=? AND user_id=? AND thread_id=?
                """,
                (job_id, user_id, thread_id),
            ).fetchone()
            if not row:
                return None
            return JobStatus(
                job_id=row[0],
                user_id=row[1],
                thread_id=row[2],
                command=row[3],
                state=row[4],
                created_at=row[5],
                updated_at=row[6],
                detail=row[7],
                result=row[8],
            )
        finally:
            conn.close()
