"""Database engine, session management, and initialization."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from accubet.config import PROJECT_ROOT, get_config
from accubet.storage.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _resolve_url(url: str) -> str:
    """Make relative SQLite paths absolute under the project root, and ensure the dir."""
    parsed = make_url(url)
    if parsed.drivername.startswith("sqlite") and parsed.database:
        db_path = Path(parsed.database)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        parsed = parsed.set(database=str(db_path))
    return parsed.render_as_string(hide_password=False)


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        url = _resolve_url(get_config().secrets.database_url)
        _engine = create_engine(url, future=True)
        if _engine.dialect.name == "sqlite":
            # Enforce foreign keys on SQLite (off by default).
            @event.listens_for(_engine, "connect")
            def _fk_pragma(dbapi_conn, _):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
