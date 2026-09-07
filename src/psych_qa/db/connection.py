"""Database connection management."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Cached SQLAlchemy engine."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Cached session factory."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Session:
    """Get a new SQLAlchemy session."""
    return get_session_factory()()
