"""Relational store using SQLAlchemy (SQLite default, PostgreSQL ready).

Stores structured material data: compositions, conductivities, synthesis
conditions, etc. This is the structured knowledge base layer.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column, String, Float, Integer, Text, DateTime, JSON,
    create_engine, inspect, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Session

from rich.console import Console

console = Console()

# Project root (independent of CWD): F:/projects/matresearcher
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Base(DeclarativeBase):
    pass


class LiteratureRecord(Base):
    """ORM model for literature entries."""
    __tablename__ = "literature"
    id = Column(String, primary_key=True)
    doi = Column(String, index=True, nullable=True)
    title = Column(Text)
    authors = Column(JSON)  # list[str]
    year = Column(Integer, nullable=True)
    journal = Column(String, nullable=True)
    abstract = Column(Text, nullable=True)
    relevance_score = Column(Float, nullable=True)
    verification_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    extra = Column(JSON, nullable=True)  # flexible storage


class KnowledgeRecordORM(Base):
    """ORM model for extracted knowledge records."""
    __tablename__ = "knowledge_records"
    id = Column(String, primary_key=True)
    literature_id = Column(String, index=True)
    material_composition = Column(String, nullable=True)
    material_alias = Column(String, nullable=True)          # e.g. "LLZO"
    crystal_structure = Column(String, nullable=True)
    ionic_conductivity = Column(Float, nullable=True)       # S/cm
    ionic_cond_temp = Column(Float, nullable=True)           # K
    electrochemical_window = Column(Float, nullable=True)    # V
    synthesis_method = Column(Text, nullable=True)
    sintering_temperature = Column(Float, nullable=True)    # K
    test_temperature = Column(Float, nullable=True)          # K
    pressure = Column(Float, nullable=True)                 # MPa
    simulation_method = Column(String, nullable=True)
    key_findings = Column(Text, nullable=True)
    raw_quotes = Column(JSON, nullable=True)
    quality_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)


class GapRecordORM(Base):
    """ORM model for Research Gaps."""
    __tablename__ = "research_gaps"
    gap_id = Column(String, primary_key=True)
    description = Column(Text)
    supporting_literature = Column(JSON)     # list[dict]
    evidence_gap_or_conflict = Column(Text)
    novelty = Column(Text)
    operability = Column(Text)
    falsifiable_hypothesis = Column(Text)
    suggested_verification = Column(Text)
    novelty_score = Column(Float, default=0.0)
    operability_score = Column(Float, default=0.0)
    evidence_completeness = Column(Float, default=0.0)
    total_score = Column(Float, default=0.0)
    rank = Column(Integer, default=0)
    verification_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLogORM(Base):
    """Audit log for every agent operation."""
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    step = Column(String)
    agent = Column(String)
    action = Column(Text)
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    extra = Column(JSON, nullable=True)


class RelationalStore:
    """SQLAlchemy-based relational store."""

    def __init__(self, db_url: str | None = None):
        # Default: absolute path under the project root (never CWD-relative —
        # a relative default used to split data between ./data and src/data).
        self.db_url = db_url or os.getenv(
            "DB_URL",
            f"sqlite:///{(PROJECT_ROOT / 'data' / 'matresearcher.db').as_posix()}",
        )

        # For SQLite, ensure the directory exists
        if self.db_url.startswith("sqlite:///"):
            db_path = self.db_url[len("sqlite:///"):]
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self.engine = create_engine(self.db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self._migrate_schema()
        console.print(f"[green]Database initialized: {self.db_url}[/green]")

    def _migrate_schema(self):
        """Auto-migrate: add missing columns to existing tables.

        SQLAlchemy's create_all only creates new tables; it never alters
        existing ones.  When we add a column to an ORM model after the table
        already exists, we need to ALTER TABLE ourselves.  This method
        compares the ORM model columns against the live SQLite schema and
        fills in any gaps.
        """
        inspector = inspect(self.engine)

        # Map: {tablename: {column_name → Column}}
        orm_models = {
            cls.__tablename__: {
                c.name: c for c in cls.__table__.columns
            }
            for cls in (LiteratureRecord, KnowledgeRecordORM, GapRecordORM, AuditLogORM)
        }

        for table_name, orm_cols in orm_models.items():
            if table_name not in inspector.get_table_names():
                continue  # table doesn't exist yet — create_all will handle it
            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
            for col_name, col in orm_cols.items():
                if col_name in existing_cols:
                    continue
                # Determine SQLite type from the Column
                col_type = str(col.type).upper()
                nullable = "" if col.nullable else " NOT NULL"
                default_clause = ""
                if col.default and col.default.arg is not None:
                    default_clause = f" DEFAULT {col.default.arg!r}"
                sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}{nullable}{default_clause}"
                with self.engine.begin() as conn:
                    conn.execute(text(sql))
                console.print(f"  [yellow]Migrated: added column {table_name}.{col_name}[/yellow]")

    def add_literature(self, lit_data: dict):
        with Session(self.engine) as session:
            record = LiteratureRecord(**lit_data)
            session.merge(record)
            session.commit()

    def add_literature_batch(self, lit_data_list: list[dict]):
        """Batch-insert literature records in a single session."""
        with Session(self.engine) as session:
            for lit_data in lit_data_list:
                session.merge(LiteratureRecord(**lit_data))
            session.commit()

    def add_knowledge(self, kr_data: dict):
        with Session(self.engine) as session:
            record = KnowledgeRecordORM(**kr_data)
            session.merge(record)
            session.commit()

    def add_knowledge_batch(self, kr_data_list: list[dict]):
        """Batch-insert knowledge records in a single session.

        Uses session.merge for upsert semantics, same as add_knowledge,
        but commits once for the entire batch.
        """
        with Session(self.engine) as session:
            for kr_data in kr_data_list:
                session.merge(KnowledgeRecordORM(**kr_data))
            session.commit()

    def add_gap(self, gap_data: dict):
        with Session(self.engine) as session:
            record = GapRecordORM(**gap_data)
            session.merge(record)
            session.commit()

    def add_audit(self, step: str, agent: str, action: str, **kwargs):
        with Session(self.engine) as session:
            entry = AuditLogORM(
                step=step, agent=agent, action=action,
                input_summary=kwargs.get("input_summary"),
                output_summary=kwargs.get("output_summary"),
                extra=kwargs.get("extra"),
            )
            session.add(entry)
            session.commit()

    def query_knowledge_by_material(self, material: str) -> list[dict]:
        """Query all knowledge records for a specific material."""
        with Session(self.engine) as session:
            stmt = select(KnowledgeRecordORM).where(
                KnowledgeRecordORM.material_composition == material
            )
            results = session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id, "literature_id": r.literature_id,
                    "material": r.material_composition,
                    "conductivity": r.ionic_conductivity,
                    "conductivity_temp": r.ionic_cond_temp,
                    "sintering_temp": r.sintering_temperature,
                    "synthesis_method": r.synthesis_method,
                    "quality_status": r.quality_status,
                }
                for r in results
            ]

    def get_all_materials(self) -> list[str]:
        """Get all unique material compositions."""
        with Session(self.engine) as session:
            stmt = select(KnowledgeRecordORM.material_composition).distinct()
            return [r for r in session.execute(stmt).scalars() if r]
