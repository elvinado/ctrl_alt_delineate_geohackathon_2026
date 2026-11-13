"""SQLAlchemy models for project-scoped geoscience resources.

Dataset is the common polymorphic parent. Only project ownership and file-level
attributes live on Dataset. Domain-specific associations live on the concrete
subtype tables: seismic -> survey; well_log/checkshot/deviation -> well.
"""
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    surveys: Mapped[list["Survey"]] = relationship(back_populates="project", cascade="all, delete-orphan", passive_deletes=True)
    wells: Mapped[list["Well"]] = relationship(back_populates="project", cascade="all, delete-orphan", passive_deletes=True)
    datasets: Mapped[list["Dataset"]] = relationship(back_populates="project", cascade="all, delete-orphan", passive_deletes=True)


class Survey(Base):
    __tablename__ = "surveys"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    crs: Mapped[str | None] = mapped_column(String(100))
    metadata_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    project: Mapped[Project] = relationship(back_populates="surveys")
    seismic_datasets: Mapped[list["SeismicDataset"]] = relationship(back_populates="survey", cascade="all, delete-orphan", passive_deletes=True)


class Well(Base):
    __tablename__ = "wells"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    uwi: Mapped[str | None] = mapped_column(String(100), index=True)
    x: Mapped[float | None]
    y: Mapped[float | None]
    rt: Mapped[float | None]
    td: Mapped[float | None]
    crs: Mapped[str | None] = mapped_column(String(100))
    metadata_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    project: Mapped[Project] = relationship(back_populates="wells")
    well_log_datasets: Mapped[list["WellLogDataset"]] = relationship(back_populates="well", cascade="all, delete-orphan", passive_deletes=True)
    checkshot_datasets: Mapped[list["CheckshotDataset"]] = relationship(back_populates="well", cascade="all, delete-orphan", passive_deletes=True)
    deviation_datasets: Mapped[list["DeviationDataset"]] = relationship(back_populates="well", cascade="all, delete-orphan", passive_deletes=True)


class Dataset(Base):
    """Common dataset identity and source-file metadata; no survey_id/well_id here."""
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(30), nullable=False)
    crs: Mapped[str | None] = mapped_column(String(100))
    storage_root: Mapped[str] = mapped_column(String(1000), nullable=False)
    storage_mode: Mapped[str] = mapped_column(String(20), default="managed", nullable=False)
    source_path: Mapped[str | None] = mapped_column(String(2000))
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="ready", nullable=False, index=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    validation_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    project: Mapped[Project] = relationship(back_populates="datasets")
    __mapper_args__ = {"polymorphic_on": kind, "polymorphic_identity": "dataset"}


class TableDataset(Dataset):
    """Tabular source files such as Excel well lists; stored in the datasets table."""
    __mapper_args__ = {"polymorphic_identity": "table"}


class SeismicDataset(Dataset):
    __tablename__ = "seismic_datasets"
    id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True)
    survey_id: Mapped[str | None] = mapped_column(ForeignKey("surveys.id", ondelete="SET NULL"), nullable=True, index=True)
    survey: Mapped[Survey | None] = relationship(back_populates="seismic_datasets")
    __mapper_args__ = {"polymorphic_identity": "seismic"}


class WellLogDataset(Dataset):
    __tablename__ = "well_log_datasets"
    id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True)
    well_id: Mapped[str | None] = mapped_column(ForeignKey("wells.id", ondelete="SET NULL"), nullable=True, index=True)
    well: Mapped[Well | None] = relationship(back_populates="well_log_datasets")
    curves: Mapped[list["Curve"]] = relationship(back_populates="dataset", cascade="all, delete-orphan", order_by="Curve.position", passive_deletes=True)
    __mapper_args__ = {"polymorphic_identity": "well_log"}


class CheckshotDataset(Dataset):
    __tablename__ = "checkshot_datasets"
    id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True)
    well_id: Mapped[str | None] = mapped_column(ForeignKey("wells.id", ondelete="SET NULL"), nullable=True, index=True)
    well: Mapped[Well | None] = relationship(back_populates="checkshot_datasets")
    __mapper_args__ = {"polymorphic_identity": "checkshot"}


class DeviationDataset(Dataset):
    __tablename__ = "deviation_datasets"
    id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True)
    well_id: Mapped[str | None] = mapped_column(ForeignKey("wells.id", ondelete="SET NULL"), nullable=True, index=True)
    well: Mapped[Well | None] = relationship(back_populates="deviation_datasets")
    __mapper_args__ = {"polymorphic_identity": "deviation"}


class Curve(Base):
    """Independently addressable curve definition owned by one LAS dataset."""
    __tablename__ = "curves"
    __table_args__ = (UniqueConstraint("dataset_id", "position", name="uq_curve_dataset_position"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("well_log_datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    mnemonic: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    unit: Mapped[str | None] = mapped_column(String(50))
    description: Mapped[str | None] = mapped_column(Text)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    is_index: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    dataset: Mapped[WellLogDataset] = relationship(back_populates="curves")


class AppSetting(Base):
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(100))


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    operation: Mapped[str] = mapped_column(String(80), nullable=False, default="gpu_test")
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="submitted", index=True)
    slurm_job_id: Mapped[str | None] = mapped_column(String(40), index=True)
    job_dir: Mapped[str] = mapped_column(String(1000), nullable=False)
    log_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    result_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ProcessingJobInput(Base):
    """A dataset selected as an input to a processing job."""
    __tablename__ = "processing_job_inputs"
    __table_args__ = (UniqueConstraint("processing_job_id", "dataset_id", name="uq_processing_job_dataset"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    processing_job_id: Mapped[str] = mapped_column(ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False, index=True)
