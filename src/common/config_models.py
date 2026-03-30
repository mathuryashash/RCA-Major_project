from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


class LogSource(BaseModel):
    label: str
    path: str
    format: str = Field(pattern="^(plaintext|json|syslog)$")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"plaintext", "json", "syslog"}
        if v not in allowed:
            raise ValueError(f"format must be one of: {allowed}")
        return v


class LogWatchingConfig(BaseModel):
    enabled: bool = True
    poll_interval_seconds: int = Field(ge=1, le=60)
    max_buffer_size: int = Field(ge=100, le=100000)


class MetricSourcesConfig(BaseModel):
    prometheus_url: Optional[str] = None
    scrape_interval_seconds: Optional[int] = Field(default=60, ge=1, le=3600)


class DatabaseConfig(BaseModel):
    host: str
    port: int = Field(ge=1, le=65535)
    user: str
    password: str
    dbname: str


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    stream_name: str = "rca_metrics"
    consumer_group: str = "rca_pipeline"


class JWTConfig(BaseModel):
    secret_key: str = Field(min_length=32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(ge=1, le=1440)
    refresh_token_expire_days: int = Field(ge=1, le=30)


class AnomalyConfig(BaseModel):
    window_size: int = Field(ge=10, le=1000)
    threshold_percentile: float = Field(ge=90.0, le=99.99)
    alpha: float = Field(ge=0.0, le=1.0)


class CausalConfig(BaseModel):
    lags: int = Field(ge=1, le=20)
    fdr_alpha: float = Field(ge=0.0, le=0.5)
    correlation_window_minutes: Optional[int] = Field(default=2, ge=1, le=60)


class AppConfig(BaseModel):
    log_sources: List[LogSource]
    log_watching: Optional[LogWatchingConfig] = None
    metric_sources: Optional[MetricSourcesConfig] = None
    database: DatabaseConfig
    redis: Optional[RedisConfig] = None
    anomaly_detection: AnomalyConfig
    causal_inference: CausalConfig
    jwt: Optional[JWTConfig] = None

    class Config:
        extra = "allow"


def validate_config(config_dict: Dict[str, Any]) -> AppConfig:
    """Validate configuration dictionary against Pydantic models."""
    return AppConfig.model_validate(config_dict)
