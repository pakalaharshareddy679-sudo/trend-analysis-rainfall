from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RainfallSeriesInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    dates: list[str] = Field(min_length=2, max_length=100_000)
    values: list[float | None] = Field(min_length=2, max_length=100_000, allow_inf_nan=False)
    unit: Literal["mm", "cm", "in", "0.1mm"] = "mm"
    frequency: Literal["auto", "daily", "monthly", "annual"] = "auto"
    source_format: str = "json"

    @model_validator(mode="after")
    def dates_and_values_match(self) -> "RainfallSeriesInput":
        if len(self.dates) != len(self.values):
            raise ValueError("dates and values must have the same length")
        return self


class AnalysisOptions(BaseModel):
    # Defaults are UI starting values, not a claim about the report's settings.
    rainy_day_mm: float = Field(default=1.0, ge=0)
    wet_day_mm: float = Field(default=2.5, ge=0)
    max_missing_per_year_pct: float = Field(default=100.0, ge=0, le=100)
    base_period_start: int | None = None
    base_period_end: int | None = None
    spi_scales: list[int] = Field(default_factory=lambda: [1, 3, 6, 12, 24], max_length=5)
    drought_threshold: float = Field(default=-1.0, le=0)

    @model_validator(mode="after")
    def validate_options(self) -> "AnalysisOptions":
        if self.wet_day_mm < self.rainy_day_mm:
            raise ValueError("wet_day_mm must be greater than or equal to rainy_day_mm")
        if any(scale < 1 or scale > 24 for scale in self.spi_scales):
            raise ValueError("SPI scales must be between 1 and 24 months")
        if len(set(self.spi_scales)) != len(self.spi_scales):
            raise ValueError("SPI scales must be unique")
        if self.base_period_start and self.base_period_end and self.base_period_start > self.base_period_end:
            raise ValueError("base_period_start must not exceed base_period_end")
        return self


class AnalysisRequest(BaseModel):
    series: list[RainfallSeriesInput] = Field(min_length=1, max_length=20)
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"]
    status_url: str
    result_url: str
    message: str
