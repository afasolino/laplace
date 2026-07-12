from __future__ import annotations

import csv
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    page: int | None
    section: str | None = None
    chunk_id: str


class Value(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_wording: str
    normalized_value: float | str | None
    unit: str | None
    confidence: float = Field(ge=0, le=1)
    provenance: Provenance

    @model_validator(mode="after")
    def numeric_requires_unit(self) -> "Value":
        if isinstance(self.normalized_value, float) and not self.unit:
            raise ValueError("Numeric values require a unit")
        return self


class ScientificRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Value | None = None
    authors: list[Value] = []
    year: Value | None = None
    research_question: Value | None = None
    contribution: Value | None = None
    methodology: list[Value] = []
    configurations: list[Value] = []
    metrics: list[Value] = []
    baselines: list[Value] = []
    limitations: list[Value] = []
    future_work: list[Value] = []


METRIC = re.compile(
    r"(?P<wording>(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mW|W|mJ|J|ns|us|ms|s|MHz|GHz|TOPS|%))", re.I
)


def extract_metrics(text: str, provenance: Provenance) -> ScientificRecord:
    values = [
        Value(
            source_wording=m.group("wording"),
            normalized_value=float(m.group("value")),
            unit=m.group("unit"),
            confidence=1.0,
            provenance=provenance,
        )
        for m in METRIC.finditer(text)
    ]
    return ScientificRecord(metrics=values)


def write_record(record: ScientificRecord, json_path: Path, csv_path: Path) -> None:
    json_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "source_wording",
                "normalized_value",
                "unit",
                "filename",
                "page",
                "chunk_id",
            ],
        )
        writer.writeheader()
        for metric in record.metrics:
            writer.writerow(
                {
                    **metric.model_dump(exclude={"provenance", "confidence"}),
                    **metric.provenance.model_dump(exclude={"section"}),
                }
            )
