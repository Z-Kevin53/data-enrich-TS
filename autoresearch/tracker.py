import csv, json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import Optional, List, Dict

@dataclass
class ExperimentRecord:
    experiment_id: int
    timestamp: str
    experiment_name: str
    metric_value: float
    metric_name: str
    params: Dict
    status: str
    git_commit: Optional[str] = None
    git_diff: Optional[str] = None
    notes: str = ""
    time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentRecord":
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in fields(cls)]})

class ExperimentTracker:
    HEADER = ["experiment_id", "timestamp", "experiment_name", "metric_name", "metric_value", "status", "params", "git_commit", "time_seconds", "notes"]

    def __init__(self, results_path="results/results.tsv", direction="lower"):
        self.results_path = Path(results_path)
        self.direction = direction
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[ExperimentRecord] = []
        self._load()

    def _load(self):
        if not self.results_path.exists():
            self._write_header(); return
        with open(self.results_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    record = ExperimentRecord(
                        experiment_id=int(row["experiment_id"]), timestamp=row["timestamp"],
                        experiment_name=row["experiment_name"], metric_name=row["metric_name"],
                        metric_value=float(row["metric_value"]), status=row["status"],
                        params=json.loads(row["params"]), git_commit=row.get("git_commit", ""),
                        time_seconds=float(row.get("time_seconds", 0)), notes=row.get("notes", ""))
                    self._records.append(record)
                except (ValueError, KeyError): continue

    def _write_header(self):
        with open(self.results_path, "w", newline="") as f:
            writer = csv.writer(f); writer.writerow(self.HEADER)

    def _write_row(self, record: ExperimentRecord):
        with open(self.results_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([record.experiment_id, record.timestamp, record.experiment_name,
                record.metric_name, record.metric_value, record.status, json.dumps(record.params),
                record.git_commit or "", record.time_seconds, record.notes])

    def add_record(self, record: ExperimentRecord) -> ExperimentRecord:
        record.experiment_id = len(self._records) + 1
        self._records.append(record); self._write_row(record); return record

    def get_best(self) -> Optional[ExperimentRecord]:
        if not self._records: return None
        improved = [r for r in self._records if r.status == "improved"]
        if not improved: return self._records[0]
        if self.direction == "lower":
            return min(improved, key=lambda r: r.metric_value)
        return max(improved, key=lambda r: r.metric_value)

    def get_all_results(self) -> List[Dict]:
        return [r.to_dict() for r in self._records]

    def get_convergence_data(self) -> Dict:
        return {"total_experiments": len(self._records),
            "improved_count": sum(1 for r in self._records if r.status == "improved"),
            "worse_count": sum(1 for r in self._records if r.status == "worse"),
            "best_metric": self.get_best().metric_value if self._records else None,
            "metrics_over_time": [r.metric_value for r in self._records],
            "timestamps": [r.timestamp for r in self._records]}

    def clear(self):
        self._records.clear(); self._write_header()