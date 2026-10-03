import math, re, json
from typing import Dict, Any, Optional

class Evaluator:
    def __init__(self, metric_name="val_loss", direction="lower"):
        self.metric_name = metric_name; self.direction = direction

    def parse_metric_from_output(self, output: str) -> float:
        lines = output.strip().split("\n")
        for line in reversed(lines):
            if self.metric_name in line or "loss" in line.lower():
                try:
                    numbers = re.findall(r"[-+]?\d*\.?\d+", line)
                    if numbers: return float(numbers[-1])
                except (ValueError, IndexError): continue
        try:
            data = json.loads(output.strip().split("\n")[-1])
            if self.metric_name in data: return float(data[self.metric_name])
        except (json.JSONDecodeError, ValueError): pass
        raise ValueError(f"Could not parse {self.metric_name} from output")

    def evaluate(self, metric_value: float, previous_best: Optional[float] = None) -> str:
        if previous_best is None: return "improved"
        if self.direction == "lower":
            if metric_value < previous_best * 0.999: return "improved"
            elif abs(metric_value - previous_best) < 1e-10: return "equal"
            else: return "worse"
        else:
            if metric_value > previous_best * 1.001: return "improved"
            elif abs(metric_value - previous_best) < 1e-10: return "equal"
            else: return "worse"

    def should_continue(self, experiment_count, max_iterations, time_elapsed, max_time) -> bool:
        return experiment_count < max_iterations and time_elapsed < max_time

    def compute_statistics(self, metrics):
        if not metrics: return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
        n = len(metrics); mean = sum(metrics) / n
        variance = sum((m - mean) ** 2 for m in metrics) / n; std = math.sqrt(variance)
        return {"mean": mean, "std": std, "min": min(metrics), "max": max(metrics), "count": n,
            "improvement_pct": ((metrics[0] - min(metrics)) / abs(metrics[0]) * 100) if metrics[0] != 0 else 0}
