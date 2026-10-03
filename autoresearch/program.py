from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import yaml

@dataclass
class ExperimentConstraint:
    max_time_seconds: int = 600
    max_iterations: int = 50
    metric_direction: str = "higher"
    metric_name: str = "val_acc"
    metric_threshold: Optional[float] = None

@dataclass
class SearchSpace:
    # Model hyperparameters
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.005, 0.01])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])
    optimizer_types: List[str] = field(default_factory=lambda: ["adam", "adamw", "sgd"])
    weight_decay: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.01])
    # Data enrichment strategy (the research dimensions of this project)
    enrich_methods: List[str] = field(default_factory=lambda: ["none", "synthetic", "oversample", "noise", "hybrid"])
    seed_ratios: List[float] = field(default_factory=lambda: [0.15, 0.25, 0.35, 0.5])
    enrich_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    noise_stds: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])

@dataclass
class ProgramConfig:
    project_name: str = "data_enrich"
    description: str = "Data enrichment strategy optimization on UCI Spambase"
    goal: str = "Maximize validation accuracy of a small MLP by enriching a seed subset of labeled spambase emails"
    target_metric: str = "val_acc"
    direction: str = "higher"
    constraint: ExperimentConstraint = field(default_factory=ExperimentConstraint)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    experiment_file: str = "experiments/enrich_experiment.py"
    results_file: str = "results/results.tsv"
    paper_output: str = "papers/generated_paper.md"
    git_branch: str = "autoresearch/experiment"
    research_phases: List[str] = field(default_factory=lambda: ["seed_setup", "enrichment_search", "hyperparameter", "regularization", "optimization"])
    paper_settings: dict = field(default_factory=lambda: {"title_prefix": "AutoResearch: Data Enrichment", "include_ablation": True, "include_analysis": True, "format": "markdown"})

    def to_dict(self):
        return asdict(self)

    def save(self, path="program_config.yaml"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, path="program_config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data.get("constraint") and isinstance(data["constraint"], dict):
            data["constraint"] = ExperimentConstraint(**data["constraint"])
        if data.get("search_space") and isinstance(data["search_space"], dict):
            data["search_space"] = SearchSpace(**data["search_space"])
        return cls(**data)

    def validate(self):
        warnings = []
        if self.constraint.max_time_seconds < 10: warnings.append("Time budget very short")
        if not self.search_space.enrich_methods: warnings.append("No enrichment methods defined")
        return warnings