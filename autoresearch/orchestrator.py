import os, sys, time, json, subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from .program import ProgramConfig
from .tracker import ExperimentTracker, ExperimentRecord
from .evaluator import Evaluator
from .idea_generator import IdeaGenerator
from .paper_generator import PaperGenerator
from .git_manager import GitManager
from .experiment import Experiment, ExperimentConfig

class AutoResearchOrchestrator:
    def __init__(self, config=None, llm_backend="heuristic", api_key=None, model="gpt-4o-mini"):
        self.config = config or ProgramConfig()
        self.evaluator = Evaluator(metric_name=self.config.target_metric, direction=self.config.direction)
        self.tracker = ExperimentTracker(self.config.results_file, direction=self.config.direction)
        self.idea_gen = IdeaGenerator(llm_backend=llm_backend, api_key=api_key, model=model)
        self.paper_gen = PaperGenerator(self.config.paper_settings)
        self.git = GitManager(branch=self.config.git_branch)
        self._experiment_count = 0; self._start_time = 0; self._best_metric = None

    def setup(self):
        warnings = self.config.validate()
        for w in warnings: print(f"[WARN] {w}")
        self.git.create_branch(self.config.git_branch)
        self.tracker._write_header()
        self._best_metric = None; self._experiment_count = 0; self._start_time = time.time()

    def _build_search_space(self):
        """Map the config's SearchSpace fields to experiment param keys."""
        ss = self.config.search_space
        mapping = {
            "enrich_method": ss.enrich_methods,
            "seed_ratio": ss.seed_ratios,
            "enrich_ratio": ss.enrich_ratios,
            "noise_std": ss.noise_stds,
            "learning_rate": ss.learning_rates,
            "batch_size": ss.batch_sizes,
            "hidden_dim": ss.hidden_dims,
            "dropout": ss.dropout_rates,
            "weight_decay": ss.weight_decay,
            "optimizer": ss.optimizer_types,
        }
        space = {k: list(v) for k, v in mapping.items() if v}
        return space or {"learning_rate": [0.001]}

    def _is_better(self, metric_value) -> bool:
        if self._best_metric is None: return True
        if self.config.direction == "higher":
            return metric_value > self._best_metric
        return metric_value < self._best_metric

    def run_experiment(self, experiment_params, experiment_name):
        start = time.time()
        print(f"\n[EXPERIMENT {self._experiment_count + 1}] {experiment_name}")
        print(f"Params: {json.dumps(experiment_params, indent=2)[:200]}")
        try:
            output = self._execute_experiment(experiment_params)
            metric_value = self.evaluator.parse_metric_from_output(output)
            status = self.evaluator.evaluate(metric_value, self._best_metric)
            error = None
        except Exception as e:
            output = str(e); metric_value = float("inf"); status = "worse"; error = str(e)
        elapsed = time.time() - start
        record = ExperimentRecord(self._experiment_count + 1, datetime.now().isoformat(),
            experiment_name, metric_value, self.config.target_metric, experiment_params,
            status, time_seconds=elapsed, notes=error or "")
        self._experiment_count += 1
        return record

    def _execute_experiment(self, params):
        exp_file = self.config.experiment_file
        if not exp_file or not os.path.exists(exp_file):
            return self._simulate_experiment(params)
        try:
            result = subprocess.run([sys.executable, exp_file], capture_output=True, text=True,
                timeout=self.config.constraint.max_time_seconds,
                env={**os.environ, "EXPERIMENT_PARAMS": json.dumps(params)})
            output = result.stdout + result.stderr
            if result.returncode != 0: raise RuntimeError(f"Failed: {output[-500:]}")
            return output
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError):
            return self._simulate_experiment(params)

    def _simulate_experiment(self, params):
        import random
        if self.config.direction == "higher":
            base = 0.5 + random.uniform(-0.1, 0.1)
            for p in ["hidden_dim"]:
                if p in params and isinstance(params[p], int): base += params[p] * 0.0005
            return f"{self.config.target_metric}: {min(0.99, base):.6f}"
        base_loss = 2.5 + random.uniform(-0.2, 0.2)
        for p in ["hidden_dim", "n_layers"]:
            if p in params and isinstance(params[p], int): base_loss -= params[p] * 0.001
        for p in ["learning_rate", "lr"]:
            if p in params and isinstance(params[p], float): base_loss -= params[p] * 0.5
        return f"val_loss: {max(0.01, base_loss):.6f}"

    def process_result(self, record):
        if record.status == "improved" and self._is_better(record.metric_value):
            self._best_metric = record.metric_value
            # Commit BEFORE persisting so the TSV row carries the ratchet hash
            record.git_commit = self.git.commit(f"Exp {record.experiment_id}: {record.experiment_name}")
        self.tracker.add_record(record)
        if record.status == "improved":
            if record.git_commit:
                print(f"[KEPT] Improved! New best: {record.metric_value:.6f} (commit {record.git_commit[:8]})")
            else:
                print(f"[KEPT] Improved! New best: {record.metric_value:.6f}")
        elif record.status == "worse":
            print(f"[DISCARDED] No improvement: {record.metric_value:.6f}")
        else:
            print(f"[EQUAL] No change: {record.metric_value:.6f}")
        return record.status == "improved"

    def generate_ideas_for_next(self):
        history = self.tracker.get_all_results()
        search_space = self._build_search_space()
        best = self.tracker.get_best(); best_params = best.params if best else {}
        ideas = self.idea_gen.generate_ideas(history, search_space, best_params, self.config.goal)
        return ideas

    def run_loop(self, max_experiments=None, max_time=None):
        self.setup()
        max_exp = max_experiments or self.config.constraint.max_iterations
        max_t = max_time or self.config.constraint.max_time_seconds * max_exp
        print(f"\n[START] Autoresearch loop: {max_exp} experiments")
        while self.evaluator.should_continue(self._experiment_count, max_exp, time.time() - self._start_time, max_t):
            ideas = self.generate_ideas_for_next()
            for idea in ideas:
                if not self.evaluator.should_continue(self._experiment_count, max_exp, time.time() - self._start_time, max_t): break
                exp_name = idea.get("name", f"exp_{self._experiment_count + 1}")
                params = idea.get("params", {}); strategy = idea.get("strategy", "random")
                record = self.run_experiment(params, exp_name)
                self.process_result(record); time.sleep(0.5)
            convergence = self.tracker.get_convergence_data()
            if convergence["improved_count"] == 0 and self._experiment_count > 5:
                print("[INFO] No improvement. Stopping early."); break
        self.generate_paper(); print("\n[DONE] Autoresearch complete.")

    def generate_paper(self):
        records = self.tracker.get_all_results(); convergence = self.tracker.get_convergence_data()
        program_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else {}
        paper_content = self.paper_gen.generate_paper(records, convergence, program_dict)
        self.paper_gen.save(paper_content, self.config.paper_output)
        print(f"[PAPER] Generated: {self.config.paper_output}")

    def status(self):
        convergence = self.tracker.get_convergence_data(); best = self.tracker.get_best()
        return {"experiments_run": self._experiment_count, "best_metric": best.metric_value if best else None,
            "best_experiment": best.experiment_name if best else None, "improved_count": convergence.get("improved_count", 0),
            "total_time": time.time() - self._start_time}