import json, os, random, copy
from typing import List, Dict, Optional

class IdeaGenerator:
    def __init__(self, llm_backend="heuristic", api_key=None, model="gpt-4o-mini", base_url=None):
        self.llm_backend = llm_backend; self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model; self.base_url = base_url

    def generate_ideas(self, experiment_history, search_space, best_params, program_instructions=""):
        return self._generate_heuristic(experiment_history, search_space, best_params, program_instructions)

    def _generate_heuristic(self, history, search_space, best_params, instructions):
        ideas = []
        if not history:
            ideas.append({"name": "baseline_experiment", "params": self._default_params(search_space), "strategy": "baseline", "rationale": "Establish baseline"})
            return ideas
        best = max(history, key=lambda x: x.get("metric_value", 0))
        last_exp = history[-1]
        improvement = best.get("metric_value", 0) - last_exp.get("metric_value", 0)
        if improvement > 0.01:
            ideas.append({"name": "continue_optimization", "params": self._tune_neighborhood(best_params, search_space), "strategy": "guided", "rationale": f"Continue from best (metric: {best.get('metric_value')})"})
        else:
            ideas.append({"name": "architecture_exploration", "params": self._random_params(search_space), "strategy": "architecture", "rationale": "Plateau detected"})
            ideas.append({"name": "regularization_experiment", "params": self._regularization_params(search_space, best_params), "strategy": "regularization", "rationale": "Try regularization"})
        if len(ideas) < 3:
            ideas.append({"name": "hyperparameter_search", "params": self._random_params(search_space), "strategy": "hyperparameter", "rationale": "Broad exploration"})
        return ideas[:3]

    def _default_params(self, search_space):
        params = {}
        for key, values in search_space.items():
            if isinstance(values, list) and len(values) > 0: params[key] = values[0]
            else: params[key] = 32 if key in ["batch_size", "n_layers", "hidden_dim"] else 0.001
        return params

    def _random_params(self, search_space):
        params = {}
        for key, values in search_space.items():
            if isinstance(values, list) and len(values) > 0:
                if all(isinstance(v, (int, float)) for v in values): params[key] = random.choice(values)
                elif all(isinstance(v, str) for v in values): params[key] = random.choice(values)
                else: params[key] = values[0]
            else: params[key] = 32 if key in ["batch_size", "n_layers", "hidden_dim"] else 0.001
        return params

    def _tune_neighborhood(self, best_params, search_space):
        params = copy.deepcopy(best_params)
        for key in params:
            if isinstance(params[key], float): params[key] *= random.uniform(0.8, 1.2)
            elif isinstance(params[key], int): params[key] += random.choice([-1, 0, 1])
        return params

    def _regularization_params(self, search_space, best_params):
        params = copy.deepcopy(best_params)
        params["dropout"] = round(random.uniform(0.2, 0.5), 2)
        params["weight_decay"] = round(random.uniform(0.001, 0.01), 4)
        return params
