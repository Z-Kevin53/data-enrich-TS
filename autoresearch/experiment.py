import random, copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class ExperimentConfig:
    name: str = "experiment"
    description: str = ""
    params: Dict = field(default_factory=dict)
    strategy: str = "random"
    parent_experiment_id: Optional[int] = None

class MutationStrategy:
    @staticmethod
    def random_mutation(params, search_space):
        mutated = dict(params)
        for key, space in search_space.items():
            if isinstance(space, list) and len(space) > 0 and random.random() < 0.5:
                mutated[key] = random.choice(space)
            elif key not in mutated and isinstance(space, list) and len(space) > 0:
                mutated[key] = random.choice(space)
        return mutated

    @staticmethod
    def guided_mutation(params, search_space, best_params, metric_history):
        mutated = copy.deepcopy(best_params)
        for key in mutated:
            if isinstance(mutated[key], float): mutated[key] *= random.uniform(0.8, 1.2)
            elif isinstance(mutated[key], int): mutated[key] += random.choice([-1, 0, 1])
            elif isinstance(mutated[key], str) and key in search_space and search_space[key]:
                mutated[key] = random.choice(search_space[key])
        return mutated

    @classmethod
    def get_strategy(cls, name):
        return {"random": cls.random_mutation, "guided": cls.guided_mutation}.get(name, cls.random_mutation)

class Experiment:
    def __init__(self, config, search_space=None):
        self.config = config; self.search_space = search_space or {}; self.params = config.params or {}
        self.metric_value = None; self.status = "pending"

    def mutate(self, strategy="random", best_params=None, metric_history=None):
        self.params = MutationStrategy.get_strategy(strategy)(self.params, self.search_space, best_params or {}, metric_history or [])
        return self.params

    def set_result(self, metric_value, output="", error=None):
        self.metric_value = metric_value; self.output = output; self.error = error; self.status = "completed"
