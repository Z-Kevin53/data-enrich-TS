import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from autoresearch import AutoResearchOrchestrator, ProgramConfig

config = ProgramConfig()
config.constraint.max_iterations = 8
config.constraint.max_time_seconds = 30
config.experiment_file = ""

orch = AutoResearchOrchestrator(config=config, llm_backend='heuristic')
orch.setup()

for i in range(8):
    random.seed(42 + i)
    params = {
        'enrich_method': random.choice(['none', 'synthetic', 'oversample', 'noise', 'hybrid']),
        'seed_ratio': random.choice([0.15, 0.25, 0.35, 0.5]),
        'enrich_ratio': random.choice([0.5, 1.0, 2.0]),
        'noise_std': random.choice([0.05, 0.1, 0.2]),
        'hidden_dim': random.choice([32, 64, 128]),
        'dropout': round(random.uniform(0.0, 0.3), 2),
        'learning_rate': random.choice([0.001, 0.005, 0.01]),
        'batch_size': random.choice([16, 32, 64]),
        'optimizer': random.choice(['adam', 'adamw', 'sgd']),
        'weight_decay': random.choice([0.0, 0.001, 0.01]),
        'seed': 42,
    }
    record = orch.run_experiment(params, f'exp_{i+1}')
    orch.process_result(record)

orch.generate_paper()
print(f'\nDone! Best: {orch.tracker.get_best().metric_value:.6f}')
print(f'Results: {os.path.abspath("results/results.tsv")}')
print(f'Paper: {os.path.abspath("papers/generated_paper.md")}')