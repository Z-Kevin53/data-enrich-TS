import argparse, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(str(Path(__file__).parent))
from autoresearch import AutoResearchOrchestrator, ProgramConfig

def main():
    parser = argparse.ArgumentParser(description="AutoResearch: Automated Train-Fine-Tune-Paper Pipeline")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--llm-backend", type=str, default="heuristic", choices=["heuristic", "openai", "ollama"])
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--config", type=str, default="program_config.yaml")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--paper-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = ProgramConfig.load(args.config)
    else:
        config = ProgramConfig(); config.save(args.config)

    if args.demo:
        config.experiment_file = ""
        config.constraint.max_time_seconds = 30
        config.constraint.max_iterations = args.max_experiments or 5

    orchestrator = AutoResearchOrchestrator(config=config, llm_backend=args.llm_backend, api_key=args.api_key, model=args.model)
    if args.status: print(f"Status: {orchestrator.status()}"); return
    if args.paper_only: orchestrator.generate_paper(); print("Paper generated."); return
    print(f"\nAutoResearch Workflow Starting - Mode: {'Demo' if args.demo else 'Full'}\n")
    orchestrator.run_loop(max_experiments=args.max_experiments, max_time=args.max_time)
    status = orchestrator.status()
    print(f"\nFinal: {status['experiments_run']} experiments, Best: {status['best_metric']:.6f}")

if __name__ == "__main__": main()
