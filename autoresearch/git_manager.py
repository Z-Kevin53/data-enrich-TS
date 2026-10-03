import subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime

class GitManager:
    def __init__(self, repo_path=".", branch="autoresearch/experiment"):
        self.repo_path = Path(repo_path); self.branch = branch; self._verify_git_repo()

    def _run_git(self, *args, check=True):
        cmd = ["git"] + list(args)
        return subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=check)

    def _verify_git_repo(self):
        result = self._run_git("rev-parse --is-inside-work-tree", check=False)
        if result.returncode != 0:
            self._run_git("init", check=False)
            self._run_git("config", "user.email", "autoresearch@auto.local", check=False)
            self._run_git("config", "user.name", "AutoResearch Agent", check=False)
        if self._run_git("rev-parse --verify HEAD", check=False).returncode != 0:
            self._run_git("commit", "--allow-empty", "-m", "Initial commit", check=False)

    def create_branch(self, branch_name=None) -> str:
        if branch_name is None:
            branch_name = f"autoresearch/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_git("checkout", "main" if self._branch_exists("main") else "master", check=False)
        self._run_git("checkout", "-b", branch_name, check=False)
        self.branch = branch_name
        return branch_name

    def _branch_exists(self, branch_name: str) -> bool:
        return self._run_git("rev-parse", "--verify", branch_name, check=False).returncode == 0

    def commit(self, message: str, files=None) -> str:
        if files:
            for f in files: self._run_git("add", f, check=False)
        else: self._run_git("add", "-A", check=False)
        self._run_git("commit", "-m", message, check=False)
        return self._run_git("rev-parse", "HEAD", check=True).stdout.strip()

    def get_current_branch(self) -> str:
        return self._run_git("rev-parse", "--abbrev-ref", "HEAD", check=True).stdout.strip()

    def get_current_commit(self) -> str:
        return self._run_git("rev-parse", "HEAD", check=True).stdout.strip()

    def get_diff(self, files=None) -> str:
        if files: return self._run_git("diff", *files, check=False).stdout
        return self._run_git("diff", check=False).stdout

    def reset_to_commit(self, commit_hash: str):
        self._run_git("checkout", commit_hash, "--", ".", check=False)

    def stash(self): self._run_git("stash", "push", check=False)
