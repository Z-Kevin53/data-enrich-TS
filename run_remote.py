"""Run the data_enrich AutoResearch pipeline on the remote GPU server (GPU 0 only).

Server layout (siblings under /home/zwk):
    Joey             - pre-existing folder (untouched)
    autoresearch_wf  - MNIST task project (provides the shared python venv)
    data_enrich      - this project (synced from this folder)

One command does: sync project -> ensure venv -> run run_autoresearch.py on
the server -> pull results/ and papers/ back to this machine.

Usage:
    python run_remote.py --max-experiments 8
    python run_remote.py --sync-only
    python run_remote.py --pull-only
    python run_remote.py --fresh --max-experiments 10
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

# xtcp P2P tunnel: must use 127.0.0.1 (IPv4); 'localhost' tries ::1 first and
# the frpc visitor does not serve it. HostKeyAlias matches the existing
# known_hosts entry "[localhost]:18094".
SSH = [
    "ssh",
    "-p", "18094",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=30",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "HostKeyAlias=[localhost]:18094",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
    "zwk@127.0.0.1",
]
REMOTE_DIR = "/home/zwk/data_enrich"
VENV_SRC = "/home/zwk/autoresearch_wf/.venv"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
SYNC_EXCLUDES = [".git", "data", "results", "papers", "__pycache__", "*.pyc", ".venv",
                 "*.log", "_tmp_*", "_smoke_*", "_ar_run*", "smoke_*", "remote_smoke*"]
PULL_DIRS = ["results", "papers"]


def ssh_run(remote_cmd, stdin=None, retries=3, retry_wait=5, capture=False):
    """Run a command on the server; retry on P2P tunnel instability."""
    last = None
    for attempt in range(1, retries + 1):
        last = subprocess.run(SSH + [remote_cmd], stdin=stdin,
                              capture_output=capture, text=capture)
        if last.returncode == 0:
            return last
        print(f"[WARN] ssh attempt {attempt}/{retries} failed (rc={last.returncode}); "
              f"retrying in {retry_wait}s (P2P tunnel may be re-negotiating)...")
        time.sleep(retry_wait)
    return last


def sync():
    """tar the project (excluded dirs kept for data cache) and stream it over ssh."""
    exclude_args = []
    for pat in SYNC_EXCLUDES:
        exclude_args += ["--exclude", pat]
    print(f"[SYNC] {LOCAL_DIR} -> {REMOTE_DIR} (excluding {SYNC_EXCLUDES})")
    pipe = subprocess.Popen(["tar", "-c", "-f", "-", *exclude_args, "-C", LOCAL_DIR, "."],
                            stdout=subprocess.PIPE)
    proc = ssh_run(f"mkdir -p {REMOTE_DIR} && tar -x -f - -C {REMOTE_DIR}",
                   stdin=pipe.stdout)
    pipe.stdout.close()
    pipe.wait()
    if proc.returncode != 0:
        print("[ERROR] sync failed")
        sys.exit(1)
    print("[SYNC] done")


def setup_remote():
    """Ensure remote dir exists and .venv is linked to the shared venv."""
    check = "import torch, numpy, yaml; print('[ENV] python ok, torch ' + torch.__version__)"
    cmd = (
        f"mkdir -p {REMOTE_DIR} && cd {REMOTE_DIR} && "
        f'if [ ! -e .venv ]; then ln -s {VENV_SRC} .venv; echo "[ENV] venv linked <- {VENV_SRC}"; fi && '
        f'.venv/bin/python -c "{check}"'
    )
    proc = ssh_run(cmd)
    if proc.returncode != 0:
        print("[ERROR] remote setup failed")
        sys.exit(1)


def _remote_launch(forward_args):
    """Launch run_autoresearch.py DETACHED on the server (nohup + pid file).

    Detaching makes the job immune to P2P tunnel drops: the run continues on
    the server while we poll short-lived ssh commands.
    """
    suffix = " ".join(shlex.quote(a) for a in forward_args)
    # GPU pinning (physical GPU 0 via nvidia-smi UUID) is done inside
    # ts_aug_experiment.py; an explicit AR_GPU env var can override it.
    remote_cmd = (
        f"cd {REMOTE_DIR} && "
        f"nohup .venv/bin/python -u run_autoresearch.py {suffix} "
        f"> remote_run.log 2>&1 & echo $! > remote_run.pid".rstrip()
    )
    print(f"[RUN] remote (detached): {remote_cmd}")
    proc = ssh_run(remote_cmd, retries=3, retry_wait=10)
    if proc.returncode != 0:
        print(f"[ERROR] remote launch failed (rc={proc.returncode})")
    return proc.returncode


def _remote_poll(poll_every=30, max_wait=4 * 3600):
    """Poll the detached job until it exits; stream log tail to local stdout."""
    import datetime
    t0 = time.time()
    while True:
        if time.time() - t0 > max_wait:
            print("[ERROR] remote job exceeded max wait; check server remote_run.log")
            return 1
        proc = ssh_run(
            f"cd {REMOTE_DIR} && if [ -f remote_run.pid ] && "
            f"kill -0 $(cat remote_run.pid) 2>/dev/null; then echo RUNNING; "
            f"else echo DONE; fi; tail -3 remote_run.log 2>/dev/null",
            retries=3, retry_wait=10, capture=True)
        out = proc.stdout or ""
        if proc.returncode == 0:
            status, _, tail = out.rpartition("\n")
            tail = tail.strip()
            if tail:
                print(f"[{datetime.datetime.now():%H:%M:%S}] {tail}", flush=True)
            if "DONE" in status:
                # final status: last line of the log
                p2 = ssh_run(f"tail -15 {REMOTE_DIR}/remote_run.log 2>/dev/null",
                             retries=2, capture=True)
                print("[FINAL]", (p2.stdout or "").strip())
                return 0
        time.sleep(poll_every)


def pull():
    """Stream results/ and papers/ back from the server.

    Both dirs go into ONE tar archive: extracting concatenated archives is
    unreliable on Windows bsdtar (it stops after the first end-of-archive).
    """
    remote_cmd = f"cd {REMOTE_DIR} && tar -c -f - {' '.join(PULL_DIRS)}"
    print(f"[PULL] {'/'.join(PULL_DIRS)} -> {LOCAL_DIR}")
    pipe = subprocess.Popen(SSH + [remote_cmd], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-f", "-", "-C", LOCAL_DIR], stdin=pipe.stdout)
    pipe.stdout.close()
    pipe.wait()
    # rc 0 = all dirs present; rc 2 = some missing (fine on a fresh server)
    if pipe.returncode not in (0, 2):
        print(f"[WARN] pull rc={pipe.returncode}")
    print("[PULL] done")


def main():
    parser = argparse.ArgumentParser(
        description="Run the data_enrich AutoResearch pipeline on the remote GPU server (GPU 0 only)")
    parser.add_argument("--no-sync", action="store_true", help="skip project sync")
    parser.add_argument("--no-pull", action="store_true", help="skip pulling results back")
    parser.add_argument("--sync-only", action="store_true", help="only sync, do not run")
    parser.add_argument("--fresh", action="store_true",
                        help="wipe remote results/papers/.git for a clean history")
    parser.add_argument("--pull-only", action="store_true",
                        help="only pull results back, do not run anything")
    parser, forward = parser.parse_known_args()

    if not parser.no_sync:
        sync()
        setup_remote()
    if parser.fresh:
        proc = ssh_run(f"cd {REMOTE_DIR} && rm -rf results papers .git")
        if proc.returncode == 0:
            print("[FRESH] remote results/papers/.git wiped")
    if parser.sync_only or parser.pull_only:
        if parser.pull_only:
            pull()
        return
    rc = _remote_launch(forward)
    if rc == 0:
        rc = _remote_poll()
    if rc != 0:
        sys.exit(rc)
    if not parser.no_pull:
        pull()
    print("[DONE] remote data_enrich run complete")


if __name__ == "__main__":
    main()