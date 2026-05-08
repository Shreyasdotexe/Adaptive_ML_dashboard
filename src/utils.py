import time, uuid
from datetime import datetime

RESET="\033[0m"; BOLD="\033[1m"; GREEN="\033[92m"; YELLOW="\033[93m"
CYAN="\033[96m"; RED="\033[91m"; DIM="\033[2m"; WHITE="\033[97m"

def print_banner():
    print(f"\n{CYAN}{BOLD}{'='*64}{RESET}")
    print(f"{CYAN}{BOLD}   ADAPTIVE ML SYSTEM — CONCEPT DRIFT DETECTION{RESET}")
    print(f"{CYAN}{BOLD}{'='*64}{RESET}\n")

def print_section(title):
    print(f"\n{YELLOW}{BOLD}  > {title}{RESET}")
    print(f"{DIM}  {'-'*56}{RESET}")

def print_ok(msg):    print(f"  {GREEN}OK{RESET} {msg}")
def print_info(msg):  print(f"  {CYAN}i{RESET}  {msg}")
def print_warn(msg):  print(f"  {YELLOW}!{RESET}  {msg}")

def print_drift_alert(index, value):
    print(f"  {RED}{BOLD} DRIFT DETECTED{RESET}  at sample {BOLD}{index:>6}{RESET}  signal={CYAN}{value:.4f}{RESET}")

def print_progress(current, total, interval=500):
    if current % interval == 0 or current == total - 1:
        pct = (current + 1) / total * 100
        filled = int(30 * (current + 1) / total)
        bar = "#" * filled + "." * (30 - filled)
        print(f"\r  {DIM}[{bar}]{RESET} {WHITE}{pct:5.1f}%{RESET}  {DIM}sample {current+1}/{total}{RESET}",
              end="", flush=True)
    if current == total - 1:
        print()

def print_summary(summary):
    print()
    for k, v in summary.items():
        print(f"  {DIM}{(k+':').ljust(32)}{RESET}{WHITE}{v}{RESET}")

def print_footer(elapsed):
    print(f"\n{CYAN}{BOLD}{'='*64}{RESET}")
    print(f"{GREEN}{BOLD}   Complete  {RESET}{DIM}— elapsed: {elapsed:.2f}s{RESET}")
    print(f"{CYAN}{BOLD}{'='*64}{RESET}\n")

class Timer:
    def __init__(self): self._start = None
    def start(self): self._start = time.time()
    def elapsed(self): return time.time() - self._start if self._start else 0.0

def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def validate_config(config):
    assert config.get("n_features", 1) >= 1
    assert 0 < config.get("adwin_delta", 0.1) < 1
    assert config.get("warmup_size", 10) >= 10
    assert config.get("cooldown_samples", 1) >= 1
