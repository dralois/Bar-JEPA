import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
MAIN       = ROOT / "bar-jepa" / "main.py"
EVAL_CFGS  = ROOT / "bar-jepa" / "configs" / "eval"

CONFIGS = [
    "classic_arp",
    "classic_ctt",
    "classic_noarp",
    "classic_vanilla",
    "simple_arp",
]

DATASETS = [
    ("UBPMC",  ROOT / "UBPMC", True),
    ("Charts", ROOT / "data",  False),
]

for path in [MAIN, EVAL_CFGS, *(ds_path for _, ds_path, _ in DATASETS)]:
    if not path.exists():
        raise FileNotFoundError(f"Expected path not found: {path}")

for config in CONFIGS:
    cfg_file = EVAL_CFGS / f"{config}.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Eval config not found: {cfg_file}")
    for ds_name, ds_path, is_ubpmc in DATASETS:
        print(f"\n{'='*60}")
        print(f"  {config}  |  {ds_name}")
        print(f"{'='*60}\n")
        subprocess.run(
            [
                sys.executable, str(MAIN),
                "--mode", "eval",
                "--fname", str(cfg_file),
                "--devices", "mps",
                "--override",
                f"data.root_path={ds_path}",
                f"data.is_ubpmc={str(is_ubpmc).lower()}",
            ],
            cwd=ROOT,
            check=True,
        )
