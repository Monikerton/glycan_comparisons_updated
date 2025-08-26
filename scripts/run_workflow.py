#!/usr/bin/env python3
"""
scripts/run_workflow.py

A lightweight runner that loads workflows from config/workflow.yaml
and invokes each step's `run(params)` function using YAML parameters.
"""
import sys
import importlib
from pathlib import Path
import yaml
import traceback

# Path to the central configuration
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "workflow.yaml"


def load_config() -> dict:
    """Load and parse the YAML configuration."""
    if not CONFIG_PATH.is_file():
        print(f"Config not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    # Usage check
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <workflow_name>")
        sys.exit(1)
    workflow_name = sys.argv[1]

    # Load config
    cfg = load_config()
    workflows = cfg.get("workflows", {})
    if workflow_name not in workflows:
        print(f"Workflow '{workflow_name}' not defined in {CONFIG_PATH}")
        sys.exit(1)

    # Retrieve params for this workflow
    db_cfg = cfg.get("databases", {}).get(workflow_name, {})
    params = db_cfg.get("params", {})

    # Execute each step in order
    for step in workflows[workflow_name]:
        print(f"→ Running: {step}")
        try:
            module = importlib.import_module(step)
            run_fn = getattr(module, "run", None)
            if not callable(run_fn):
                raise AttributeError(f"Module '{step}' has no callable run()")
            run_fn(params)
        except Exception as e:
            # Extract the deepest traceback frame
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            fname = tb.tb_frame.f_code.co_filename
            lineno = tb.tb_lineno
            print(f"Error in step '{step}': {e} (File '{fname}', line {lineno})")
            # Optionally, print full traceback for detail
            traceback.print_exc()
            sys.exit(1)


    print(f"Workflow '{workflow_name}' completed successfully.")


if __name__ == "__main__":
    main()
