# ============================================================
# config.py
# Loads the project configuration from config.yaml.
# All other scripts import load_config() to get hyperparameters
# and paths. This keeps configuration in one central place.
# ============================================================

import os
import yaml


def load_config(config_path: str = None) -> dict:
    # If no path is given, use the config.yaml at the project root
    if config_path is None:
        # __file__ is Code/utils/config.py, so go up two levels
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, "config.yaml")

    # Load and parse the YAML file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"[config] Loaded config from {config_path}")
    return config
