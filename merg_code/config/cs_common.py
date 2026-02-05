
import os, yaml
from types import SimpleNamespace

def load_cs_config(path):
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return SimpleNamespace(**raw)
