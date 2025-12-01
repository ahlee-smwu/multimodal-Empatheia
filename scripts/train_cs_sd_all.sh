#!/usr/bin/env bash
set -e
python -u merg_code/train_stage1.py
python -u merg_code/train_stage2.py
python -u merg_code/train_stage3.py
python -u merg_code/train_stage4.py
