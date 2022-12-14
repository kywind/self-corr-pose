import os
zsp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(zsp)
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
LABELS_DIR = os.path.join(DATA_DIR, 'class_labels')
DETERM_EVAL_SETS = os.path.join(DATA_DIR, 'determ_eval')