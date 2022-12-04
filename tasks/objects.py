import os
from pathlib import Path

from utils.utills import make_celery

celery = make_celery()

tmp_path = Path('tmp')
if not tmp_path.exists():
    os.mkdir(tmp_path)

DATASET_PATH = '/home/danya-sakharov/AvanpostHackathon/dataset'
