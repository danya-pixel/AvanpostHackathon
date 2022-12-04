import os
import shutil
import json
from pathlib import Path
import asyncio

from tasks.objects import celery, tmp_path
from tasks.utils import prepare_files


@celery.task(name="predict_by_model", bind=True)
def predict_by_model(self, images_url, models_url):
    from ml.finetuner import predict_samples
    task_id = self.request.id
    task_folder = tmp_path / task_id
    if not task_folder.exists():
        os.mkdir(task_folder)
    prepare_files(task_folder, images_url, "images")
    prepare_files(task_folder, models_url, "model")

    with open(task_folder / "model" / "config.json", "r") as f:
        model_config = json.load(f)
    classes = model_config['classes']
    model_name = model_config['model_name']

    result = predict_samples(classes_names=classes,
                             pth_path=str((task_folder / "model" / model_name).resolve()),
                             new_data_dir=str((task_folder / "images").resolve())
                             )
    # output_result = {}
    # for path, class_idx in result.items():
    #     output_result[Path(path).name] = classes[class_idx]

    shutil.rmtree(task_folder)
    return result, classes
