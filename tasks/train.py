import json
import os
import shutil
from pathlib import Path

from tasks.objects import tmp_path, celery, DATASET_PATH
from tasks.utils import prepare_files


@celery.task(name="predict_by_model", bind=True)
def finetune_model(self, images_url, models_url):
    from ml.finetuner import finetune_model
    task_id = self.request.id
    task_folder = tmp_path / task_id
    if not task_folder.exists():
        os.mkdir(task_folder)
    prepare_files(task_folder, models_url, "model")

    with open(task_folder / "model" / "config.json", "r") as f:
        model_config = json.load(f)
    classes = model_config['classes']
    model_name = model_config['model_name']
    result = finetune_model(data_dir=DATASET_PATH,
                            classes_names=classes,
                            pth_path=str((task_folder / "model" / model_name).resolve()),
                            new_data_dir=str((task_folder / "images").resolve())
                            )
    output_result = {}
    for path, class_idx in result.items():
        output_result[Path(path).name] = classes[class_idx]

    shutil.rmtree(task_folder)
    return output_result
