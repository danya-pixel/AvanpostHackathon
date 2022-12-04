import asyncio
import json
import os
import shutil
from pathlib import Path

from requests import post

from parsers.YandexParser.ImageParser import main
from ml.finetuner import finetune_model
from tasks.objects import tmp_path, celery, DATASET_PATH
from tasks.utils import prepare_files
from utils.upload_to_fds import JWT_TOKEN


@celery.task(name="train_model", bind=True)
def finetune_model(self, request, models_url):
    from ml.finetuner import finetune_model
    task_id = self.request.id
    task_folder = tmp_path / task_id
    if not task_folder.exists():
        os.mkdir(task_folder)
    prepare_files(task_folder, models_url, "model")
    if not (task_folder / "images").exists():
        os.mkdir(task_folder / "images")
    asyncio.run(main(request, task_folder / "images"))
    with open(task_folder / "model" / "config.json", "r") as f:
        model_config = json.load(f)

    classes = model_config['classes']
    model_name = model_config['model_name']
    result = finetune_model(data_dir=DATASET_PATH,
                            classes_names=classes,
                            pth_path=str((task_folder / "model" / model_name).resolve()),
                            new_data_dir=str((task_folder / "images").resolve()),
                            new_data_name=request
                            )

    pth_model = result['pth_path']
    with open(pth_model, 'rb') as f:
        files = {'file': ('model.pth',f,'application/gzip')}
        headers = {
            "Authorization": f"Bearer {JWT_TOKEN}"
        }
        r = post("https://fds.es.nsu.ru/upload/", files=files, headers=headers)
        if r.status_code != 200:
            raise ValueError("Upload error")
        result['pth_url'] = "https://fds.es.nsu.ru/uploads/" + r.json()['file_id']
    # output_result = {}
    # for path, class_idx in result.items():
    #     output_result[Path(path).name] = classes[class_idx]

    # shutil.rmtree(task_folder)
    return result, classes, pth_model
