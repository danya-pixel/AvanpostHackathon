import os
import zipfile
import shutil
import json
from requests import get
from pathlib import Path
from utils.utills import make_celery


celery = make_celery()

tmp_path = Path('tmp')
if not tmp_path.exists():
    os.mkdir(tmp_path)


def download_file(url):
    r = get(url)
    if r.status_code != 200:
        raise ValueError("Хуйня ваши модельки")
    return r.content


def prepare_files(task_folder, url, type='model'):
    with open(task_folder/f"{type}.zip", "wb") as f:
        f.write(download_file(url))
    with zipfile.ZipFile(task_folder/f"{type}.zip", 'r') as zip_model:
        zip_model.extractall(task_folder/type)


@celery.task(name="predict_by_model", bind=True)
def predict_by_model(self, images_url, models_url):
    from ml.finetuner import predict_samples
    task_id = self.request.id
    task_folder = tmp_path/task_id
    if not task_folder.exists():
        os.mkdir(task_folder)
    prepare_files(task_folder, images_url, "images")
    prepare_files(task_folder, models_url, "model")

    with open(task_folder / "model" / "config.json", "r") as f:
        model_config = json.load(f)
    classes = model_config['classes']
    model_name = model_config['model_name']
    print(classes, )
    result = predict_samples(classes_names=classes,
                    pth_path=(task_folder / "model" / model_name).resolve(),
                    new_data_dir=(task_folder / "images").resolve()
                    )
    
    # TODO: uncomment it
    # shutil.rmtree(task_folder)
    return result

if __name__ == "__main__":
    from ml.finetuner import predict_samples
