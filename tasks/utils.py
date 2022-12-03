import zipfile

from requests import get


def download_file(url):
    r = get(url)
    if r.status_code != 200:
        raise ValueError("Хуйня ваши модельки")
    return r.content


def prepare_files(task_folder, url, type='model'):
    with open(task_folder / f"{type}.zip", "wb") as f:
        f.write(download_file(url))
    with zipfile.ZipFile(task_folder / f"{type}.zip", 'r') as zip_model:
        zip_model.extractall(task_folder / type)
