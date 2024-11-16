import os
from dotenv import load_dotenv
load_dotenv()

os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')

def case_insensitive_exists(path):
    """
    Check if a path exists, ignoring case sensitivity.
    """
    dir_name, base_name = os.path.split(path)
    if not os.path.exists(dir_name):
        return False

    # Check case-insensitively in the directory
    return base_name.lower() in (entry.lower() for entry in os.listdir(dir_name))


def download_dataset(dataset_path='sondosaabed/casia-iris-thousand', path='./datasets'):
    from kaggle.api.kaggle_api_extended import KaggleApi

    dataset_name = dataset_path.split('/')[-1]
    dataset_dir = os.path.join(path, dataset_name)

    if not case_insensitive_exists(dataset_dir):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_path, path=path, unzip=True)


if __name__ == '__main__':
    download_dataset()