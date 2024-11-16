import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset_path='sondosaabed/casia-iris-thousand', path='./datasets'):

    dataset_name = dataset_path.split('/')[-1]
    if not os.path.exists(path+dataset_name):
        os.mkdir(path+dataset_name)
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(dataset_path, path=path, unzip=True)


if __name__ == '__main__':
    download_dataset()