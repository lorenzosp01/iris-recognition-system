from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset_path='sondosaabed/casia-iris-thousand', path='./datasets'):
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset_path, path=path, unzip=True)

if __name__ == '__main__':
    download_dataset()