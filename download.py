#!/usr/bin/env python3
from pypdl import Downloader
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import requests
import fire

def get_redirect_header(url):
    session = requests.Session()
    response = session.get(url, allow_redirects=False)
    redirect_url = response.headers.get('Location')
    return url if redirect_url is None else redirect_url

def get_filenames(model_name, quant, branch = None):
    api = HfApi()
    
    try:
        branches = api.list_repo_refs(model_name).branches
        if branch is None:
            branch = branches[0].name
        else:
            branch = branches[branches.index(branch)].name

        files = api.list_files_info(model_name)
        found = False
        for file_info in files:
            if (file_info.rfilename.find(quant) != -1 or quant == '*'):
                found = True
                yield branch, file_info.rfilename
        
        if not found:
            print('Quant not found: ' + quant)
            return None, None
    except RepositoryNotFoundError:
        print('Model not found: ', model_name)
        return None, None

def build_url(model_name, branch, filename):
    return f'https://huggingface.co/{model_name}/resolve/{branch}/{filename}'

def parallel_download(lfs_url, filename):
    dl = Downloader()
    dl.start(lfs_url, filename)

def download_model(model_name : str, quant : str = ".bin", branch : str = ""):
    """ Downloads a quantized model from hugging face hub using parallel download streams"""
    for branch, filename in get_filenames(model_name, quant, branch if branch else None):
        if filename is None: break
        url = build_url(model_name, branch, filename)
        print(f"Downloading {filename} from {url}")
        lfs_url = get_redirect_header(url)
        parallel_download(lfs_url, filename)

if __name__ == "__main__":
    fire.Fire(download_model)