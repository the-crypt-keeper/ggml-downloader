#!/usr/bin/env python3
from pypdl import Downloader
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import sys
import requests

def get_redirect_header(url):
    session = requests.Session()
    response = session.get(url, allow_redirects=False)
    redirect_url = response.headers.get('Location')
    return redirect_url

def get_filenames(model_name, quant, branch = None):
    api = HfApi()
    
    try:
        branches = api.list_repo_refs("gpt2").branches
        if branch is None:
            branch = branches[0].name
        else:
            branch = branches[branches.index(branch)].name

        files = api.list_files_info(model_name)
        for file_info in files:
            if (file_info.rfilename.find(quant) != -1):
                return branch, file_info.rfilename

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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: download.py <model_name> <quant> [branch]')
        sys.exit(1)

    model_name = sys.argv[1]
    quant = sys.argv[2]

    branch, filename = get_filenames(model_name, quant, sys.argv[3] if len(sys.argv) > 3 else None)
    if filename is None:
        sys.exit(1)

    url = build_url(model_name, branch, filename)
    print(f"Downloading {filename} from {url}")

    lfs_url = get_redirect_header(url)

    parallel_download(lfs_url, filename)