# ggml-downloader

**Problem:** huggingface `download_model` only supports parallel download when the model is chunked.

GGML models can be quite large (30B+ especially) but chunking is not supported its always a single .bin file.

**Solution:** use pypdl library that implements multi-threaded downloading via dynamic chunking

## Requirements

* [pypdl](https://github.com/m-jishnu/pypdl) :heart_eyes:
* [huggingface_hub](https://github.com/huggingface/huggingface_hub) :rocket:
* requests

`pip install -r requirements.txt`

## Usage - Command line

`./download.py <model> <quant> [branch]`

`<model>` is the model you're downloading for example `TheBloke/vicuna-33B-GGML`

`<quant>` is the quantization you're downloading for example `q5_0`

`[branch]` is optional, if omitted will download from first avilable branch

## Usage - Library

1. Import the helper functions: `from download import get_filenames, build_url, get_redirect_header, parallel_download`

2. Get the branch and filename of the quant you're looking for: `get_filenames(model_name, quant)` returns `(branch, filename)`

3. Build the HF download URL: `build_url(model_name, branch, filename)` returns `url`

4. Get the LFS URL: `get_redirect_header(url)` returns `lfs_url`

5. Download the file: `parallel_download(lfs_url, filename)` will create `filename` in the current directory