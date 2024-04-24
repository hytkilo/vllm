import redis
import json
from vllm.lora.request import LoRARequest
import requests
import os
from urllib.parse import urlparse
import zipfile

redis_host = '10.12.0.16'
redis_port = 6379
redis_password = '2sSwwJ7@f8UT'

r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True, db=5)

key_prefix = 'YX:MODEL:LORA:'
model_zip_dir = '/data/riki_vllm_models/zip/'
model_models_dir = '/data/riki_vllm_models/models/'


def extract_filename_from_url(url):
    # 解析URL
    parsed_url = urlparse(url)
    # 提取路径部分
    path = parsed_url.path
    # 分割路径以获取文件名
    filename = path.split('/')[-1]
    return filename


def download_file(url, filename):
    # 发送GET请求
    r = requests.get(url)
    # 确保请求成功
    r.raise_for_status()

    # 以二进制写模式打开一个文件
    with open(model_zip_dir + filename, 'wb') as f:
        f.write(r.content)
    print(f"模型下载完毕，地址：{url}")


def download_and_unzip(modelUrl, modelFile):
    download_file(modelUrl, modelFile)
    with zipfile.ZipFile(model_zip_dir + modelFile, 'r') as zip_ref:
        zip_ref.extractall(model_models_dir + modelFile.split('.')[0] + '/')
    print(f"模型解压完毕")
    pass


def get_path(model_info):
    modelUrl = model_info['modelUrl']
    modelFile = extract_filename_from_url(modelUrl)
    path = model_models_dir + modelFile.split('.')[0]
    if os.path.isfile(model_zip_dir + modelFile) is not True:
        download_and_unzip(modelUrl, modelFile)
    return path


def get_lora(request):
    if 'qwen/' in request.model:
        return None
    model_info_str = r.get(key_prefix + request.model)
    model_info = json.loads(model_info_str)
    path = get_path(model_info)
    return LoRARequest(
        lora_name=request.model,
        lora_int_id=model_info['id'],
        lora_local_path=path,
    )
