# 采用gpt-sovits方案，bert-sovits适合长音频训练，gpt-sovits运行短音频快速推理
## 部署tts推理
git clone https://github.com/X-T-E-R/GPT-SoVITS-Inference.git

1. 安装依赖库
```
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```
从 [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，并将它们放置在 `GPT_SoVITS\pretrained_models` 中

2. Model Folder Format
模型文件下载地址 https://www.yuque.com/xter/zibxlp/gsximn7ditzgispg  
下载的模型文件放到trained目录下, 如 `trained/Character1/`  
Put the pth / ckpt / wav files in it, the wav should be named as the prompt text  
Like :

```
trained
--hutao
----hutao-e75.ckpt
----hutao_e60_s3360.pth
----hutao said something.wav
```

3. 启动
后端接口: python Inference/src/tts_backend.py  
如果有错误提示找不到cmudict，从这下载https://github.com/nltk/nltk_data，将packages改名为nltk_data放到home目录下  
管理页面: python Inference/src/TTS_Webui.py, 浏览器打开可以管理character和emotion


4. 接口测试  
  Character and Emotion List  
To obtain the supported characters and their corresponding emotions, please visit the following URL:
- URL: `http://127.0.0.1:5000/character_list`
- Returns: A JSON format list of characters and corresponding emotions
- Method: `GET`

```
{
    "Hanabi": [
        "default",
        "Normal",
        "Yandere",
    ],
    "Hutao": [
        "default"
    ]
}
```