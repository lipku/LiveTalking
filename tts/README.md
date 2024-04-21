# 采用gpt-sovits方案，bert-sovits适合长音频训练，gpt-sovits运行短音频快速推理
## 部署tts推理
git clone https://github.com/X-T-E-R/GPT-SoVITS-Inference.git

## 1. 安装依赖库
```
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```
从 [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，并将它们放置在 `GPT_SoVITS\pretrained_models` 中

## 2. Model Folder Format
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

## 3. 启动
### 3.1 后端服务: 
python Inference/src/tts_backend.py  
如果有错误提示找不到cmudict，从这下载https://github.com/nltk/nltk_data，将packages改名为nltk_data放到home目录下
### 3.2 管理character: 
python Inference/src/Character_Manager.py  
浏览器打开可以管理character和emotion
### 3.3 测试tts功能: 
python Inference/src/TTS_Webui.py  


## 4. 接口说明  
### 4.1 Character and Emotion List  
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

### 4.2 Text-to-Speech

- URL: `http://127.0.0.1:5000/tts`
- Returns:  Audio on success. Error message on failure.
- Method: `GET`/`POST`
```
{
    "method": "POST",
    "body": {
        "character": "${chaName}",
        "emotion": "${Emotion}",
        "text": "${speakText}",
        "text_language": "${textLanguage}",
        "batch_size": ${batch_size},
        "speed": ${speed},
        "top_k": ${topK},
        "top_p": ${topP},
        "temperature": ${temperature},
        "stream": "${stream}",
        "format": "${Format}",
        "save_temp": "${saveTemp}"
    }
}
```

##### Parameter Explanation

- **text**: The text to be converted, URL encoding is recommended.
- **character**: Character folder name, pay attention to case sensitivity, full/half width, and language.
- **emotion**: Character emotion, must be an actually supported emotion of the character, otherwise, the default emotion will be used.
- **text_language**: Text language (auto / zh / en / ja), default is multilingual mixed. 
- **top_k**, **top_p**, **temperature**: GPT model parameters, no need to modify if unfamiliar.

- **batch_size**: How many batches at a time, can be increased for faster processing if you have a powerful computer, integer, default is 1.
- **speed**: Speech speed, default is 1.0.
- **save_temp**: Whether to save temporary files, when true, the backend will save the generated audio, and subsequent identical requests will directly return that data, default is false.
- **stream**: Whether to stream, when true, audio will be returned sentence by sentence, default is false.
- **format**: Format, default is WAV, allows MP3/ WAV/ OGG.

## 部署tts训练
https://github.com/RVC-Boss/GPT-SoVITS  
根据文档说明部署，将训练后的模型拷到推理服务的trained目录下