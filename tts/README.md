# 采用gpt-sovits方案，bert-sovits适合长音频训练，gpt-sovits运行短音频快速推理
## 部署tts推理
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
git checkout fast_inference_
## 1. 安装依赖库
```
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```
从 [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，并将它们放置在 `GPT_SoVITS/GPT_SoVITS/pretrained_models` 中

注意
```
是将 GPT-SoVITS  的模型文件放入 pretrained_models目录中
```
如下
```
pretrained_models/
--chinese-hubert-base
--chinese-roberta-wwm-ext-large
s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
s2D488k.pth
s2G488k.pth
```

## 3. 启动
### 3.1 启动webui界面（测试效果用）
python GPT_SoVITS/inference_webui.py

### 3.2 启动api服务: 
python api_v3.py


## 4. 接口说明  

### 4.1 Text-to-Speech

endpoint: `/tts`  
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                                                 # str.(required) text to be synthesized
    "text_lang": "",                                            # str.(required) language of the text to be synthesized
    "ref_audio_path": "",                                       # str.(required) reference audio path.
    "prompt_text": "",                                          # str.(optional) prompt text for the reference audio
    "prompt_lang": "",                                          # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                                                 # int.(optional) top k sampling
    "top_p": 1,                                                 # float.(optional) top p sampling
    "temperature": 1,                                           # float.(optional) temperature for sampling
    "text_split_method": "cut5",                                # str.(optional) text split method, see text_segmentation_method.py for details.
    "batch_size": 1,                                            # int.(optional) batch size for inference
    "batch_threshold": 0.75,                                    # float.(optional) threshold for batch splitting.
    "split_bucket": true,                                       # bool.(optional) whether to split the batch into multiple buckets.
    "speed_factor":1.0,                                         # float.(optional) control the speed of the synthesized audio.
    "fragment_interval":0.3,                                    # float.(optional) to control the interval of the audio fragment.
    "seed": -1,                                                 # int.(optional) random seed for reproducibility.
    "media_type": "wav",                                        # str.(optional) media type of the output audio, support "wav", "raw", "ogg", "aac".
    "streaming_mode": false,                                    # bool.(optional) whether to return a streaming response.
    "parallel_infer": True,                                     # bool.(optional) whether to use parallel inference.
    "repetition_penalty": 1.35,                                 # float.(optional) repetition penalty for T2S model.
    "tts_infer_yaml_path": “GPT_SoVITS/configs/tts_infer.yaml”  # str.(optional) tts infer yaml path
}
```

## 部署tts训练
https://github.com/RVC-Boss/GPT-SoVITS  
切换自己训练的模型  
### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/xxx.ckpt
```
RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/xxx.pth
```

RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400
    
"""

## 如果你需要使用autodl 进行部署 
请使用  https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS 作为基础镜像你能快速进行部署
