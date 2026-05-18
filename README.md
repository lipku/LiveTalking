
 # [English](./README-EN.md) | ä¸­æç  
 <p align="center">
 <img src="./assets/LiveTalking-logo.jpg" align="middle" width = "300"/>
<p align="center">
<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/lipku/LiveTalking/releases"><img src="https://img.shields.io/github/v/release/lipku/LiveTalking?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/lipku/LiveTalking/graphs/contributors"><img src="https://img.shields.io/github/contributors/lipku/LiveTalking?color=c4f042&style=flat-square"></a>
    <a href="https://github.com/lipku/LiveTalking/network/members"><img src="https://img.shields.io/github/forks/lipku/LiveTalking?color=8ae8ff"></a>
    <a href="https://github.com/lipku/LiveTalking/stargazers"><img src="https://img.shields.io/github/stars/lipku/LiveTalking?color=ccf"></a>
</p>

 å®æ¶äº¤äºæµå¼æ°å­äººï¼å®ç°é³è§é¢åæ­¥å¯¹è¯ãåºæ¬å¯ä»¥è¾¾å°åç¨ææ  
[wav2lipææ](https://www.bilibili.com/video/BV1scwBeyELA/) | [ernerfææ](https://www.bilibili.com/video/BV1G1421z73r/) | [musetalkææ](https://www.bilibili.com/video/BV1bUwezvEnG/)  
å½åéåå°å:<https://gitee.com/lipku/LiveTalking> 

## ä¸ºé¿åä¸3dæ°å­äººæ··æ·ï¼åé¡¹ç®metahuman-streamæ¹åä¸ºlivetalkingï¼åæé¾æ¥å°åç»§ç»­å¯ç¨

## Table of Contents

- [Features](#features)
- [Installation](#1-installation)
- [Quick Start](#2-quick-start)
- [Architecture](#3-architecture)
- [More Usage](#4-more-usage)
- [Docker](#5-docker-run)
- [Performance](#6-性能)

## Features
1. æ¯æå¤ç§æ°å­äººæ¨¡å: ernerfãmusetalkãwav2lipãUltralight-Digital-Human
2. æ¯æå£°é³åé
3. æ¯ææ°å­äººè¯´è¯è¢«ææ­
4. æ¯æwebrtcãrtmpãèææåå¤´è¾åº
5. æ¯æå¨ä½ç¼æï¼ä¸è¯´è¯æ¶æ­æ¾èªå®ä¹è§é¢
6. æ¯æå¤å¹¶å
7. æ¯æèªå®ä¹æ°å­äººå½¢è±¡

## 1. Installation

Tested on Ubuntu 24.04, Python3.10, Pytorch 2.5.0 and CUDA 12.4

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
#å¦æcudaçæ¬ä¸ä¸º12.4(è¿è¡nvidia-smiç¡®è®¤çæ¬)ï¼æ ¹æ®<https://pytorch.org/get-started/previous-versions/>å®è£å¯¹åºçæ¬çpytorch 
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
``` 
å®è£å¸¸è§é®é¢[FAQ](https://livetalking-doc.readthedocs.io/zh-cn/latest/faq.html)  
linux cudaç¯å¢æ­å»ºå¯ä»¥åèè¿ç¯æç«  <https://zhuanlan.zhihu.com/p/674972886>  
è§é¢è¿ä¸ä¸è§£å³æ¹æ³ <https://mp.weixin.qq.com/s/MVUkxxhV2cgMMHalphr2cg>


## 2. Quick Start
- ä¸è½½æ¨¡å  
å¤¸åäºç<https://pan.quark.cn/s/83a750323ef0>    
GoogleDriver <https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing>  
å°wav2lip256.pthæ·å°æ¬é¡¹ç®çmodelsä¸, éå½åä¸ºwav2lip.pth;  
å°wav2lip256_avatar1.tar.gzè§£ååæ´ä¸ªæä»¶å¤¹æ·å°æ¬é¡¹ç®çdata/avatarsä¸
- è¿è¡  
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1  
<font color=red>æå¡ç«¯éè¦å¼æ¾ç«¯å£ tcp:8010; udp:1-65536 </font>  
å®¢æ·ç«¯å¯ä»¥éç¨ä»¥ä¸ä¸¤ç§æ¹å¼:  
(1)ç¨æµè§å¨æå¼http://serverip:8010/webrtcapi.html , åç¹âstart',æ­æ¾æ°å­äººè§é¢ï¼ç¶åå¨ææ¬æ¡è¾å¥ä»»ææå­ï¼æäº¤ãæ°å­äººæ­æ¥è¯¥æ®µæå­  
(2)ç¨å®¢æ·ç«¯æ¹å¼, ä¸è½½å°å<https://pan.quark.cn/s/d7192d8ac19b>   

- å¿«éä½éª  
[å¨çº¿éå](https://www.compshare.cn/images/4458094e-a43d-45fe-9b57-de79253befe4?referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking) ç¨è¯¥éååå»ºå®ä¾å³å¯è¿è¡æå

å®è£è¿è¡è¿ç¨ä¸­å¦æè®¿é®ä¸äºhuggingfaceï¼å¨è¿è¡å
```
export HF_ENDPOINT=https://hf-mirror.com
``` 

## 3. Architecture
### æ°æ®æµç¨å¾
<img src="./assets/dataflow.png" align="middle" />  

### ç³»ç»æ¶æå¾

```mermaid
graph TD
    User["User / Frontend Web"] -->|"Text Input / Audio File"| API["API Routes: /human, /humanaudio"]
    
    subgraph "Server Layer"
        API --> SessionMgr["Session Manager"]
        SessionMgr --> AvatarSession["Avatar Session Instance"]
    end

    subgraph "Logic Layer"
        AvatarSession -->|"Request Type: chat"| LLM["LLM Response Engine"]
        LLM -->|"Generated Text"| TTS["TTS Engine: Edge/CosyVoice/Tencent..."]
        AvatarSession -->|"Request Type: echo"| TTS
        TTS -->|"PCM Audio (16k)"| ASR["Audio Feature Extraction"]
        API -->|"Uploaded audio"| ASR
    end

    subgraph "Rendering Layer"
        ASR -->|"Audio Features / Mel"| Infer["Inference Engine: Wav2Lip/MuseTalk/ERNeRF"]
        Infer -->|"Generated Mouth Sync"| Paste["Paste Back"]
    end

    subgraph "Streaming Layer"
        Paste -->|"Video Frames"| Output["Output Module: WebRTC/RTMP/Virtualcam"]
        ASR -->|"Audio Frames"| Output
        Output -->|"Real-time Media Stream"| User
    end

    subgraph "Modular Plugin System"
        Reg["Registry"] -.-> TTS
        Reg -.-> Infer
        Reg -.-> Output
    end

    style User fill:#f9f,stroke:#333,stroke-width:2px
    style Reg fill:#fff2cc,stroke:#d6b656,stroke-width:2px
    style LLM fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px
    style Infer fill:#d5e8d4,stroke:#82b366,stroke-width:2px
```

### 1. APIå±
- **æ¥å£ç«¯ç¹**ï¼
    - `/human`ï¼æ¥æ¶ææ¬ï¼ç¨äºâï¼echoï¼âï¼ç´æ¥æ­æ¾ï¼æâèå¤©ï¼chatï¼âï¼å¤§è¯­è¨æ¨¡åäº¤äºï¼åºæ¯ã
    - `/humanaudio`ï¼æ¥æ¶åå§é³é¢æä»¶ç¨äºæ­æ¾ã
- **ä¼è¯ç®¡ç**ï¼æ¯ä¸ªè¿æ¥é½ä¼åéä¸ä¸ª`sessionid`ï¼ç¨äºç»´æ¤ç¶æå¹¶å¤çå¤ç¨æ·å¹¶åè¯·æ±ã

### 2. é»è¾å±
- **å¤§è¯­è¨æ¨¡åï¼LLMï¼å¼æ**ï¼ä¸éä¹åé®ï¼Qwenï¼ç­æ¨¡åå¯¹æ¥ï¼çæå¯¹è¯å¼åå¤ã
- **è¯­é³åæï¼TTSï¼å¼æ**ï¼æ¨¡ååç³»ç»ï¼æ¯æå¤ç§æå¡åï¼EdgeTTSãGPT-SoVITSç­ï¼ï¼å®ç°ææ¬å°è¯­é³çè½¬æ¢ã
- **è¯­é³ç¹å¾æå**ï¼æåè§è§åå½¢åæ­¥æéçå£°å­¦ç¹å¾ï¼å¦æ¢å°é¢è°±å¾ï¼ã

### 3. æ¸²æå±
- **æ¨¡åæ¨ç**ï¼åºäºæ·±åº¦å­¦ä¹ æ¨¡åï¼å¦Wav2LipãMuseTalkï¼ï¼æ ¹æ®é³é¢ç¹å¾çæåå½¢åæ­¥çè§é¢å¸§ã
- **åå¤ç**ï¼å°çæçå´é¨åºåå¹³æ»å å ååå§é«æ¸èæå½¢è±¡è§é¢ä¸ã

### 4. æµåªä½å±
- **ä¼ è¾åè®®**ï¼
    - **WebRTC**ï¼ä½å»¶è¿çæµè§å¨ç«¯æµåªä½ä¼ è¾åè®®ã
    - **RTMP**ï¼éç¨äºYouTubeãåå©åå©ç­å¹³å°çæ åæµåªä½åè®®ã
    - **èææåå¤´**ï¼åè®¸å°è¾åºåå®¹ä½ä¸ºç³»ç»æåå¤´ä½¿ç¨ã

### 5. æä»¶ç³»ç»
- **æ³¨åä¸­å¿**ï¼éç¨å»ä¸­å¿åçæ³¨åæºå¶ï¼[registry.py](./registry.py)ï¼ï¼å¼åèå¯è½»æ¾æ°å¢è¯­é³åæï¼TTSï¼ãèæå½¢è±¡ï¼Avatarï¼æè¾åºï¼Outputï¼æ¨¡åã æ¬¢è¿æææ´å¥½çæ¨¡ååæå¡æ¥å¥ï¼ä¹å¯ä»¥è¿è¡åä¸åä½ã

## 4. More Usage
ä½¿ç¨è¯´æ: <https://livetalking-doc.readthedocs.io/>
  
## 5. Docker Run  
ä¸éè¦åé¢çå®è£ï¼ç´æ¥è¿è¡ã
```
docker run --gpus all -it --network=host --rm registry.cn-beijing.aliyuncs.com/codewithgpu2/lipku-metahuman-stream:2K9qaMBu8v
```
ä»£ç å¨/root/metahuman-streamï¼ågit pullæä¸ä¸ææ°ä»£ç ï¼ç¶åæ§è¡å½ä»¤åç¬¬2ã3æ­¥ 

æä¾å¦ä¸ç½ç»éå
- ucloudéå: <https://www.compshare.cn/images/4458094e-a43d-45fe-9b57-de79253befe4?referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking>  
[ucloudæç¨](https://livetalking-doc.readthedocs.io/zh-cn/latest/ucloud/ucloud.html) 
- autodléå: <https://www.codewithgpu.com/i/lipku/livetalking/base>   
[autodlæç¨](https://livetalking-doc.readthedocs.io/zh-cn/latest/autodl/README.html)ï¼autodlç±äºä¸è½å¼æ¾udpç«¯å£ï¼éè¦é¨ç½²è½¬åæå¡ï¼å¦æçä¸å°è§é¢ï¼è¯·èªè¡é¨ç½²srsæturnæå¡


## 6. æ§è½
- æ§è½ä¸»è¦è·cpuågpuç¸å³: æ¯è·¯è§é¢åç¼©éè¦æ¶ècpuï¼cpuæ§è½ä¸è§é¢åè¾¨çæ­£ç¸å³ï¼æ¯è·¯å£åæ¨çè·gpuæ§è½ç¸å³ã  
- ä¸è¯´è¯æ¶çå¹¶åæ°è·cpuç¸å³ï¼åæ¶è¯´è¯çå¹¶åæ°è·gpuç¸å³ã  
- åç«¯æ¥å¿inferfpsè¡¨ç¤ºæ¾å¡æ¨çå¸§çï¼finalfpsè¡¨ç¤ºæç»æ¨æµå¸§çãä¸¤èé½è¦å¨25ä»¥ä¸æè½å®æ¶ãå¦æinferfpså¨25ä»¥ä¸ï¼finalfpsè¾¾ä¸å°25è¡¨ç¤ºcpuæ§è½ä¸è¶³ã  
- å®æ¶æ¨çæ§è½  

æ¨¡å    |æ¾å¡åå·   |fps
:----   |:---   |:---
wav2lip256 | 3060    | 60
wav2lip256 | 3080Ti  | 120
musetalk   | 3080Ti  | 42
musetalk   | 3090    | 45
musetalk   | 4090    | 72 

wav2lip256æ¾å¡3060ä»¥ä¸å³å¯ï¼musetalkéè¦3080Tiä»¥ä¸ã 

## 7. åä¸ç
æä¾å¦ä¸æ©å±åè½ï¼éç¨äºå¯¹å¼æºé¡¹ç®å·²ç»æ¯è¾çæï¼éè¦æ©å±äº§ååè½çç¨æ·
1. é«æ¸wav2lipæ¨¡å
2. å®å¨è¯­é³äº¤äºï¼æ°å­äººåç­è¿ç¨ä¸­æ¯æéè¿å¤éè¯æèæé®ææ­æé®
3. å®æ¶åæ­¥å­å¹ï¼ç»åç«¯æä¾æ°å­äººæ¯å¥è¯æ­æ¥å¼å§ãç»æäºä»¶
4. æä¾å®æ¶é³é¢æµè¾å¥æ¥å£
5. æ°å­äººéæèæ¯ï¼å å å¨æèæ¯ 
6. avatarå®æ¶åæ¢  
7. åä¸ç»é¢éå¤ä¸ªæ°å­äººäºå¨  
8. æåå¤´é©±å¨æ°å­äººå½¢è±¡å¨ä½åè¡¨æ  
9. ä¸livekitå¯¹æ¥

æ´å¤è¯¦æ<https://livetalking-doc.readthedocs.io/zh-cn/latest/service.html>

## 8. å£°æ
åºäºæ¬é¡¹ç®å¼åå¹¶åå¸å¨Bç«ãè§é¢å·ãæé³ç­ç½ç«ä¸çè§é¢éå¸¦ä¸LiveTalkingæ°´å°åæ è¯ã

---  
å¦ææ¬é¡¹ç®å¯¹ä½ æå¸®å©ï¼å¸®å¿ç¹ä¸ªstarãä¹æ¬¢è¿æå´è¶£çæåä¸èµ·æ¥å®åè¯¥é¡¹ç®.
* ç¥è¯æç: https://t.zsxq.com/7NMyO æ²æ·é«è´¨éå¸¸è§é®é¢ãæä½³å®è·µç»éªãé®é¢è§£ç­  
* å¾®ä¿¡ï¼wxwubug (å ç¾¤è¯·å¤æ³¨)      
* Telegram: https://t.me/livetalking  
* Discord: https://discord.gg/n5jSPCT3Uf  
* Email: lipku@foxmail.com  
* å¾®ä¿¡å¬ä¼å·ï¼æ°å­äººææ¯    
<img src="./assets/qrcode-wechat.jpg" align="middle" />

