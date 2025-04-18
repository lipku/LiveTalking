Real-time interactive streaming digital human enables synchronous audio and video dialogue. It can basically achieve commercial effects.

[Effect of wav2lip](https://www.bilibili.com/video/BV1scwBeyELA/) | [Effect of ernerf](https://www.bilibili.com/video/BV1G1421z73r/) |  [Effect of musetalk](https://www.bilibili.com/video/BV1gm421N7vQ/)  

## News
- December 8, 2024: Improved multi-concurrency, and the video memory does not increase with the number of concurrent connections.
- December 21, 2024: Added model warm-up for wav2lip and musetalk to solve the problem of stuttering during the first inference. Thanks to [@heimaojinzhangyz](https://github.com/heimaojinzhangyz)
- December 28, 2024: Added the digital human model Ultralight-Digital-Human. Thanks to [@lijihua2017](https://github.com/lijihua2017)
- February 7, 2025: Added fish-speech tts
- February 21, 2025: Added the open-source model wav2lip256. Thanks to @不蠢不蠢
- March 2, 2025: Added Tencent's speech synthesis service
- March 16, 2025: Supports mac gpu inference. Thanks to [@GcsSloop](https://github.com/GcsSloop) 

## Features
1. Supports multiple digital human models: ernerf, musetalk, wav2lip, Ultralight-Digital-Human
2. Supports voice cloning
3. Supports interrupting the digital human while it is speaking
4. Supports full-body video stitching
5. Supports rtmp and webrtc
6. Supports video arrangement: Play custom videos when not speaking
7. Supports multi-concurrency

## 1. Installation

Tested on Ubuntu 20.04, Python 3.10, Pytorch 1.12 and CUDA 11.3

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
# If the cuda version is not 11.3 (confirm the version by running nvidia-smi), install the corresponding version of pytorch according to <https://pytorch.org/get-started/previous-versions/> 
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
# If you need to train the ernerf model, install the following libraries
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# pip install tensorflow-gpu==2.8.0
# pip install --upgrade "protobuf<=3.20.1"
``` 
Common installation issues [FAQ](https://livetalking-doc.readthedocs.io/en/latest/faq.html)  
For setting up the linux cuda environment, you can refer to this article https://zhuanlan.zhihu.com/p/674972886


## 2. Quick Start
- Download the models  
Quark Cloud Disk <https://pan.quark.cn/s/83a750323ef0>    
Google Drive <https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing>  
Copy wav2lip256.pth to the models folder of this project and rename it to wav2lip.pth;  
Extract wav2lip256_avatar1.tar.gz and copy the entire folder to the data/avatars folder of this project.
- Run  
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1  
Open http://serverip:8010/webrtcapi.html in a browser. First click'start' to play the digital human video; then enter any text in the text box and submit it. The digital human will broadcast this text.  
<font color=red>The server side needs to open ports tcp:8010; udp:1-65536</font>  
If you need to purchase a high-definition wav2lip model for commercial use, [Link](https://livetalking-doc.readthedocs.io/zh-cn/latest/service.html#wav2lip).  

- Quick experience  
<https://www.compshare.cn/images-detail?ImageID=compshareImage-18tpjhhxoq3j&referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking1.3> Create an instance with this image to run it.

If you can't access huggingface, before running
```
export HF_ENDPOINT=https://hf-mirror.com
``` 


## 3. More Usage
Usage instructions: <https://livetalking-doc.readthedocs.io/en/latest>
  
## 4. Docker Run  
No need for the previous installation, just run directly.
```
docker run --gpus all -it --network=host --rm registry.cn-beijing.aliyuncs.com/codewithgpu2/lipku-metahuman-stream:2K9qaMBu8v
```
The code is in /root/metahuman-stream. First, git pull to get the latest code, and then execute the commands as in steps 2 and 3. 

The following images are provided:
- autodl image: <https://www.codewithgpu.com/i/lipku/metahuman-stream/base>   
[autodl Tutorial](https://livetalking-doc.readthedocs.io/en/latest/autodl/README.html)
- ucloud image: <https://www.compshare.cn/images-detail?ImageID=compshareImage-18tpjhhxoq3j&referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_livetalking1.3>  
Any port can be opened, and there is no need to deploy an srs service additionally.  
[ucloud Tutorial](https://livetalking-doc.readthedocs.io/en/latest/ucloud/ucloud.html) 


## 5. TODO
- [x] Added chatgpt to enable digital human dialogue
- [x] Voice cloning
- [x] Replace the digital human with a video when it is silent
- [x] MuseTalk
- [x] Wav2Lip
- [x] Ultralight-Digital-Human

---
If this project is helpful to you, please give it a star. Friends who are interested are also welcome to join in and improve this project together.
* Knowledge Planet: https://t.zsxq.com/7NMyO, where high-quality common problems, best practice experiences, and problem solutions are accumulated.
* WeChat Official Account: Digital Human Technology  
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/l3ZibgueFiaeyfaiaLZGuMGQXnhLWxibpJUS2gfs8Dje6JuMY8zu2tVyU9n8Zx1yaNncvKHBMibX0ocehoITy5qQEZg/640?wxfrom=12&tp=wxpic&usePicPrefetch=1&wx_fmt=jpeg&amp;from=appmsg) 