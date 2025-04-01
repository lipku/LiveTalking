 实时交互流式数字人，实现音视频同步对话。基本可以达到商用效果

[wav2lip效果](https://www.bilibili.com/video/BV1Bw4m1e74P/) | [ernerf效果](https://www.bilibili.com/video/BV1PM4m1y7Q2/) | [musetalk效果](https://www.bilibili.com/video/BV1gm421N7vQ/)

## 为避免与3d数字人混淆，原项目metahuman-stream改名为livetalking，原有链接地址继续可用

## News
- 2024.12.8 完善多并发，显存不随并发数增加
- 2024.12.21 添加wav2lip、musetalk模型预热，解决第一次推理卡顿问题。感谢[@heimaojinzhangyz](https://github.com/heimaojinzhangyz)
- 2024.12.28 添加数字人模型Ultralight-Digital-Human。 感谢[@lijihua2017](https://github.com/lijihua2017)
- 2025.2.7 添加fish-speech tts
- 2025.2.21 添加wav2lip256开源模型 感谢@不蠢不蠢
- 2025.3.2 添加腾讯语音合成服务
- 2025.3.16 支持mac gpu推理，感谢[@GcsSloop](https://github.com/GcsSloop) 

## Features
1. 支持多种数字人模型: ernerf、musetalk、wav2lip、Ultralight-Digital-Human
2. 支持声音克隆
3. 支持数字人说话被打断
4. 支持全身视频拼接
5. 支持rtmp和webrtc
6. 支持视频编排：不说话时播放自定义视频
7. 支持多并发

## 1. Installation

Tested on Ubuntu 20.04, Python3.10, Pytorch 1.12 and CUDA 11.3

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
#如果cuda版本不为11.3(运行nvidia-smi确认版本)，根据<https://pytorch.org/get-started/previous-versions/>安装对应版本的pytorch 
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
#如果需要训练ernerf模型，安装下面的库
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# pip install tensorflow-gpu==2.8.0
# pip install --upgrade "protobuf<=3.20.1"
``` 
安装常见问题[FAQ](https://livetalking-doc.readthedocs.io/en/latest/faq.html)  
linux cuda环境搭建可以参考这篇文章 https://zhuanlan.zhihu.com/p/674972886


## 2. Quick Start
- 下载模型  
夸克云盘<https://pan.quark.cn/s/83a750323ef0>    
GoogleDriver <https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing>  
将wav2lip256.pth拷到本项目的models下, 重命名为wav2lip.pth;  
将wav2lip256_avatar1.tar.gz解压后整个文件夹拷到本项目的data/avatars下
- 运行  
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1  
用浏览器打开http://serverip:8010/webrtcapi.html , 先点‘start',播放数字人视频；然后在文本框输入任意文字，提交。数字人播报该段文字  
<font color=red>服务端需要开放端口 tcp:8010; udp:1-65536 </font>  
如果需要商用高清wav2lip模型，可以与我联系购买  

- 快速体验  
<https://www.compshare.cn/images-detail?ImageID=compshareImage-18tpjhhxoq3j&referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking1.3> 用该镜像创建实例即可运行成功

如果访问不了huggingface，在运行前
```
export HF_ENDPOINT=https://hf-mirror.com
``` 


## 3. More Usage
使用说明: <https://livetalking-doc.readthedocs.io/>
  
## 4. Docker Run  
不需要前面的安装，直接运行。
```
docker run --gpus all -it --network=host --rm registry.cn-beijing.aliyuncs.com/codewithgpu2/lipku-metahuman-stream:2K9qaMBu8v
```
代码在/root/metahuman-stream，先git pull拉一下最新代码，然后执行命令同第2、3步 

提供如下镜像
- autodl镜像: <https://www.codewithgpu.com/i/lipku/metahuman-stream/base>   
[autodl教程](https://livetalking-doc.readthedocs.io/en/latest/autodl/README.html)
- ucloud镜像: <https://www.compshare.cn/images-detail?ImageID=compshareImage-18tpjhhxoq3j&referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_livetalking1.3>  
可以开放任意端口，不需要另外部署srs服务.  
[ucloud教程](https://livetalking-doc.readthedocs.io/en/latest/ucloud/ucloud.html) 


## 5. TODO
- [x] 添加chatgpt实现数字人对话
- [x] 声音克隆
- [x] 数字人静音时用一段视频代替
- [x] MuseTalk
- [x] Wav2Lip
- [x] Ultralight-Digital-Human

---
如果本项目对你有帮助，帮忙点个star。也欢迎感兴趣的朋友一起来完善该项目.
* 知识星球: https://t.zsxq.com/7NMyO 沉淀高质量常见问题、最佳实践经验、问题解答  
* 微信公众号：数字人技术  
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/l3ZibgueFiaeyfaiaLZGuMGQXnhLWxibpJUS2gfs8Dje6JuMY8zu2tVyU9n8Zx1yaNncvKHBMibX0ocehoITy5qQEZg/640?wxfrom=12&tp=wxpic&usePicPrefetch=1&wx_fmt=jpeg&amp;from=appmsg)  

