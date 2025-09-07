 [English](./README-EN.md) | 中文版  
 <p align="center">
 <img src="./assets/LiveTalking-logo.jpg" align="middle" width = "600"/>
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

 实时交互流式数字人，实现音视频同步对话。基本可以达到商用效果  
[wav2lip效果](https://www.bilibili.com/video/BV1scwBeyELA/) | [ernerf效果](https://www.bilibili.com/video/BV1G1421z73r/) | [musetalk效果](https://www.bilibili.com/video/BV1gm421N7vQ/)  
国内镜像地址:<https://gitee.com/lipku/LiveTalking> 

## 为避免与3d数字人混淆，原项目metahuman-stream改名为livetalking，原有链接地址继续可用

## News
- 2024.12.8 完善多并发，显存不随并发数增加
- 2024.12.21 添加wav2lip、musetalk模型预热，解决第一次推理卡顿问题。感谢[@heimaojinzhangyz](https://github.com/heimaojinzhangyz)
- 2024.12.28 添加数字人模型Ultralight-Digital-Human。 感谢[@lijihua2017](https://github.com/lijihua2017)
- 2025.2.7 添加fish-speech tts
- 2025.2.21 添加wav2lip256开源模型 感谢@不蠢不蠢
- 2025.3.2 添加腾讯语音合成服务
- 2025.3.16 支持mac gpu推理，感谢[@GcsSloop](https://github.com/GcsSloop) 
- 2025.5.1 精简运行参数，ernerf模型移至git分支ernerf-rtmp
- 2025.6.7 添加虚拟摄像头输出
- 2025.7.5 添加豆包语音合成, 感谢[@ELK-milu](https://github.com/ELK-milu)
- 2025.7.26 支持musetalk v1.5版本

## Features
1. 支持多种数字人模型: ernerf、musetalk、wav2lip、Ultralight-Digital-Human
2. 支持声音克隆
3. 支持数字人说话被打断
4. 支持全身视频拼接
5. 支持webrtc、虚拟摄像头输出
6. 支持动作编排：不说话时播放自定义视频
7. 支持多并发

## 1. Installation

Tested on Ubuntu 24.04, Python3.10, Pytorch 2.5.0 and CUDA 12.4

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
#如果cuda版本不为12.4(运行nvidia-smi确认版本)，根据<https://pytorch.org/get-started/previous-versions/>安装对应版本的pytorch 
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
``` 
安装常见问题[FAQ](https://livetalking-doc.readthedocs.io/zh-cn/latest/faq.html)  
linux cuda环境搭建可以参考这篇文章 <https://zhuanlan.zhihu.com/p/674972886>  
视频连不上解决方法 <https://mp.weixin.qq.com/s/MVUkxxhV2cgMMHalphr2cg>


## 2. Quick Start
- 下载模型  
夸克云盘<https://pan.quark.cn/s/83a750323ef0>    
GoogleDriver <https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing>  
将wav2lip256.pth拷到本项目的models下, 重命名为wav2lip.pth;  
将wav2lip256_avatar1.tar.gz解压后整个文件夹拷到本项目的data/avatars下
- 运行  
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1  
<font color=red>服务端需要开放端口 tcp:8010; udp:1-65536 </font>  
客户端可以选用以下两种方式:  
(1)用浏览器打开http://serverip:8010/webrtcapi.html , 先点‘start',播放数字人视频；然后在文本框输入任意文字，提交。数字人播报该段文字  
(2)用客户端方式, 下载地址<https://pan.quark.cn/s/d7192d8ac19b>   

- 快速体验  
<https://www.compshare.cn/images/4458094e-a43d-45fe-9b57-de79253befe4?referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking> 用该镜像创建实例即可运行成功

如果访问不了huggingface，在运行前
```
export HF_ENDPOINT=https://hf-mirror.com
``` 


## 3. More Usage
使用说明: <https://livetalking-doc.readthedocs.io/>
  
## 4. Docker Run  
不需要前面的安装，直接运行。
```
docker run --gpus all -it --network=host --rm registry.cn-zhangjiakou.aliyuncs.com/codewithgpu3/lipku-livetalking:toza2irpHZ
```
代码在/root/livetalking，先git pull拉一下最新代码，然后执行命令同第2、3步 

提供如下镜像
- autodl镜像: <https://www.codewithgpu.com/i/lipku/livetalking/base>   
[autodl教程](https://livetalking-doc.readthedocs.io/en/latest/autodl/README.html)
- ucloud镜像: <https://www.compshare.cn/images/4458094e-a43d-45fe-9b57-de79253befe4?referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_GitHub_livetalking>  
可以开放任意端口，不需要另外部署srs服务.  
[ucloud教程](https://livetalking-doc.readthedocs.io/en/latest/ucloud/ucloud.html) 


## 5. 性能
- 性能主要跟cpu和gpu相关，每路视频压缩需要消耗cpu，cpu性能与视频分辨率正相关；每路口型推理跟gpu性能相关。  
- 不说话时的并发数跟cpu相关，同时说话的并发数跟gpu相关。  
- 后端日志inferfps表示显卡推理帧率，finalfps表示最终推流帧率。两者都要在25以上才能实时。如果inferfps在25以上，finalfps达不到25表示cpu性能不足。  
- 实时推理性能  

模型    |显卡型号   |fps
:----   |:---   |:---
wav2lip256 | 3060    | 60
wav2lip256 | 3080Ti  | 120
musetalk   | 3080Ti  | 42
musetalk   | 3090    | 45
musetalk   | 4090    | 72 

wav2lip256显卡3060以上即可，musetalk需要3080Ti以上。 

## 6. 商业版
提供如下扩展功能，适用于对开源项目已经比较熟悉，需要扩展产品功能的用户
1. 高清wav2lip模型
2. 完全语音交互，数字人回答过程中支持通过唤醒词或者按钮打断提问
3. 实时同步字幕，给前端提供数字人每句话播报开始、结束事件
4. 每个连接可以指定对应avatar和音色，avatar图片加载加速
5. 支持不限时长的数字人形象avatar
6. 提供实时音频流输入接口
7. 数字人透明背景，叠加动态背景 
8. avatar实时切换  
9. python客户端  

更多详情<https://livetalking-doc.readthedocs.io/zh-cn/latest/service.html#wav2lip>

## 7. 声明
基于本项目开发并发布在B站、视频号、抖音等网站上的视频需带上LiveTalking水印和标识。

---
如果本项目对你有帮助，帮忙点个star。也欢迎感兴趣的朋友一起来完善该项目.
* 知识星球: https://t.zsxq.com/7NMyO 沉淀高质量常见问题、最佳实践经验、问题解答  
* 微信公众号：数字人技术  
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/l3ZibgueFiaeyfaiaLZGuMGQXnhLWxibpJUS2gfs8Dje6JuMY8zu2tVyU9n8Zx1yaNncvKHBMibX0ocehoITy5qQEZg/640?wxfrom=12&tp=wxpic&usePicPrefetch=1&wx_fmt=jpeg&amp;from=appmsg)  

