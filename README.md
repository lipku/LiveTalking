A streaming digital human based on the Ernerf model， realize audio video synchronous dialogue. It can basically achieve commercial effects.  
基于ernerf模型的流式数字人，实现音视频同步对话。基本可以达到商用效果

[![Watch the video]](/assets/demo.mp4)

## Features
1. 支持声音克隆
2. 支持大模型对话
3. 支持多种音频特征驱动：wav2vec、hubert
4. 支持全身视频拼接
5. 支持rtmp和webrtc

## 1. Installation

Tested on Ubuntu 20.04, Python3.10, Pytorch 1.12 and CUDA 11.3

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
conda activate nerfstream
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
```
linux cuda环境搭建可以参考这篇文章 https://zhuanlan.zhihu.com/p/674972886


## 2. Quick Start
默认采用webrtc推流到srs  
### 2.1 运行rtmpserver (srs)
```
export CANDIDATE='<服务器外网ip>'
docker run --rm --env CANDIDATE=$CANDIDATE \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc.conf
```

### 2.2 启动数字人：

```python
python app.py
```

如果访问不了huggingface，在运行前
```
export HF_ENDPOINT=https://hf-mirror.com
```

用浏览器打开http://serverip:8010/rtcpush.html, 在文本框输入任意文字，提交。数字人播报该段文字  
备注：服务端需要开放端口 tcp:8000,8010,1985; udp:8000

## 3. More Usage
### 3.1 使用LLM模型进行数字人对话

目前借鉴数字人对话系统[LinlyTalker](https://github.com/Kedreamix/Linly-Talker)的方式，LLM模型支持Chatgpt,Qwen和GeminiPro。需要在app.py中填入自己的api_key。    

用浏览器打开http://serverip:8010/rtcpushchat.html

### 3.2 声音克隆
可以任意选用下面两种服务，推荐用gpt-sovits
#### 3.2.1 gpt-sovits
服务部署参照[gpt-sovits](/tts/README.md)  
运行
```
python app.py --tts gpt-sovits --TTS_SERVER http://127.0.0.1:5000 --CHARACTER test --EMOTION default
```
#### 3.2.2 xtts
运行xtts服务，参照 https://github.com/coqui-ai/xtts-streaming-server
```
docker run --gpus=all -e COQUI_TOS_AGREED=1 --rm -p 9000:80 ghcr.io/coqui-ai/xtts-streaming-server:latest
```
然后运行，其中ref.wav为需要克隆的声音文件
```
python app.py --tts xtts --REF_FILE data/ref.wav --TTS_SERVER http://localhost:9000
```

### 3.3 音频特征用hubert
如果训练模型时用的hubert提取音频特征，用如下命令启动数字人
```
python app.py --asr_model facebook/hubert-large-ls960-ft 
```

### 3.4 设置背景图片
```
python app.py --bg_img bg.jpg 
```

### 3.5 全身视频拼接
#### 3.5.1 切割训练用的视频
```
ffmpeg -i fullbody.mp4 -vf crop="400:400:100:5" train.mp4 
```
用train.mp4训练模型
#### 3.5.2 提取全身图片
```
ffmpeg -i fullbody.mp4 -vf fps=25 -qmin 1 -q:v 1 -start_number 0 data/fullbody/img/%d.jpg
```
#### 3.5.2 启动数字人
```
python app.py --fullbody --fullbody_img data/fullbody/img --fullbody_offset_x 100 --fullbody_offset_y 5 --fullbody_width 580 --fullbody_height 1080 --W 400 --H 400
```
- --fullbody_width、--fullbody_height 全身视频的宽、高
- --W、--H 训练视频的宽、高  
- ernerf训练第三步torso如果训练的不好，在拼接处会有接缝。可以在上面的命令加上--torso_imgs data/xxx/torso_imgs，torso不用模型推理，直接用训练数据集里的torso图片。这种方式可能头颈处会有些人工痕迹。

### 3.6 webrtc p2p
此种模式不需要srs
```
python app.py --transport webrtc
```
用浏览器打开http://serverip:8010/webrtc.html

### 3.7 rtmp推送到srs
- 安装rtmpstream库  
参照 https://github.com/lipku/python_rtmpstream

- 启动srs
```
docker run --rm -it -p 1935:1935 -p 1985:1985 -p 8080:8080 registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5
```
- 然后运行
```python
python app.py --transport rtmp --push_url 'rtmp://localhost/live/livestream'
```
用浏览器打开http://serverip:8010/echo.html
  
## 4. Docker Run  
不需要第1步的安装，直接运行。
```
docker run --gpus all -it --network=host --rm  registry.cn-hangzhou.aliyuncs.com/lipku/nerfstream:v1.3
```
docker版本已经不是最新代码，可以作为一个空环境，把最新代码拷进去运行。

## 5. Data flow
![](/assets/dataflow.png)

## 6. 数字人模型文件
可以替换成自己训练的模型(https://github.com/Fictionarry/ER-NeRF)
```python
.
├── data
│   ├── data_kf.json
│   ├── au.csv			
│   ├── pretrained
│   └── └── ngp_kf.pth

```

## 7. 性能分析
1. 帧率  
在Tesla T4显卡上测试整体fps为18左右，如果去掉音视频编码推流，帧率在20左右。用4090显卡可以达到40多帧/秒。  
优化：新开一个线程运行音视频编码推流  
2. 延时  
整体延时3s左右  
（1）tts延时1.7s左右，目前用的edgetts，需要将每句话转完后一次性输入，可以优化tts改成流式输入  
（2）wav2vec延时0.4s，需要缓存18帧音频做计算 
（3）srs转发延时，设置srs服务器减少缓冲延时。具体配置可看 https://ossrs.net/lts/zh-cn/docs/v5/doc/low-latency

## 8. TODO
- [x] 添加chatgpt实现数字人对话
- [x] 声音克隆
- [ ] 数字人静音时用一段视频代替

如果本项目对你有帮助，帮忙点个star。也欢迎感兴趣的朋友一起来完善该项目。  
Email: lipku@foxmail.com  
微信公众号：数字人技术  
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/l3ZibgueFiaeyfaiaLZGuMGQXnhLWxibpJUS2gfs8Dje6JuMY8zu2tVyU9n8Zx1yaNncvKHBMibX0ocehoITy5qQEZg/640?wxfrom=12&tp=wxpic&usePicPrefetch=1&wx_fmt=jpeg&amp;from=appmsg)
