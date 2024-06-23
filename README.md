Real time interactive streaming digital human， realize audio video synchronous dialogue. It can basically achieve commercial effects.  
实时交互流式数字人，实现音视频同步对话。基本可以达到商用效果

[ernerf效果](https://www.bilibili.com/video/BV1PM4m1y7Q2/)  [musetalk效果](https://www.bilibili.com/video/BV1gm421N7vQ/)  [wav2lip效果](https://www.bilibili.com/video/BV1Bw4m1e74P/)

## Features
1. 支持多种数字人模型: ernerf、musetalk、wav2lip
2. 支持声音克隆
3. 支持多种音频特征驱动：wav2vec、hubert
4. 支持全身视频拼接
5. 支持rtmp和webrtc
6. 支持视频编排：不说话时播放自定义视频
7. 支持大模型对话

## 1. Installation

Tested on Ubuntu 20.04, Python3.10, Pytorch 1.12 and CUDA 11.3

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
#如果只用musetalk或者wav2lip模型，不需要安装下面的库
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
pip install --upgrade "protobuf<=3.20.1"
```
安装常见问题[FAQ](/assets/faq.md)  
linux cuda环境搭建可以参考这篇文章 https://zhuanlan.zhihu.com/p/674972886


## 2. Quick Start
默认采用ernerf模型，webrtc推流到srs  
### 2.1 运行srs
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

用浏览器打开http://serverip:8010/rtcpushapi.html, 在文本框输入任意文字，提交。数字人播报该段文字  
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
python app.py --tts gpt-sovits --TTS_SERVER http://127.0.0.1:9880 --REF_FILE data/ref.wav --REF_TEXT xxx
```
REF_TEXT为REF_FILE中语音内容，时长不宜过长

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
python app.py --bg_img bc.jpg 
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

### 3.6 不说话时用自定义视频替代
- 提取自定义视频图片
```
ffmpeg -i silence.mp4 -vf fps=25 -qmin 1 -q:v 1 -start_number 0 data/customvideo/img/%d.png
```
- 运行数字人
```
python app.py --customvideo --customvideo_img data/customvideo/img --customvideo_imgnum 100
```

### 3.7 webrtc p2p
此种模式不需要srs
```
python app.py --transport webrtc
```
服务端需要开放端口 tcp:8010; udp:50000~60000  
用浏览器打开http://serverip:8010/webrtcapi.html

### 3.8 rtmp推送到srs
- 安装rtmpstream库  
参照 https://github.com/lipku/python_rtmpstream

- 启动srs
```
docker run --rm -it -p 1935:1935 -p 1985:1985 -p 8080:8080 registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5
```
- 运行数字人
```python
python app.py --transport rtmp --push_url 'rtmp://localhost/live/livestream'
```
用浏览器打开http://serverip:8010/echoapi.html

### 3.9 模型用musetalk
暂不支持rtmp推送
- 安装依赖库
```bash
conda install ffmpeg
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0"
```
- 下载模型  
下载MuseTalk运行需要的模型，提供一个下载地址 https://caiyun.139.com/m/i?2eAjs2nXXnRgr  提取码:qdg2
解压后，将models下文件拷到本项目的models下  
下载数字人模型，链接: https://caiyun.139.com/m/i?2eAjs8optksop  提取码:3mkt, 解压后将整个文件夹拷到本项目的data/avatars下
- 运行  
python app.py --model musetalk --transport webrtc  
用浏览器打开http://serverip:8010/webrtcapi.html  
可以设置--batch_size 提高显卡利用率，设置--avatar_id 运行不同的数字人
#### 替换成自己的数字人
```bash
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
修改configs/inference/realtime.yaml，将preparation改为True
python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml
运行后将results/avatars下文件拷到本项目的data/avatars下
```

```bash
也可以试用本地目录下的 simple_musetalk.py
cd musetalk
python simple_musetalk.py --avatar_id 2  --file D:\\ok\\test.mp4
运行后将直接生成在data/avatars下
```

### 3.10 模型用wav2lip
暂不支持rtmp推送
- 下载模型  
下载wav2lip运行需要的模型，网盘地址 https://drive.uc.cn/s/551be97d7cfa4
将s3fd.pth拷到本项目wav2lip/face_detection/detection/sfd/s3fd.pth, 将wav2lip.pth拷到本项目的models下  
数字人模型文件 wav2lip_avatar1.tar.gz, 解压后将整个文件夹拷到本项目的data/avatars下
- 运行  
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip_avatar1  
用浏览器打开http://serverip:8010/webrtcapi.html  
可以设置--batch_size 提高显卡利用率，设置--avatar_id 运行不同的数字人
#### 替换成自己的数字人
```bash
cd wav2lip
python genavatar.py --video_path xxx.mp4
运行后将results/avatars下文件拷到本项目的data/avatars下
```
  
## 4. Docker Run  
不需要前面的安装，直接运行。
```
docker run --gpus all -it --network=host --rm registry.cn-beijing.aliyuncs.com/codewithgpu2/lipku-metahuman-stream:vjo1Y6NJ3N
```
代码在/root/metahuman-stream，先git pull拉一下最新代码，然后执行命令同第2、3步 

另外提供autodl镜像： 
https://www.codewithgpu.com/i/lipku/metahuman-stream/base  
[autodl教程](autodl/README.md)


## 5. 数字人模型文件
可以替换成自己训练的模型(https://github.com/Fictionarry/ER-NeRF)
```python
.
├── data
│   ├── data_kf.json
│   ├── au.csv			
│   ├── pretrained
│   └── └── ngp_kf.pth

```

## 6. 性能分析
1. 帧率  
在Tesla T4显卡上测试整体fps为18左右，如果去掉音视频编码推流，帧率在20左右。用4090显卡可以达到40多帧/秒。  
优化：新开一个线程运行音视频编码推流  
2. 延时  
整体延时3s左右  
（1）tts延时1.7s左右，目前用的edgetts，需要将每句话转完后一次性输入，可以优化tts改成流式输入  
（2）wav2vec延时0.4s，需要缓存18帧音频做计算 
（3）srs转发延时，设置srs服务器减少缓冲延时。具体配置可看 https://ossrs.net/lts/zh-cn/docs/v5/doc/low-latency

## 7. TODO
- [x] 添加chatgpt实现数字人对话
- [x] 声音克隆
- [x] 数字人静音时用一段视频代替
- [x] MuseTalk
- [x] Wav2Lip
- [ ] SyncTalk

如果本项目对你有帮助，帮忙点个star。也欢迎感兴趣的朋友一起来完善该项目。   
知识星球: https://t.zsxq.com/7NMyO 沉淀高质量常见问题、最佳实践经验、问题解答  
微信公众号：数字人技术  
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/l3ZibgueFiaeyfaiaLZGuMGQXnhLWxibpJUS2gfs8Dje6JuMY8zu2tVyU9n8Zx1yaNncvKHBMibX0ocehoITy5qQEZg/640?wxfrom=12&tp=wxpic&usePicPrefetch=1&wx_fmt=jpeg&amp;from=appmsg)  

