Real time interactive streaming digital human， realize audio video synchronous dialogue. It can basically achieve commercial effects.  
实时交互流式数字人，实现音视频同步对话。基本可以达到商用效果

[ernerf效果](https://www.bilibili.com/video/BV1PM4m1y7Q2/)  [musetalk效果](https://www.bilibili.com/video/BV1gm421N7vQ/)  [wav2lip效果](https://www.bilibili.com/video/BV1Bw4m1e74P/)

## Features
1. 支持多种数字人模型: ernerf、musetalk、wav2lip
2. 支持声音克隆
3. 支持数字人说话被打断
4. 支持全身视频拼接
5. 支持rtmp和webrtc
6. 支持视频编排：不说话时播放自定义视频

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
如果用pytorch2.1，torchvision用0.16（可以去torchvision官网根据pytorch版本找匹配的）,cudatoolkit可以不用装  
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
使用说明: <https://livetalking-doc.readthedocs.io/>
  
## 4. Docker Run  
不需要前面的安装，直接运行。
```
docker run --gpus all -it --network=host --rm registry.cn-beijing.aliyuncs.com/codewithgpu2/lipku-metahuman-stream:vjo1Y6NJ3N
```
代码在/root/metahuman-stream，先git pull拉一下最新代码，然后执行命令同第2、3步 

另外提供autodl镜像: <https://www.codewithgpu.com/i/lipku/metahuman-stream/base>   
[autodl教程](autodl/README.md)


## 5. 性能分析
1. 帧率  
在Tesla T4显卡上测试整体fps为18左右，如果去掉音视频编码推流，帧率在20左右。用4090显卡可以达到40多帧/秒。  
优化：新开一个线程运行音视频编码推流  
2. 延时  
整体延时3s左右  
（1）tts延时1.7s左右，目前用的edgetts，需要将每句话转完后一次性输入，可以优化tts改成流式输入  
（2）wav2vec延时0.4s，需要缓存18帧音频做计算 
（3）srs转发延时，设置srs服务器减少缓冲延时。具体配置可看 https://ossrs.net/lts/zh-cn/docs/v5/doc/low-latency


## 6. TODO
- [x] 添加chatgpt实现数字人对话
- [x] 声音克隆
- [x] 数字人静音时用一段视频代替
- [x] MuseTalk
- [x] Wav2Lip
- [ ] TalkingGaussian

---
如果本项目对你有帮助，帮忙点个star。也欢迎感兴趣的朋友一起来完善该项目.
* 知识星球: https://t.zsxq.com/7NMyO 沉淀高质量常见问题、最佳实践经验、问题解答  
* 微信公众号：数字人技术  
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/l3ZibgueFiaeyfaiaLZGuMGQXnhLWxibpJUS2gfs8Dje6JuMY8zu2tVyU9n8Zx1yaNncvKHBMibX0ocehoITy5qQEZg/640?wxfrom=12&tp=wxpic&usePicPrefetch=1&wx_fmt=jpeg&amp;from=appmsg)  

