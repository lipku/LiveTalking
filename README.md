A streaming digital human based on the Ernerf model， realize audio video synchronous dialogue. It can basically achieve commercial effects.  
基于ernerf模型的流式数字人，实现音视频同步对话。基本可以达到商用效果


## Installation

Tested on Ubuntu 18.04, Pytorch 1.12 and CUDA 11.3.

### Install dependency

```bash
pip install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
```
linux cuda环境搭建可以参考这篇文章 https://zhuanlan.zhihu.com/p/674972886

安装rtmpstream库  
参照 https://github.com/lipku/python_rtmpstream


## Run

### 运行rtmpserver (srs)
```
docker run --rm -it -p 1935:1935 -p 1985:1985 -p 8080:8080 registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5
```

### 环境配置完成后，启动：

```python
python app.py
```

如果访问不了huggingface，在运行前
```
export HF_ENDPOINT=https://hf-mirror.com
```

运行成功后，用vlc访问rtmp://serverip/live/livestream

### 网页端数字人播报输入文字
安装并启动nginx
```
apt install nginx
nginx
```
修改echo.html中websocket和视频播放地址，将serverip替换成实际服务器ip
然后将echo.html和mpegts-1.7.3.min.js拷到/var/www/html下

启动数字人
```python
python app.py
```

用浏览器打开http://serverip/echo.html，在文本框输入任意文字，提交。数字人播报该段文字

## Data flow
![](/assets/dataflow.png)

## 数字人模型文件，可以替换成自己训练的模型(https://github.com/Fictionarry/ER-NeRF)

```python
.
├── data
│   ├── data_kf.json			
│   ├── pretrained
│   └── └── ngp_kg.pth

```

## TODO
- 添加chatgpt实现数字人对话
- 声音克隆
- 数字人静音时用一段视频代替

如果本项目对你有帮助，帮忙点个star。也欢迎感兴趣的朋友一起来完善该项目。  
Email: lipku@foxmail.com
