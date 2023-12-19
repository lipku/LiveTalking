# 虚拟人说话头生成(照片虚拟人实时驱动)
![](/img/example.gif)
# Get Started

## Installation

Tested on Ubuntu 22.04, Pytorch 1.12 and CUDA 11.6，or  Pytorch 1.12 and CUDA 11.3

```python
git clone https://github.com/waityousea/xuniren.git
cd xuniren
```

### Install dependency

```python
# for ubuntu, portaudio is needed for pyaudio to work.
sudo apt install portaudio19-dev

pip install -r requirements.txt
or
## environment.yml中的pytorch使用的1.12和cuda 11.3
conda env create -f environment.yml 
## install pytorch3d
#ubuntu/mac
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

**windows安装pytorch3d**

- gcc & g++ ≥ 4.9

在windows中，需要安装gcc编译器，可以根据需求自行安装，例如采用MinGW

以下安装步骤来自于[pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)官方, 可以根据需求进行选择。

```python
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

对于 CUB 构建时间依赖项，仅当您的 CUDA 早于 11.7 时才需要，如果您使用的是 conda，则可以继续

```
conda install -c bottler nvidiacub
```

```
# Demos and examples
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python

# Tests/Linting
pip install black usort flake8 flake8-bugbear flake8-comprehensions
```

任何必要的补丁后，你可以去“x64 Native Tools Command Prompt for VS 2019”编译安装

```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
python setup.py install
```

### Build extension 

By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime. However, this may be inconvenient sometimes. Therefore, we also provide the `setup.py` to build each extension:

```
# install all extension modules
# notice: 该模块必须安装。
# 在windows下，建议采用vs2019的x64 Native Tools Command Prompt for VS 2019命令窗口安装
bash scripts/install_ext.sh
```

### **start(独立运行)**

环境配置完成后，启动虚拟人生成器：

```python
python app.py
```
### **start（对接fay，在ubuntu 20下完成测试）**
环境配置完成后，启动fay对接脚本
```python
python fay_connect.py
```
![](img/weplay.png)

扫码支助开源开发工作，凭支付单号入qq交流群



接口的输入与输出信息 [Websoket.md](https://github.com/waityousea/xuniren/blob/main/WebSocket.md)

虚拟人生成的核心文件

```python
## 注意，核心文件需要单独训练
.
├── data
│   ├── kf.json			
│   ├── pretrained
│   └── └── ngp_kg.pth

```

### Inference Speed

在台式机RTX A4000或笔记本RTX 3080ti的显卡（显存16G）上进行视频推理时，1s可以推理35~43帧，假如1s视频25帧，则1s可推理约1.5s视频。

# Acknowledgement

- The data pre-processing part is adapted from [AD-NeRF](https://github.com/YudongGuo/AD-NeRF).
- The NeRF framework is based on [torch-ngp](https://github.com/ashawkey/torch-ngp).
- The algorithm core come from  [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF).
- Usage example [Fay](https://github.com/TheRamU/Fay).

学术交流可发邮件到邮箱：waityousea@126.com
