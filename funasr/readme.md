1、启动语言识别服务端
创建虚拟环境
conda create -n funasr
conda activate funasr
安装依赖库
pip install torch
pip install modelscope
pip install testresources
pip install websockets
pip install torchaudio
pip install FunASR
pip install pyaudio


python funasr_wss_server.py --port 10095
或者
python funasr_wss_server.py --host "0.0.0.0" --port 10197 --ngpu 0 





https://github.com/alibaba-damo-academy/FunASR
https://zhuanlan.zhihu.com/p/649935170
