1、启动语言识别服务端
创建虚拟环境
conda create -n funasr
conda activate funasr
安装依赖库

python funasr_wss_server.py --port 10095
或者
python funasr_wss_server.py --host "0.0.0.0" --port 10197 --ngpu 0 

