1、利用vllm可以显著推理加速大模型

conda create -n vllm python=3.10
conda activate vllm
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

2、启动推理
python -m vllm.entrypoints.openai.api_server --tensor-parallel-size=1  --trust-remote-code --max-model-len 1024 --model THUDM/chatglm3-6b
指定ip和端口：--host 127.0.0.1 --port 8101

python -m vllm.entrypoints.openai.api_server --port 8101 --tensor-parallel-size=1  --trust-remote-code --max-model-len 1024 --model THUDM/chatglm3-6b

CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
--model="/data/mnt/ShareFolder/common_models/Ziya-Reader-13B-v1.0" \
--max-model-len=8192 \
--tensor-parallel-size=2 \
--trust-remote-code \
--port=8101


3、测试
curl http://127.0.0.1:8101/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "THUDM/chatglm3-6b",
        "prompt": "请用20字内回复我,你今年多大了",
        "max_tokens": 20,
        "temperature": 0
    }'

多轮对话
curl -X POST "http://127.0.0.1:8101/v1/completions" \
-H "Content-Type: application/json" \
-d "{\"model\": \"THUDM/chatglm3-6b\",\"prompt\": \"你叫什么名字\", \"history\": [{\"role\": \"user\", \"content\": \"你出生在哪里.\"}, {\"role\": \"assistant\", \"content\": \"出生在北京\"}]}"

多轮对话
curl -X POST "http://127.0.0.1:8101/v1/chat/completions" \
-H "Content-Type: application/json" \
-d "{\"model\": \"THUDM/chatglm3-6b\", \"messages\": [{\"role\": \"system\", \"content\": \"You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.\"}, {\"role\": \"user\", \"content\": \"你好，给我讲一个故事，大概100字\"}], \"stream\": false, \"max_tokens\": 100, \"temperature\": 0.8, \"top_p\": 0.8}"


4、启动前端访问
docker run -d \
--network=host \
--name nginx2 --restart=always \
-v $PWD/nginx/conf/nginx.conf:/etc/nginx/nginx.conf \
-v $PWD/nginx/html:/usr/share/nginx/html \
-v $PWD/nginx/logs:/var/log/nginx \
--privileged=true \
--restart=always \
nginx


参考文档：https://docs.vllm.ai/en/latest/