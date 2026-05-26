#如果cuda版本不为12.4(运行nvidia-smi确认版本)，根据<https://pytorch.org/get-started/previous-versions/>安装对应版本的pytorch 
#conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt


# 语音采用 edge-tts
# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
# conda init powershell
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
conda activate zl-talking
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 --use_onnx
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 --use_onnx --onnx_model_path ./models/onnx/wav2lip.onnx













# 语音采用 豆包
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 --tts doubao --REF_FILE zh_female_yingyujiaoxue_uranus_bigtts

# 语音采用GPT-SoVITS
# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
# conda init powershell
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 --tts gpt-sovits --TTS_SERVER "http://127.0.0.1:9880" --REF_FILE "./Keira.wav"


# 采用其它形象和模型
# 1.基于 video 创建一个形象
python wav2lip/genavatar.py --video_path .\test_data\videos\1_2.mp4 --avatar_id 1_2 --img_size 256 --face_det_batch_size 2
python wav2lip/genavatar.py --video_path .\test_data\videos\1_2_2.mp4 --avatar_id 1_2_2 --img_size 256 --face_det_batch_size 2
python wav2lip/genavatar.py --video_path .\test_data\videos\1_2_3.mp4 --avatar_id 1_2_3 --img_size 256 --face_det_batch_size 2
python wav2lip/genavatar.py --video_path .\test_data\videos\2_1.mp4 --avatar_id 2_1 --img_size 256 --face_det_batch_size 2

# 1.1基于 png 创建一个形象
# python wav2lip/genavatar.py --video_path .\test_data\videos\00000000.png --avatar_id 1_2_1 --img_size 256 --face_det_batch_size 2
# 2.从 result/data 复制形象 1_2 到 data/avaters 下面，然后再执行以下命令
python app.py --transport webrtc --model wav2lip --avatar_id 1_2
python app.py --transport webrtc --model wav2lip --avatar_id 1_2_1
python app.py --transport webrtc --model wav2lip --avatar_id 1_2_2



python wav2lip/genavatar.py --video_path .\test_data\videos\1_2_4.mp4 --avatar_id 1_2_4 --img_size 256 --face_det_batch_size 2

cd F:\Projects\AI-Human\LiveTalking
conda init
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
python app.py --transport webrtc --model wav2lip --avatar_id 1_2_4



# 语音采用豆包
export DOUBAO_APPID='1055299334'
export DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'


# 1_2_4
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model wav2lip --avatar_id 1_2_4 --tts doubao --REF_FILE zh_female_roumeinvyou_emo_v2_mars_bigtts


# 2_1
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model wav2lip --avatar_id 2_1 --tts doubao --REF_FILE zh_female_roumeinvyou_emo_v2_mars_bigtts --batch_size 2




# 模型用musetalk
conda install ffmpeg
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0"

$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avatar1
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avatar1 --tts doubao --REF_FILE zh_female_roumeinvyou_emo_v2_mars_bigtts



# 提高GPU利用率
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avatar1 --tts doubao --REF_FILE zh_female_roumeinvyou_emo_v2_mars_bigtts --batch_size 10

python genavatar_musetalk.py --avatar_id 2_1 --file F:\\Projects\\AI-Human\\LiveTalking\\test_data\\videos\\2_1.mp4 --face_det_batch_size 2






# 用新的形象
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id avator_1 --tts doubao --REF_FILE zh_female_roumeinvyou_emo_v2_mars_bigtts --batch_size 10
# 音色匹配
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id avator_1 --tts doubao --REF_FILE zh_female_mizai_saturn_bigtts --batch_size 10
# 音色匹配(原人物)
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id avator_1 --tts doubaoclone --REF_FILE S_OvUXHN3J1 --batch_size 10
# 音色匹配（原人物坐姿）
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_2_1 --tts doubaoclone --REF_FILE S_OvUXHN3J1 --batch_size 10
# 音色匹配（720p）
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_2_3 --tts doubaoclone --REF_FILE S_OvUXHN3J1 --batch_size 10
# 音色匹配（720p V2）
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_2_6 --tts doubaoclone --REF_FILE S_OvUXHN3J1 --batch_size 10
# 音色匹配（720p V3）
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_2_7 --tts doubaoclone --REF_FILE S_OvUXHN3J1 --batch_size 10
# 形象优化（V4）
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_2_8 --tts doubaoclone --REF_FILE S_OvUXHN3J1 --batch_size 10
# 前端启动
python -m http.server 8000
访问：http://localhost:8000/web/webrtcapi-diy.html



# 录屏版本（720p）
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_2_9 --tts doubao --REF_FILE ICL_zh_female_qingyingduoduo_cs_tob --batch_size 10



# 录屏版本V2（720p）
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_2_10 --tts doubao --REF_FILE ICL_zh_female_qingyingduoduo_cs_tob --batch_size 10


# 卡通版本（720）
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_3_3 --tts doubao --REF_FILE ICL_zh_female_qingyingduoduo_cs_tob --batch_size 10



python wav2lip/genavatar.py --video_path .\test_data\videos\3_4.mp4 --avatar_id 3_4 --img_size 256 --face_det_batch_size 2





# 银行数字人V1
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_bank_1 --tts doubao --REF_FILE zh_female_yingyujiaoxue_uranus_bigtts --batch_size 10


# 银行数字人V2
## 静默状态
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
ffmpeg -i .\test_data\videos\musetalk_avater_bank_2_silent.mp4 -vf fps=25 -qmin 1 -q:v 1 -start_number 0 data/customvideo/image/%08d.png
ffmpeg -i .\test_data\videos\musetalk_avater_bank_2_silent.mp4 -vn -acodec pcm_s16le -ac 1 -ar 16000 data/customvideo/audio.wav
## 启动数字人
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_bank_2 --tts doubao --REF_FILE zh_female_yingyujiaoxue_uranus_bigtts --batch_size 10 --customvideo_config data/custom_config.json


# 银行数字人V3
## 静默状态
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
ffmpeg -i .\test_data\videos\musetalk_avater_bank_3_silent.mp4 -vf fps=25 -qmin 1 -q:v 1 -start_number 0 data/customvideo/image/%08d.png
ffmpeg -i .\test_data\videos\musetalk_avater_bank_3_silent.mp4 -vn -acodec pcm_s16le -ac 1 -ar 16000 data/customvideo/audio.wav
## 启动数字人(musetalk)
conda activate D:\zg117\C\Users\zg117\.conda\envs\livetalking_new
$env:DOUBAO_APPID='1055299334'
$env:DOUBAO_TOKEN='fOHuq4R4dirMYiOruCU3Ek9q75zV0KVW'
python app.py --transport webrtc --model musetalk --avatar_id musetalk_avater_bank_3 --tts doubao --REF_FILE zh_female_yingyujiaoxue_uranus_bigtts --batch_size 10 --customvideo_config data/custom_config.json
## wav2lip数字人
python wav2lip/genavatar.py --video_path .\test_data\videos\wav2lip_aveter_bank_2.mp4 --avatar_id wav2lip_aveter_bank_2 --img_size 256 --face_det_batch_size 2
## 启动数字人(wav2lip)
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip_aveter_bank_2 --tts doubao --REF_FILE zh_female_yingyujiaoxue_uranus_bigtts --batch_size 2