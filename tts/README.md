一、采用gpt-sovits方案，bert-sovits适合长音频训练，gpt-sovits运行短音频快速推理
下载tts服务端代码
https://github.com/yanyuxiyangzk/GPT-SoVITS/tree/fast_inference_
api_v2.py即启动的服务端代码，也可以打开声音克隆界面进行训练，可以训练带感情语气等

1、启动
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml 


http://127.0.0.1:9880/set_sovits_weights?weights_path=SoVITS_weights/maimai_e55_s1210.pth
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_weights/maimai-e21.ckpt


2、接口测试
http://127.0.0.1:9880/set_ava?ava=maimai

http://127.0.0.1:9880/tts_ava?ava=maimai&text=我是一个粉刷匠，粉刷本领强。我要把那新房子，刷得更漂亮。刷了房顶又刷墙，刷子像飞一样。哎呀我的小鼻子，变呀变了样

http://127.0.0.1:9880/tts_ava?ava=maimai&text=我是一个粉刷匠，粉刷本领强。我要把那新房子，刷得更漂亮。刷了房顶又刷墙，刷子像飞一样。哎呀我的小鼻子，变呀变了样&streaming_mode=true

http://127.0.0.1:9880/tts_ava?ava=maimai&text=我是一个粉刷匠，粉刷本领强。我要把那新房子，刷得更漂亮。刷了房顶又刷墙，刷子像飞一样。哎呀我的小鼻子，变呀变了样。&text_lang=zh&ref_audio_path=mengpai.wav&prompt_lang=zh&prompt_text=呜哇好生气啊！不要把我跟一斗相提并论！&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true

3、使用
设置角色
http://127.0.0.1:9880/set_ava?ava=maimai
tts接口
http://127.0.0.1:9880/tts_ava?ava=maimai&text=我是一个粉刷匠，粉刷本领强。我要把那新房子，刷得更漂亮。刷了房顶又刷墙，刷子像飞一样。哎呀我的小鼻子，变呀变了样&streaming_mode=true