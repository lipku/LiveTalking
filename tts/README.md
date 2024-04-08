一、采用gpt-sovits方案，bert-sovits适合长音频训练，gpt-sovits运行短音频快速推理

1、启动
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml 





2、接口测试
http://127.0.0.1:9880/tts?text=我是一个粉刷匠，粉刷本领强。我要把那新房子，刷得更漂亮。刷了房顶又刷墙，刷子像飞一样。哎呀我的小鼻子，变呀变了样。&text_lang=zh&ref_audio_path=mengpai.wav&prompt_lang=zh&prompt_text=呜哇好生气啊！不要把我跟一斗相提并论！&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true

http://127.0.0.1:9880/tts?text=这是一段测试文本，旨在通过多种语言风格和复杂性的内容来全面检验文本到语音系统的性能。接下来，我们会探索各种主题和语言结构，包括文学引用、技术性描述、日常会话以及诗歌等。首先，让我们从一段简单的描述性文本开始：“在一个阳光明媚的下午，一位年轻的旅者站在山顶上，眺望着下方那宽广而繁忙的城市。他的心中充满了对未来的憧憬和对旅途的期待。”这段文本测试了系统对自然景观描写的处理能力和情感表达的细腻程度。&stream=true
