import requests
import pyaudio

# 流式传输音频的URL，你可以自由改成Post
stream_url = 'http://127.0.0.1:5000/tts?text=这是一段测试文本，旨在通过多种语言风格和复杂性的内容来全面检验文本到语音系统的性能。接下来，我们会探索各种主题和语言结构，包括文学引用、技术性描述、日常会话以及诗歌等。首先，让我们从一段简单的描述性文本开始：“在一个阳光明媚的下午，一位年轻的旅者站在山顶上，眺望着下方那宽广而繁忙的城市。他的心中充满了对未来的憧憬和对旅途的期待。”这段文本测试了系统对自然景观描写的处理能力和情感表达的细腻程度。&stream=true'

# 初始化pyaudio
p = pyaudio.PyAudio()

# 打开音频流
stream = p.open(format=p.get_format_from_width(2),
                channels=1,
                rate=32000,
                output=True)

# 使用requests获取音频流，你可以自由改成Post
response = requests.get(stream_url, stream=True)

# 读取数据块并播放
for data in response.iter_content(chunk_size=1024):
    stream.write(data)

# 停止和关闭流
stream.stop_stream()
stream.close()

# 终止pyaudio
p.terminate()
