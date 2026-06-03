"""
WebRTC 命令行测试工具 — 连接 zl-talking 数字人服务，接收并保存音视频流
支持测量数字人推理响应时间（从发送文本到开始说话）

用法:
  # 默认连接本地 8010 端口，接收 15 秒
  python test_webrtc_cli.py

  # 自定义参数
  python test_webrtc_cli.py --server http://127.0.0.1:8010 --output demo.mp4 --duration 30

  # 自动发送一段文本触发 TTS 说话，自动检测说话结束
  python test_webrtc_cli.py --text "你好，今天天气真不错"
  python test_webrtc_cli.py --text "你好，今天天气怎么样" --max-wait 30

  # 只测试信令连接，不保存文件
  python test_webrtc_cli.py --no-save

依赖:
  - aiortc, aiohttp, av  (项目 requirements.txt 已包含)
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.rtcrtpreceiver import RTCRtpReceiver


# ============================================================
#  全局状态
# ============================================================
stats = {
    "video_frames": 0,
    "audio_packets": 0,
    "bytes_received": 0,
    "start_time": 0,
    "ice_state": "new",
    "connected": False,
    "sessionid": None,
}

# TTS 计时相关
tts_stats = {
    "text_sent_time": None,      # 发送文本时间
    "audio_start_time": None,    # 首次收到音频时间（开始说话）
    "audio_end_time": None,      # 音频结束时间
    "last_audio_time": None,     # 上次收到音频时间
    "audio_silence_threshold": 2.0,  # 音频静默阈值（秒），超过此时间无音频认为说话结束
}


def _print(s: str):
    """带时间戳的输出"""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
    print(f"[{ts}] {s}", flush=True)


# ============================================================
#  媒体接收器
# ============================================================
class MediaSink:
    """接收并保存 WebRTC 音视频流到文件"""

    def __init__(self, output_path: str, no_save: bool = False, record_on_text: bool = False):
        self.output_path = output_path
        self.no_save = no_save
        self.record_on_text = record_on_text  # 是否在发送文本后才开始录制
        self._container = None
        self._vstream = None
        self._astream = None
        self._start_pts = None
        self._has_error = False
        self._recording = False  # 是否正在录制
        self._record_start_time = None

    def start_recording(self):
        """开始录制"""
        if self._recording or self.no_save:
            return
        self._recording = True
        self._record_start_time = time.time()
        self._ensure_container()
        _print("🎬 开始录制视频...")

    def stop_recording(self):
        """停止录制"""
        if not self._recording:
            return
        self._recording = False
        if self._record_start_time:
            duration = time.time() - self._record_start_time
            _print(f"🛑 停止录制，录制时长: {duration:.2f}s")

    def _ensure_container(self):
        if self.no_save or self._container is not None or self._has_error:
            return
        try:
            import av
            self._container = av.open(self.output_path, mode="w")
            _print(f"输出文件: {os.path.abspath(self.output_path)}")
        except Exception as e:
            _print(f"⚠ 无法打开输出文件: {e}，将仅打印统计信息")
            self.no_save = True
            self._has_error = True

    def on_video_frame(self, frame):
        stats["video_frames"] += 1
        stats["bytes_received"] += (
            (frame.width * frame.height * 3 // 2)
            if hasattr(frame, "width")
            else 0
        )
        # 只有在开始录制后才保存视频
        if not self._recording:
            return
        self._ensure_container()
        if self._container:
            if self._vstream is None:
                import av
                import fractions
                self._vstream = self._container.add_stream("h264", rate=30)
                self._vstream.width = frame.width
                self._vstream.height = frame.height
                self._vstream.pix_fmt = "yuv420p"
                self._vstream.time_base = fractions.Fraction(1, 90000)
                _print(f"视频流初始化: {frame.width}x{frame.height} @ 30fps")
            try:
                for packet in self._vstream.encode(frame):
                    self._container.mux(packet)
            except Exception:
                pass

    def on_audio_frame(self, frame):
        global tts_stats
        stats["audio_packets"] += 1
        
        # 记录音频时间
        now = time.time()
        tts_stats["last_audio_time"] = now
        
        # 首次收到音频（开始说话）
        if tts_stats["audio_start_time"] is None:
            tts_stats["audio_start_time"] = now
            if tts_stats["text_sent_time"]:
                ttf = now - tts_stats["text_sent_time"]
                _print(f"🔊 数字人开始说话！TTF (Time To First audio): {ttf:.3f}s")
            else:
                _print(f"🔊 数字人开始说话！")
            # 开始录制
            self.start_recording()
        
        if not self._recording:
            return
            
        self._ensure_container()
        if self._container:
            if self._astream is None:
                self._astream = self._container.add_stream("aac")
                _print(f"音频流初始化...")
            try:
                for packet in self._astream.encode(frame):
                    self._container.mux(packet)
            except Exception:
                pass

    async def close(self):
        if self._container:
            try:
                if self._vstream:
                    for packet in self._vstream.encode(None):
                        self._container.mux(packet)
            except Exception:
                pass
            try:
                if self._astream:
                    for packet in self._astream.encode(None):
                        self._container.mux(packet)
            except Exception:
                pass
            try:
                self._container.close()
            except Exception:
                pass
        self._container = None

    def print_stats(self):
        elapsed = time.time() - stats["start_time"]
        _print(f"  ├ 运行时长: {elapsed:.1f}s")
        _print(f"  ├ ICE 状态: {stats['ice_state']}")
        _print(f"  ├ 视频帧数: {stats['video_frames']} ({stats['video_frames']/elapsed:.1f} fps)")
        _print(f"  ├ 音频包数: {stats['audio_packets']}")
        _print(f"  └ 总接收量: {stats['bytes_received']/1024:.0f} KB")
        if not self.no_save and self.output_path and os.path.exists(self.output_path):
            fsize = os.path.getsize(self.output_path)
            _print(f"  文件大小: {fsize/1024:.0f} KB → {self.output_path}")


def print_tts_stats():
    """打印 TTS 统计信息"""
    _print(f"\n{'='*50}")
    _print(f"  📊 数字人对话统计")
    _print(f"{'='*50}")
    
    if tts_stats["text_sent_time"]:
        _print(f"  ├ 文本发送时间: {datetime.fromtimestamp(tts_stats['text_sent_time']).strftime('%H:%M:%S.%f')[:12]}")
    
    if tts_stats["audio_start_time"]:
        _print(f"  ├ 开始说话时间: {datetime.fromtimestamp(tts_stats['audio_start_time']).strftime('%H:%M:%S.%f')[:12]}")
        if tts_stats["text_sent_time"]:
            ttf = tts_stats["audio_start_time"] - tts_stats["text_sent_time"]
            _print(f"  ├ ⏱️  首响延迟 (TTF): {ttf:.3f}s")
    
    if tts_stats["audio_end_time"] and tts_stats["audio_start_time"]:
        _print(f"  ├ 结束说话时间: {datetime.fromtimestamp(tts_stats['audio_end_time']).strftime('%H:%M:%S.%f')[:12]}")
        speech_duration = tts_stats["audio_end_time"] - tts_stats["audio_start_time"]
        _print(f"  ├ 🗣️  说话时长: {speech_duration:.3f}s")
        
        if tts_stats["text_sent_time"]:
            total_duration = tts_stats["audio_end_time"] - tts_stats["text_sent_time"]
            _print(f"  ├ ⏱️  总响应时长: {total_duration:.3f}s")
            _print(f"  ├  推理思考时间: {(tts_stats['audio_start_time'] - tts_stats['text_sent_time']):.3f}s")
            _print(f"  └ 📹 录制时长: {total_duration:.3f}s（从文本发送到说话结束）")
    elif tts_stats["audio_start_time"]:
        _print(f"  └ ⚠️ 数字人仍在说话或未检测到结束")
    else:
        _print(f"  └ ⚠️ 未检测到数字人说话")
    _print(f"{'='*50}\n")


# ============================================================
#  WebRTC 客户端
# ============================================================
async def run_client(
    server_url: str,
    output: str,
    duration: int,
    text: str = None,
    no_save: bool = False,
    max_wait: int = 60,
):
    """
    主流程:
      1. 创建 RTCPeerConnection
      2. 发送 Offer → 服务端
      3. 接收 Answer，建立连接
      4. 如果指定了 text，通过 /human 触发 TTS
      5. 检测音频开始和结束，自动停止
    """
    global stats, tts_stats
    stats["start_time"] = time.time()

    sink = MediaSink(output, no_save, record_on_text=(text is not None))
    pc = RTCPeerConnection()

    # ── 轨道回调 ────────────────────────────────────────────
    @pc.on("track")
    def on_track(track):
        _print(f"← 收到轨道: {track.kind}")

        async def _recv_loop():
            while True:
                try:
                    frame = await track.recv()
                    if track.kind == "video":
                        sink.on_video_frame(frame)
                    else:
                        sink.on_audio_frame(frame)
                except Exception:
                    break

        asyncio.create_task(_recv_loop())

    # ── 连接状态 ────────────────────────────────────────────
    @pc.on("connectionstatechange")
    def on_conn_state():
        stats["ice_state"] = pc.connectionState
        _print(f"ICE 状态变更: {pc.connectionState}")
        if pc.connectionState == "connected":
            stats["connected"] = True
            _print("✓ WebRTC 连接已建立")

    @pc.on("iceconnectionstatechange")
    def on_ice_state():
        _print(f"ICE 连接: {pc.iceConnectionState}")

    @pc.on("icegatheringstatechange")
    def on_gather():
        _print(f"ICE 收集: {pc.iceGatheringState}")

    # ── 创建 Offer ──────────────────────────────────────────
    _print(f"连接服务器: {server_url}")
    pc.addTransceiver("audio", direction="recvonly")
    pc.addTransceiver("video", direction="recvonly")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # 等待 ICE 候选收集完成（最多 3 秒）
    if pc.iceGatheringState != "complete":
        await asyncio.sleep(1)

    # ── 发送 Offer ──────────────────────────────────────────
    payload = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{server_url}/offer",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                _print(f"✗ 服务端返回 {resp.status}: {body}")
                return False
            answer = await resp.json()

    stats["sessionid"] = answer.get("sessionid")
    _print(f"← 收到 Answer, sessionid: {stats['sessionid']}")

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )

    # ── 等待连接建立 ────────────────────────────────────────
    for _ in range(30):
        if stats["connected"] or pc.connectionState in ("failed", "disconnected"):
            break
        await asyncio.sleep(0.5)

    if pc.connectionState == "failed":
        _print("✗ WebRTC 连接失败")
        await pc.close()
        return False

    if not stats["connected"]:
        _print("⚠ 连接尚未完全建立，但将继续接收...")

    # ── 发送文本触发 TTS ────────────────────────────────────
    text_sent = False
    if text and stats["sessionid"]:
        _print(f"发送 TTS 文本: \"{text}\"")
        tts_stats["text_sent_time"] = time.time()
        text_sent = True
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/human",
                json={
                    "sessionid": stats["sessionid"],
                    "text": text,
                    "type": "echo",
                    "interrupt": False,
                },
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status == 200:
                    _print("✓ TTS 请求已发送，等待数字人响应...")
                    # 从发送文本开始录制，包含推理思考时间
                    sink.start_recording()
                else:
                    body = await resp.text()
                    _print(f"✗ TTS 请求失败 ({resp.status}): {body}")

    # ── 持续接收 ────────────────────────────────────────────
    if text_sent:
        # 有文本：轮询 /is_speaking 检测说话结束，静默检测为备用
        _print(f"等待数字人说话（最多 {max_wait}s）...")
        waited = 0
        check_interval = 0.1
        speak_poll_interval = 0.5  # 每 500ms 轮询 /is_speaking
        was_speaking = False  # 记录上一次说话状态
        
        async with aiohttp.ClientSession() as http_session:
            while waited < max_wait:
                await asyncio.sleep(check_interval)
                waited += check_interval
                
                if pc.connectionState in ("failed", "closed"):
                    _print("连接已断开，提前退出")
                    break
                
                # 每 speak_poll_interval 秒轮询 /is_speaking
                if int(waited * 100) % int(speak_poll_interval * 100) < 10:
                    try:
                        async with http_session.post(
                            f"{server_url}/is_speaking",
                            json={"sessionid": stats["sessionid"]},
                            headers={"Content-Type": "application/json"},
                            timeout=aiohttp.ClientTimeout(total=2),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                is_speaking = data.get("data", False)
                                
                                if is_speaking:
                                    was_speaking = True
                                
                                # speaking: true → false，说话结束，等待10秒后停止录制
                                if was_speaking and not is_speaking:
                                    _print(f"✓ 数字人说话结束（/is_speaking = false），等待10秒后停止录制...")
                                    tts_stats["audio_end_time"] = time.time()
                                    await asyncio.sleep(10)
                                    sink.stop_recording()
                                    break
                    except:
                        pass
                
                # 静默检测（备用方案）
                if tts_stats["audio_start_time"] is not None:
                    now = time.time()
                    silence = now - tts_stats["last_audio_time"]
                    if silence >= tts_stats["audio_silence_threshold"]:
                        if sink._recording:
                            tts_stats["audio_end_time"] = tts_stats["last_audio_time"]
                            _print(f"✓ 检测到数字人说话结束（静默 {silence:.1f}s），等待10秒后停止录制...")
                            await asyncio.sleep(10)
                            sink.stop_recording()
                        break
                
                # 每 5 秒打印一次状态
                if int(waited) % 5 == 0 and waited % 1 < check_interval:
                    if tts_stats["audio_start_time"] is None:
                        _print(f"  等待中... 已等待 {waited:.1f}s，尚未开始说话")
                    else:
                        speech_duration = time.time() - tts_stats["audio_start_time"]
                        _print(f"  说话中... 已说话 {speech_duration:.1f}s")
            else:
                _print(f"⚠ 达到最大等待时间 {max_wait}s")
                if tts_stats["last_audio_time"]:
                    tts_stats["audio_end_time"] = tts_stats["last_audio_time"]
                    sink.stop_recording()
    else:
        # 无文本：按固定时长接收
        _print(f"接收中，将持续 {duration} 秒...")
        check_interval = min(5, max(1, duration // 5))
        for remaining in range(duration, 0, -check_interval):
            await asyncio.sleep(check_interval)
            if pc.connectionState in ("failed", "closed"):
                _print("连接已断开，提前退出")
                break
            fps = stats["video_frames"] / max(1, time.time() - stats["start_time"])
            left = max(0, remaining - check_interval)
            _print(
                f"  运行中 ... ICE={pc.connectionState} "
                f"video={stats['video_frames']}帧({fps:.1f}fps) "
                f"audio={stats['audio_packets']}包  剩余{left}s"
            )

    # ── 关闭 ────────────────────────────────────────────────
    _print("关闭连接...")
    await pc.close()
    await sink.close()

    _print(f"\n{'='*40}")
    _print(f"  测试完成")
    _print(f"{'='*40}")
    sink.print_stats()
    
    # 打印 TTS 统计
    if text_sent:
        print_tts_stats()

    return stats["video_frames"] > 0


# ============================================================
#  入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="zl-talking WebRTC 命令行测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                                        连接 localhost:8010，接收 15s
  %(prog)s --server http://10.0.0.1:8010          自定义服务地址
  %(prog)s --text "你好"                          发送文本，自动检测说话结束
  %(prog)s --text "你好" --max-wait 30            最多等待30秒
  %(prog)s --output test.mp4 --duration 30        输出到 test.mp4，时长 30s
  %(prog)s --no-save                              只打印统计，不保存文件
  %(prog)s --verbose                              打印详细 SDP/ICE 日志
        """,
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8010",
        help="服务端地址 (默认: http://127.0.0.1:8010)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径 (默认: webrtc_<时间戳>.mp4)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="接收时长(秒)，仅在未指定 --text 时有效 (默认: 15)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="发送文本触发 TTS 说话，自动检测说话开始和结束",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=60,
        help="发送文本后最大等待时间(秒) (默认: 60)",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=2.0,
        help="检测说话结束的静默阈值(秒) (默认: 2.0)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存文件，只打印统计信息",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细日志 (包括 aiortc 调试信息)",
    )
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("aiortc").setLevel(logging.DEBUG)

    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join("data", "results", f"webrtc_{ts}.mp4")

    # 设置静默阈值
    tts_stats["audio_silence_threshold"] = args.silence

    _print(f"zl-talking WebRTC CLI 测试工具")
    _print(f"  ├ 服务端: {args.server}")
    _print(f"  ├ 输出文件: {args.output}")
    if args.text:
        _print(f"  ├ 触发文本: \"{args.text}\"")
        _print(f"  ├ 最大等待: {args.max_wait}s")
        _print(f"  ├ 静默阈值: {args.silence}s")
    else:
        _print(f"  ├ 固定时长: {args.duration}s")
    _print(f"  └ 保存文件: {'否' if args.no_save else '是'}")
    _print("")

    success = asyncio.run(
        run_client(
            server_url=args.server.rstrip("/"),
            output=args.output,
            duration=args.duration,
            text=args.text,
            no_save=args.no_save,
            max_wait=args.max_wait,
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
