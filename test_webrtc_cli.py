"""
WebRTC 命令行测试工具 — 连接 zl-talking 数字人服务，接收并保存音视频流

用法:
  # 默认连接本地 8010 端口，接收 15 秒
  python test_webrtc_cli.py

  # 自定义参数
  python test_webrtc_cli.py --server http://127.0.0.1:8010 --output demo.mp4 --duration 30

  # 自动发送一段文本触发 TTS 说话
  python test_webrtc_cli.py --text "你好，今天天气真不错" --duration 20
  python test_webrtc_cli.py --text "你好，今天天气怎么样" --duration 30

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


def _print(s: str):
    """带时间戳的输出"""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
    print(f"[{ts}] {s}", flush=True)


# ============================================================
#  媒体接收器
# ============================================================
class MediaSink:
    """接收并保存 WebRTC 音视频流到文件"""

    def __init__(self, output_path: str, no_save: bool = False):
        self.output_path = output_path
        self.no_save = no_save
        self._container = None
        self._vstream = None
        self._astream = None
        self._start_pts = None
        self._has_error = False

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
        stats["audio_packets"] += 1
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


# ============================================================
#  WebRTC 客户端
# ============================================================
async def run_client(
    server_url: str,
    output: str,
    duration: int,
    text: str = None,
    no_save: bool = False,
):
    """
    主流程:
      1. 创建 RTCPeerConnection
      2. 发送 Offer → 服务端
      3. 接收 Answer，建立连接
      4. 如果指定了 text，通过 /human 触发 TTS
      5. 接收流直到超时
    """
    global stats
    stats["start_time"] = time.time()

    sink = MediaSink(output, no_save)
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

    # ── 可选：发送文本触发 TTS ───────────────────────────────
    if text and stats["sessionid"]:
        _print(f"发送 TTS 文本: \"{text}\"")
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
                    _print("✓ TTS 请求已发送")
                else:
                    body = await resp.text()
                    _print(f"✗ TTS 请求失败 ({resp.status}): {body}")

    # ── 持续接收 ────────────────────────────────────────────
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
  %(prog)s --output test.mp4 --duration 30        输出到 test.mp4，时长 30s
  %(prog)s --text "你好"                          自动发送文本触发 TTS
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
        help="接收时长(秒) (默认: 15)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="发送文本触发 TTS 说话 (默认: 不发送)",
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

    _print(f"zl-talking WebRTC CLI 测试工具")
    _print(f"  ├ 服务端: {args.server}")
    _print(f"  ├ 输出文件: {args.output}")
    _print(f"  ├ 时长: {args.duration}s")
    _print(f"  ├ 触发文本: {args.text or '(无)'}")
    _print(f"  └ 保存文件: {'否' if args.no_save else '是'}")
    _print("")

    success = asyncio.run(
        run_client(
            server_url=args.server.rstrip("/"),
            output=args.output,
            duration=args.duration,
            text=args.text,
            no_save=args.no_save,
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
