"""
控制虚拟摄像头中的数字人说话（交互式控制台）
使用 HTTP API 控制 Session 0
"""

import requests
import time

BASE_URL = "http://localhost:8010"

def make_avatar_speak(text: str, mode: str = "echo", sessionid: str = "0"):
    """让数字人说话

    Args:
        text: 要说的文本
        mode: 'echo' 直接说话，'chat' AI对话
        sessionid: 会话ID（虚拟摄像头固定为 '0'）
    """
    url = f"{BASE_URL}/human"
    payload = {
        "sessionid": sessionid,
        "type": mode,
        "text": text
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"[OK] 已发送文本: {text}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Error] {e}")
        return None

def interrupt_avatar(sessionid: str = "0"):
    """打断数字人说话"""
    url = f"{BASE_URL}/interrupt_talk"
    payload = {"sessionid": sessionid}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("[OK] 已打断说话")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Error] {e}")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("虚拟摄像头数字人控制台 (Session 0)")
    print("=" * 50)
    print("命令说明：")
    print("  - 直接输入文本：让数字人说出该文本")
    print("  - 输入 'interrupt'：打断当前说话")
    print("  - 输入 'exit' 或 'quit'：退出程序")
    print("  - 输入空行：不执行任何操作")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n请输入文本 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出程序")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("退出程序")
            break
        elif user_input.lower() == "interrupt":
            interrupt_avatar()
        else:
            # 默认使用 echo 模式直接说话
            make_avatar_speak(user_input)