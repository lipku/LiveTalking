#!/usr/bin/env python3
"""
MCP 接口测试代码
用于测试数字人控制接口
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

class MCPTester:
    """MCP 接口测试类"""
    
    def __init__(self, base_url: str = "http://localhost:8011"):
        self.base_url = base_url
        self.test_results = []
        
    async def test_speak(self, text: str, use_llm: bool = False) -> Dict[str, Any]:
        """测试说话接口"""
        print(f"\n📝 测试说话接口...")
        print(f"   文本: {text}")
        print(f"   使用LLM: {use_llm}")
        
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "text": text,
                    "use_llm": use_llm
                }
                async with session.post(f"{self.base_url}/api/speak", json=data) as resp:
                    result = await resp.json()
                    print(f"   ✅ 响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    self.test_results.append(("speak", True, result))
                    return result
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            self.test_results.append(("speak", False, str(e)))
            return {"error": str(e)}
    
    async def test_interrupt(self) -> Dict[str, Any]:
        """测试打断接口"""
        print(f"\n⏹️ 测试打断接口...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/interrupt") as resp:
                    result = await resp.json()
                    print(f"   ✅ 响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    self.test_results.append(("interrupt", True, result))
                    return result
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            self.test_results.append(("interrupt", False, str(e)))
            return {"error": str(e)}
    
    async def test_status(self) -> Dict[str, Any]:
        """测试状态查询接口"""
        print(f"\n📊 测试状态查询接口...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/status") as resp:
                    result = await resp.json()
                    print(f"   ✅ 响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    self.test_results.append(("status", True, result))
                    return result
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            self.test_results.append(("status", False, str(e)))
            return {"error": str(e)}
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("🚀 开始 MCP 接口测试")
        print("=" * 60)
        
        # 测试 1: 获取初始状态
        print("\n【测试 1】获取初始状态")
        await self.test_status()
        
        # 测试 2: 简单文本说话
        print("\n【测试 2】简单文本说话")
        await self.test_speak("你好，我是数字人助手")
        await asyncio.sleep(2)  # 等待数字人开始说话
        
        # 测试 3: 查询说话状态
        print("\n【测试 3】查询说话状态（应该在说话中）")
        await self.test_status()
        
        # 测试 4: 打断说话
        print("\n【测试 4】打断说话")
        await self.test_interrupt()
        await asyncio.sleep(1)
        
        # 测试 5: 再次查询状态
        print("\n【测试 5】查询状态（应该已停止说话）")
        await self.test_status()
        
        # 测试 6: 使用 LLM 生成回复
        print("\n【测试 6】使用 LLM 生成回复")
        await self.test_speak("请做一个简短的自我介绍", use_llm=True)
        await asyncio.sleep(3)
        
        # 测试 7: 长文本测试
        print("\n【测试 7】长文本测试")
        long_text = """
        今天天气真不错，阳光明媚，微风徐徐。
        这样的天气最适合出去散步了。
        我想去公园看看花，听听鸟叫声。
        """
        await self.test_speak(long_text.strip())
        await asyncio.sleep(5)
        
        # 测试 8: 连续请求测试
        print("\n【测试 8】连续请求测试")
        for i in range(3):
            await self.test_speak(f"这是第 {i+1} 条测试消息")
            await asyncio.sleep(1)
        
        # 测试 9: 中文和英文混合
        print("\n【测试 9】中英文混合测试")
        await self.test_speak("Hello 大家好，Welcome to LiveTalking 数字人系统")
        await asyncio.sleep(3)
        
        # 测试 10: 特殊字符测试
        print("\n【测试 10】特殊字符测试")
        await self.test_speak("测试特殊字符：😊 @ # $ % & * ()")
        
        # 打印测试总结
        self.print_summary()
    
    async def run_interactive_test(self):
        """交互式测试"""
        print("=" * 60)
        print("🎮 交互式 MCP 测试")
        print("=" * 60)
        print("\n命令说明:")
        print("  1. 输入文本 - 让数字人说话")
        print("  2. /llm <文本> - 使用 LLM 生成回复")
        print("  3. /interrupt - 打断数字人")
        print("  4. /status - 查询状态")
        print("  5. /quit - 退出测试")
        print("-" * 60)
        
        while True:
            try:
                command = input("\n请输入命令: ").strip()
                
                if command == "/quit":
                    print("👋 退出测试")
                    break
                elif command == "/interrupt":
                    await self.test_interrupt()
                elif command == "/status":
                    await self.test_status()
                elif command.startswith("/llm "):
                    text = command[5:].strip()
                    if text:
                        await self.test_speak(text, use_llm=True)
                    else:
                        print("❌ 请提供文本")
                elif command:
                    await self.test_speak(command)
                else:
                    print("❌ 请输入有效命令")
                    
            except KeyboardInterrupt:
                print("\n👋 退出测试")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    async def performance_test(self):
        """性能测试"""
        print("=" * 60)
        print("⚡ MCP 接口性能测试")
        print("=" * 60)
        
        # 测试响应时间
        print("\n【响应时间测试】")
        response_times = []
        
        for i in range(10):
            start_time = time.time()
            await self.test_status()
            elapsed = time.time() - start_time
            response_times.append(elapsed)
            print(f"   请求 {i+1}: {elapsed:.3f} 秒")
        
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n📊 统计结果:")
        print(f"   平均响应时间: {avg_time:.3f} 秒")
        print(f"   最快响应时间: {min_time:.3f} 秒")
        print(f"   最慢响应时间: {max_time:.3f} 秒")
        
        # 并发测试
        print("\n【并发请求测试】")
        print("   发送 5 个并发请求...")
        
        tasks = []
        for i in range(5):
            tasks.append(self.test_speak(f"并发测试消息 {i+1}"))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        success_count = sum(1 for r in results if not isinstance(r, Exception) and "error" not in r)
        print(f"   ✅ 成功: {success_count}/5")
        print(f"   ⏱️ 总耗时: {elapsed:.3f} 秒")
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("📋 测试总结")
        print("=" * 60)
        
        success_count = sum(1 for _, success, _ in self.test_results if success)
        total_count = len(self.test_results)
        
        print(f"\n总测试数: {total_count}")
        print(f"成功: {success_count}")
        print(f"失败: {total_count - success_count}")
        print(f"成功率: {success_count/total_count*100:.1f}%")
        
        if total_count - success_count > 0:
            print("\n失败的测试:")
            for test_name, success, result in self.test_results:
                if not success:
                    print(f"  - {test_name}: {result}")

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP 接口测试工具")
    parser.add_argument("--url", default="http://localhost:8011", help="MCP 服务器地址")
    parser.add_argument("--mode", choices=["all", "interactive", "performance"], 
                       default="all", help="测试模式")
    
    args = parser.parse_args()
    
    tester = MCPTester(args.url)
    
    print(f"🔗 连接到 MCP 服务器: {args.url}")
    
    # 检查服务器是否可用
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/api/status", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"⚠️ 警告: 服务器返回状态码 {resp.status}")
    except Exception as e:
        print(f"❌ 无法连接到 MCP 服务器: {e}")
        print("请确保:")
        print("  1. 主服务已启动: python app.py")
        print("  2. MCP 服务器已启动: python mcp_server.py")
        return
    
    if args.mode == "all":
        await tester.run_all_tests()
    elif args.mode == "interactive":
        await tester.run_interactive_test()
    elif args.mode == "performance":
        await tester.performance_test()

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║                   MCP 数字人接口测试工具                    ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 测试中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
