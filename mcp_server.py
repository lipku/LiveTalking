#!/usr/bin/env python3
"""
MCP (Model Context Protocol) 服务器
提供数字人控制接口
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
from aiohttp import web
import aiohttp
import aiohttp_cors
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.one_api import get_answer_from_query
from logger import logger

# 配置日志
logging.basicConfig(level=logging.INFO)

class DigitalHumanMCP:
    """数字人 MCP 服务器"""
    
    def __init__(self, base_url: str = "http://localhost:8010"):
        self.base_url = base_url
        self.session_id = None
        self.is_initialized = False
        
    async def initialize(self):
        """初始化连接"""
        try:
            # 创建 WebRTC 连接获取 session ID
            async with aiohttp.ClientSession() as session:
                # 模拟创建 offer
                offer_data = {
                    "sdp": "dummy_sdp",  # 实际使用时需要真实的 SDP
                    "type": "offer"
                }
                async with session.post(f"{self.base_url}/offer", json=offer_data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        self.session_id = result.get("sessionid")
                        self.is_initialized = True
                        logger.info(f"初始化成功，Session ID: {self.session_id}")
                        return True
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    async def speak_text(self, text: str, use_llm: bool = False) -> Dict[str, Any]:
        """
        让数字人说话
        
        Args:
            text: 要说的文本
            use_llm: 是否使用 LLM 生成回复
            
        Returns:
            操作结果
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # 如果使用 LLM，先获取回复
            if use_llm:
                try:
                    llm_response = get_answer_from_query(
                        text, 
                        system_prompt="你是一个友好的数字人助手，请用简洁友好的方式回复。"
                    )
                    actual_text = llm_response
                    logger.info(f"LLM 回复: {llm_response}")
                except Exception as e:
                    logger.error(f"LLM 调用失败: {e}")
                    actual_text = text
            else:
                actual_text = text
            
            # 发送文本到数字人
            async with aiohttp.ClientSession() as session:
                data = {
                    "sessionid": self.session_id,
                    "type": "echo",
                    "text": actual_text
                }
                async with session.post(f"{self.base_url}/human", json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return {
                            "status": "success",
                            "message": f"数字人正在说: {actual_text}",
                            "session_id": self.session_id,
                            "text": actual_text
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"请求失败: {resp.status}"
                        }
        except Exception as e:
            logger.error(f"speak_text 错误: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def interrupt_speaking(self) -> Dict[str, Any]:
        """
        打断数字人说话
        
        Returns:
            操作结果
        """
        try:
            if not self.is_initialized:
                return {
                    "status": "error",
                    "message": "未初始化，请先调用 speak_text"
                }
            
            async with aiohttp.ClientSession() as session:
                data = {
                    "sessionid": self.session_id
                }
                async with session.post(f"{self.base_url}/interrupt_talk", json=data) as resp:
                    if resp.status == 200:
                        return {
                            "status": "success",
                            "interrupted": True,
                            "message": "已打断数字人说话",
                            "session_id": self.session_id
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"请求失败: {resp.status}"
                        }
        except Exception as e:
            logger.error(f"interrupt_speaking 错误: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_speaking_status(self) -> Dict[str, Any]:
        """
        获取数字人说话状态
        
        Returns:
            说话状态信息
        """
        try:
            if not self.is_initialized:
                return {
                    "status": "error",
                    "message": "未初始化",
                    "is_speaking": False
                }
            
            async with aiohttp.ClientSession() as session:
                data = {
                    "sessionid": self.session_id
                }
                async with session.post(f"{self.base_url}/is_speaking", json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        is_speaking = result.get("data", False)
                        return {
                            "status": "success",
                            "is_speaking": is_speaking,
                            "session_id": self.session_id,
                            "message": "正在说话" if is_speaking else "未在说话"
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"请求失败: {resp.status}",
                            "is_speaking": False
                        }
        except Exception as e:
            logger.error(f"get_speaking_status 错误: {e}")
            return {
                "status": "error",
                "message": str(e),
                "is_speaking": False
            }

# 全局 MCP 实例
mcp_instance = DigitalHumanMCP()

# Web API 路由处理
async def handle_speak(request):
    """处理说话请求"""
    try:
        data = await request.json()
        text = data.get("text", "")
        use_llm = data.get("use_llm", False)
        
        if not text:
            return web.Response(
                text=json.dumps({"status": "error", "message": "文本不能为空"}),
                content_type="application/json",
                status=400
            )
        
        result = await mcp_instance.speak_text(text, use_llm)
        return web.Response(
            text=json.dumps(result),
            content_type="application/json"
        )
    except Exception as e:
        return web.Response(
            text=json.dumps({"status": "error", "message": str(e)}),
            content_type="application/json",
            status=500
        )

async def handle_interrupt(request):
    """处理打断请求"""
    try:
        result = await mcp_instance.interrupt_speaking()
        return web.Response(
            text=json.dumps(result),
            content_type="application/json"
        )
    except Exception as e:
        return web.Response(
            text=json.dumps({"status": "error", "message": str(e)}),
            content_type="application/json",
            status=500
        )

async def handle_status(request):
    """处理状态查询请求"""
    try:
        result = await mcp_instance.get_speaking_status()
        return web.Response(
            text=json.dumps(result),
            content_type="application/json"
        )
    except Exception as e:
        return web.Response(
            text=json.dumps({"status": "error", "message": str(e)}),
            content_type="application/json",
            status=500
        )

async def handle_test_page(request):
    """提供测试页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP 数字人控制接口测试</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .control-panel {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .input-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #555;
            }
            input[type="text"], textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            textarea {
                min-height: 100px;
                resize: vertical;
            }
            button {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                margin-right: 10px;
                font-size: 16px;
            }
            button:hover {
                background: #45a049;
            }
            button.interrupt {
                background: #f44336;
            }
            button.interrupt:hover {
                background: #da190b;
            }
            button.status {
                background: #2196F3;
            }
            button.status:hover {
                background: #0b7dda;
            }
            .status-display {
                background: #e7f3ff;
                padding: 15px;
                border-radius: 4px;
                margin-top: 20px;
                border: 1px solid #2196F3;
            }
            .response {
                background: #f0f0f0;
                padding: 15px;
                border-radius: 4px;
                margin-top: 20px;
                white-space: pre-wrap;
                font-family: monospace;
            }
            .checkbox-group {
                margin: 10px 0;
            }
            .checkbox-group input {
                margin-right: 5px;
            }
        </style>
    </head>
    <body>
        <h1>🤖 MCP 数字人控制接口测试</h1>
        
        <div class="control-panel">
            <h2>控制面板</h2>
            
            <div class="input-group">
                <label for="textInput">输入文本:</label>
                <textarea id="textInput" placeholder="请输入要让数字人说的话...">你好，我是数字人助手，很高兴为您服务！</textarea>
            </div>
            
            <div class="checkbox-group">
                <label>
                    <input type="checkbox" id="useLLM">
                    使用 LLM 生成回复
                </label>
            </div>
            
            <div>
                <button onclick="speak()">🗣️ 让数字人说话</button>
                <button class="interrupt" onclick="interrupt()">⏹️ 打断说话</button>
                <button class="status" onclick="getStatus()">📊 查询状态</button>
            </div>
        </div>
        
        <div class="control-panel">
            <h2>快速测试</h2>
            <button onclick="quickTest('你好')">说"你好"</button>
            <button onclick="quickTest('今天天气真不错')">说"今天天气真不错"</button>
            <button onclick="quickTest('我是一个数字人', true)">LLM: 自我介绍</button>
            <button onclick="quickTest('给我讲个笑话', true)">LLM: 讲笑话</button>
        </div>
        
        <div class="status-display" id="statusDisplay">
            状态: 等待操作...
        </div>
        
        <div class="response" id="response">
            响应将显示在这里...
        </div>
        
        <script>
            const API_BASE = '';
            
            async function speak() {
                const text = document.getElementById('textInput').value;
                const useLLM = document.getElementById('useLLM').checked;
                
                if (!text) {
                    alert('请输入文本');
                    return;
                }
                
                updateStatus('发送请求中...');
                
                try {
                    const response = await fetch(API_BASE + '/api/speak', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            use_llm: useLLM
                        })
                    });
                    
                    const data = await response.json();
                    displayResponse(data);
                    
                    if (data.status === 'success') {
                        updateStatus('✅ 成功: ' + data.message);
                    } else {
                        updateStatus('❌ 错误: ' + data.message);
                    }
                } catch (error) {
                    displayResponse({error: error.toString()});
                    updateStatus('❌ 请求失败: ' + error);
                }
            }
            
            async function interrupt() {
                updateStatus('发送打断请求...');
                
                try {
                    const response = await fetch(API_BASE + '/api/interrupt', {
                        method: 'POST'
                    });
                    
                    const data = await response.json();
                    displayResponse(data);
                    
                    if (data.status === 'success') {
                        updateStatus('✅ ' + data.message);
                    } else {
                        updateStatus('❌ 错误: ' + data.message);
                    }
                } catch (error) {
                    displayResponse({error: error.toString()});
                    updateStatus('❌ 请求失败: ' + error);
                }
            }
            
            async function getStatus() {
                updateStatus('查询状态中...');
                
                try {
                    const response = await fetch(API_BASE + '/api/status');
                    const data = await response.json();
                    displayResponse(data);
                    
                    if (data.status === 'success') {
                        const speakingText = data.is_speaking ? '🔊 正在说话' : '🔇 未在说话';
                        updateStatus(speakingText + ' (Session: ' + data.session_id + ')');
                    } else {
                        updateStatus('❌ 错误: ' + data.message);
                    }
                } catch (error) {
                    displayResponse({error: error.toString()});
                    updateStatus('❌ 请求失败: ' + error);
                }
            }
            
            function quickTest(text, useLLM = false) {
                document.getElementById('textInput').value = text;
                document.getElementById('useLLM').checked = useLLM;
                speak();
            }
            
            function updateStatus(message) {
                document.getElementById('statusDisplay').textContent = '状态: ' + message;
            }
            
            function displayResponse(data) {
                document.getElementById('response').textContent = JSON.stringify(data, null, 2);
            }
            
            // 页面加载时自动获取状态
            window.onload = function() {
                getStatus();
            };
        </script>
    </body>
    </html>
    """
    return web.Response(text=html_content, content_type="text/html")

def create_app():
    """创建 Web 应用"""
    app = web.Application()
    
    # 添加路由
    app.router.add_post('/api/speak', handle_speak)
    app.router.add_post('/api/interrupt', handle_interrupt)
    app.router.add_get('/api/status', handle_status)
    app.router.add_get('/', handle_test_page)
    
    # 配置 CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    
    # 为所有路由应用 CORS
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

def main():
    """主函数"""
    app = create_app()
    port = 8011
    
    logger.info(f"MCP 服务器启动在 http://localhost:{port}")
    logger.info(f"测试页面: http://localhost:{port}/")
    logger.info("API 端点:")
    logger.info("  POST /api/speak - 让数字人说话")
    logger.info("  POST /api/interrupt - 打断数字人")
    logger.info("  GET  /api/status - 获取说话状态")
    
    web.run_app(app, host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()
