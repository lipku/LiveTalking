# 添加到 app.py 的 GPU 监控端点代码
# 将以下代码添加到 app.py 中的适当位置

# 在文件顶部导入部分添加：
from gpu_monitor import get_gpu_status, get_gpu_status_detailed

# 在路由定义部分（约第 395-403 行附近）添加：
async def gpu_status(request):
    """返回 GPU 使用状态"""
    try:
        gpu_info = get_gpu_status()
        return web.Response(
            content_type="application/json",
            text=json.dumps(gpu_info)
        )
    except Exception as e:
        logger.exception('gpu_status exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "error": str(e),
                "gpu_usage": 0,
                "mem_used": 0,
                "mem_total": 0,
                "temperature": 0,
                "power": 0
            }),
            status=500
        )

async def gpu_status_detailed(request):
    """返回详细的 GPU 信息"""
    try:
        gpu_info = get_gpu_status_detailed()
        return web.Response(
            content_type="application/json",
            text=json.dumps(gpu_info)
        )
    except Exception as e:
        logger.exception('gpu_status_detailed exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps({"error": str(e)}),
            status=500
        )

# 在应用路由注册部分（约第 395-404 行）添加：
# appasync.router.add_get("/gpu_status", gpu_status)
# appasync.router.add_get("/gpu_status_detailed", gpu_status_detailed)

# 完整的路由注册示例：
"""
appasync = web.Application(client_max_size=1024**2*100)
appasync.on_shutdown.append(on_shutdown)
appasync.router.add_post("/offer", offer)
appasync.router.add_post("/human", human)
appasync.router.add_post("/humanaudio", humanaudio)
appasync.router.add_post("/set_audiotype", set_audiotype)
appasync.router.add_post("/record", record)
appasync.router.add_post("/interrupt_talk", interrupt_talk)
appasync.router.add_post("/is_speaking", is_speaking)
appasync.router.add_get("/gpu_status", gpu_status)  # 新增
appasync.router.add_get("/gpu_status_detailed", gpu_status_detailed)  # 新增
appasync.router.add_static('/',path='web')
"""
