# app.py
import sys
import os

# 添加当前目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uuid
import numpy as np
from PIL import Image

from api.endpoints import repair, quality
from core.config import settings
from services.repair_service import repair_image
from services.quality_service import calculate_quality_metrics

# 创建FastAPI应用
app = FastAPI(
    title="水下图像修复API",
    description="基于Sea-Thru+Monodepth2的水下图像修复服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件服务
from fastapi.staticfiles import StaticFiles

# 确保静态目录存在
os.makedirs(os.path.join(os.getcwd(), "uploads"), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "outputs"), exist_ok=True)

# 挂载静态文件目录
app.mount("/uploads", StaticFiles(directory=os.path.join(os.getcwd(), "uploads")), name="uploads")
app.mount("/outputs", StaticFiles(directory=os.path.join(os.getcwd(), "outputs")), name="outputs")

# 注册路由
app.include_router(repair.router, prefix="/api", tags=["图像修复"])
app.include_router(quality.router, prefix="/api", tags=["图像质量评估"])

# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "水下图像修复API服务正常运行"}

# 根路径
@app.get("/")
def root():
    return {"message": "欢迎使用水下图像修复API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
