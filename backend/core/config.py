# core/config.py
import os
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 应用基本配置
    APP_NAME: str = "水下图像修复API"
    DEBUG: bool = True
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 图像处理配置
    MAX_IMAGE_SIZE: int = 1024  # 最大图像处理尺寸
    DEFAULT_MODEL_NAME: str = "mono_1024x320"
    
    # 文件存储配置
    UPLOAD_DIR: str = os.path.join(os.getcwd(), "uploads")
    OUTPUT_DIR: str = os.path.join(os.getcwd(), "outputs")
    
    # 模型路径配置
    MODELS_DIR: str = os.path.join(os.getcwd(), "models")
    
    # 质量评估配置
    QUALITY_METRICS: list = ["psnr", "ssim", "mse"]
    
    # 批量处理配置
    MAX_BATCH_SIZE: int = 10  # 最大批量处理数量
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 创建配置实例
settings = Settings()

# 确保存储目录存在
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
