# api/schemas/quality.py
from typing import Optional
from pydantic import BaseModel, Field

# 质量评估请求模型
class QualityRequest(BaseModel):
    original_image_url: str = Field(description="原始图像URL")
    repaired_image_url: str = Field(description="修复后图像URL")

# 质量评估结果模型
class QualityMetrics(BaseModel):
    psnr: float = Field(description="峰值信噪比")
    ssim: float = Field(description="结构相似性")
    mse: float = Field(description="均方误差")
    uciqe: Optional[float] = Field(default=None, description="水下图像质量指标")
    uiqm: Optional[float] = Field(default=None, description="水下图像质量指标")

# 质量评估响应模型
class QualityResponse(BaseModel):
    metrics: QualityMetrics = Field(description="质量评估指标")
    message: str = Field(default="质量评估成功", description="响应消息")
