# api/schemas/repair.py
from typing import Optional, List
from pydantic import BaseModel, Field

# 修复参数模型
class RepairParams(BaseModel):
    model_name: str = Field(default="mono_1024x320", description="模型名称")
    size: int = Field(default=1024, description="图像处理的最大分辨率")
    depth_scale: float = Field(default=10.0, description="深度图整体缩放")
    depth_offset: float = Field(default=2.0, description="深度补偿偏移")
    save_depth: bool = Field(default=False, description="保存深度图调试文件")
    save_intermediate: bool = Field(default=False, description="保存中间结果")

# 单张图像修复请求模型
class RepairRequest(BaseModel):
    params: Optional[RepairParams] = Field(default_factory=RepairParams, description="修复参数")

# 批量图像修复请求模型
class BatchRepairRequest(BaseModel):
    params: Optional[RepairParams] = Field(default_factory=RepairParams, description="修复参数")

# 图像修复结果模型
class RepairResult(BaseModel):
    original_url: str = Field(description="原始图像URL")
    repaired_url: str = Field(description="修复后图像URL")
    quality_metrics: Optional[dict] = Field(default=None, description="质量评估指标")
    processing_time: float = Field(description="处理时间（秒）")

# 批量修复任务响应模型
class BatchRepairResponse(BaseModel):
    task_id: str = Field(description="任务ID")
    total_images: int = Field(description="总图像数量")
    message: str = Field(description="任务创建成功信息")

# 任务状态模型
class TaskStatus(BaseModel):
    task_id: str = Field(description="任务ID")
    status: str = Field(description="任务状态：pending, processing, completed, failed")
    progress: int = Field(description="处理进度（0-100）")
    completed_images: int = Field(description="已完成图像数量")
    total_images: int = Field(description="总图像数量")
    results: Optional[List[RepairResult]] = Field(default=None, description="修复结果列表")
    error: Optional[str] = Field(default=None, description="错误信息")
