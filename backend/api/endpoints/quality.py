# api/endpoints/quality.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
import os
import numpy as np
from PIL import Image

from api.schemas.quality import QualityResponse, QualityMetrics
from services.quality_service import calculate_quality_metrics
from core.config import settings

router = APIRouter()

@router.post("/quality", response_model=QualityResponse, summary="图像质量评估")
async def evaluate_quality(
    original_file: UploadFile = File(...),
    repaired_file: UploadFile = File(...)
):
    """评估修复前后图像的质量指标
    
    - **original_file**: 原始图像文件
    - **repaired_file**: 修复后图像文件
    
    返回的质量指标包括：
    - PSNR (峰值信噪比)
    - SSIM (结构相似性)
    - MSE (均方误差)
    - UIQE (水下图像质量指标，可选)
    """
    try:
        # 读取原始图像
        original_image = Image.open(original_file.file).convert("RGB")
        original_np = np.array(original_image)
        
        # 读取修复后图像
        repaired_image = Image.open(repaired_file.file).convert("RGB")
        repaired_np = np.array(repaired_image)
        
        # 确保图像大小相同
        if original_np.shape != repaired_np.shape:
            raise HTTPException(
                status_code=400, 
                detail="原始图像和修复后图像的尺寸必须相同"
            )
        
        # 计算质量指标
        metrics = calculate_quality_metrics(original_np, repaired_np)
        
        # 构造响应
        quality_metrics = QualityMetrics(**metrics)
        
        return QualityResponse(
            metrics=quality_metrics,
            message="质量评估成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"质量评估失败: {str(e)}")

@router.post("/quality/by-url", response_model=QualityResponse, summary="通过URL评估图像质量")
async def evaluate_quality_by_url(
    original_image_url: str = Query(..., description="原始图像URL"),
    repaired_image_url: str = Query(..., description="修复后图像URL")
):
    """通过URL评估修复前后图像的质量指标
    
    - **original_image_url**: 原始图像URL
    - **repaired_image_url**: 修复后图像URL
    
    返回的质量指标包括：
    - PSNR (峰值信噪比)
    - SSIM (结构相似性)
    - MSE (均方误差)
    - UIQE (水下图像质量指标，可选)
    """
    try:
        # 从URL获取图像路径
        def get_image_path_from_url(url: str) -> str:
            """从URL中提取图像路径"""
            # 处理相对URL
            if url.startswith("/uploads/"):
                file_name = url.split("/")[-1]
                return os.path.join(settings.UPLOAD_DIR, file_name)
            elif url.startswith("/outputs/"):
                file_name = url.split("/")[-1]
                return os.path.join(settings.OUTPUT_DIR, file_name)
            else:
                # 绝对URL暂时不支持
                raise HTTPException(
                    status_code=400, 
                    detail="只支持本地相对URL（/uploads/ 或 /outputs/）"
                )
        
        # 获取图像路径
        original_path = get_image_path_from_url(original_image_url)
        repaired_path = get_image_path_from_url(repaired_image_url)
        
        # 验证文件是否存在
        if not os.path.exists(original_path):
            raise HTTPException(status_code=404, detail="原始图像不存在")
        if not os.path.exists(repaired_path):
            raise HTTPException(status_code=404, detail="修复后图像不存在")
        
        # 读取图像
        original_image = Image.open(original_path).convert("RGB")
        original_np = np.array(original_image)
        
        repaired_image = Image.open(repaired_path).convert("RGB")
        repaired_np = np.array(repaired_image)
        
        # 确保图像大小相同
        if original_np.shape != repaired_np.shape:
            raise HTTPException(
                status_code=400, 
                detail="原始图像和修复后图像的尺寸必须相同"
            )
        
        # 计算质量指标
        metrics = calculate_quality_metrics(original_np, repaired_np)
        
        # 构造响应
        quality_metrics = QualityMetrics(**metrics)
        
        return QualityResponse(
            metrics=quality_metrics,
            message="质量评估成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"质量评估失败: {str(e)}")
