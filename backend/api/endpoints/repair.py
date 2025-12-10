# api/endpoints/repair.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import uuid

from api.schemas.repair import (
    RepairRequest, 
    BatchRepairRequest, 
    RepairResult, 
    BatchRepairResponse, 
    TaskStatus
)
from services.repair_service import repair_service
from core.config import settings

router = APIRouter()

# 全局任务存储（实际项目中应该使用数据库）
tasks = {}

@router.post("/repair", response_model=RepairResult, summary="单张图像修复")
async def repair_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query(settings.DEFAULT_MODEL_NAME, description="模型名称"),
    size: Optional[int] = Query(1024, description="图像处理的最大分辨率"),
    depth_scale: Optional[float] = Query(10.0, description="深度图整体缩放"),
    depth_offset: Optional[float] = Query(2.0, description="深度补偿偏移"),
    save_depth: Optional[bool] = Query(False, description="保存深度图调试文件"),
    save_intermediate: Optional[bool] = Query(False, description="保存中间结果")
):
    """修复单张水下图像
    
    - **file**: 要修复的水下图像文件
    - **model_name**: 使用的模型名称，默认是 mono_1024x320
    - **size**: 图像处理的最大分辨率，默认是 1024
    - **depth_scale**: 深度图整体缩放系数，默认是 10.0
    - **depth_offset**: 深度补偿偏移，默认是 2.0
    - **save_depth**: 是否保存深度图调试文件，默认是 False
    - **save_intermediate**: 是否保存中间结果，默认是 False
    """
    try:
        # 保存上传的图像
        image_path, file_name = repair_service.save_uploaded_image(file)
        
        # 准备修复参数
        params = {
            "model_name": model_name,
            "size": size,
            "depth_scale": depth_scale,
            "depth_offset": depth_offset,
            "save_depth": save_depth,
            "save_intermediate": save_intermediate
        }
        
        # 修复图像
        result = repair_service.repair_single_image(image_path, params)
        
        # 返回结果
        return RepairResult(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像修复失败: {str(e)}")

@router.post("/repair/batch", response_model=BatchRepairResponse, summary="批量图像修复")
async def repair_images_batch(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Query(settings.DEFAULT_MODEL_NAME, description="模型名称"),
    size: Optional[int] = Query(1024, description="图像处理的最大分辨率"),
    depth_scale: Optional[float] = Query(10.0, description="深度图整体缩放"),
    depth_offset: Optional[float] = Query(2.0, description="深度补偿偏移"),
    save_depth: Optional[bool] = Query(False, description="保存深度图调试文件"),
    save_intermediate: Optional[bool] = Query(False, description="保存中间结果")
):
    """批量修复水下图像
    
    - **files**: 要修复的水下图像文件列表
    - **model_name**: 使用的模型名称，默认是 mono_1024x320
    - **size**: 图像处理的最大分辨率，默认是 1024
    - **depth_scale**: 深度图整体缩放系数，默认是 10.0
    - **depth_offset**: 深度补偿偏移，默认是 2.0
    - **save_depth**: 是否保存深度图调试文件，默认是 False
    - **save_intermediate**: 是否保存中间结果，默认是 False
    """
    try:
        # 验证批量大小
        if len(files) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"批量大小超过限制，最大允许 {settings.MAX_BATCH_SIZE} 张图像"
            )
        
        # 保存上传的图像
        image_paths = []
        for file in files:
            image_path, file_name = repair_service.save_uploaded_image(file)
            image_paths.append(image_path)
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 准备修复参数
        params = {
            "model_name": model_name,
            "size": size,
            "depth_scale": depth_scale,
            "depth_offset": depth_offset,
            "save_depth": save_depth,
            "save_intermediate": save_intermediate
        }
        
        # 立即处理批量任务（实际项目中应该使用异步任务队列）
        result = repair_service.repair_batch_images(image_paths, params)
        
        # 保存任务结果
        tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "total_images": len(image_paths),
            "completed_images": len(result["results"]),
            "results": result["results"],
            "errors": result["errors"]
        }
        
        # 返回结果
        return BatchRepairResponse(
            task_id=task_id,
            total_images=len(image_paths),
            message="批量修复任务已创建"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量图像修复失败: {str(e)}")

@router.get("/task/{task_id}", response_model=TaskStatus, summary="查询任务状态")
async def get_task_status(task_id: str):
    """查询批量修复任务的状态
    
    - **task_id**: 任务ID
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        total_images=task["total_images"],
        completed_images=task["completed_images"],
        results=task["results"] if task["status"] == "completed" else None,
        error=None
    )

@router.get("/uploads/{file_name}", summary="获取上传的图像")
async def get_uploaded_image(file_name: str):
    """获取上传的原始图像
    
    - **file_name**: 图像文件名
    """
    file_path = os.path.join(settings.UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="图像不存在")
    
    return FileResponse(file_path)

@router.get("/outputs/{file_name}", summary="获取修复后的图像")
async def get_output_image(file_name: str):
    """获取修复后的图像
    
    - **file_name**: 修复后的图像文件名
    """
    file_path = os.path.join(settings.OUTPUT_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="修复后的图像不存在")
    
    return FileResponse(file_path)
