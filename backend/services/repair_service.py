# services/repair_service.py
import os
import time
import uuid
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional

# 导入现有修复算法
import sys
import os
import importlib.util

# 获取当前文件目录
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)

# 添加backend目录到Python搜索路径，确保可以找到deps模块
sys.path.append(backend_dir)

# 动态导入seathru-mono-e2e.py
seathru_file_path = os.path.join(backend_dir, "seathru-mono-e2e.py")
spec = importlib.util.spec_from_file_location("seathru_mono_e2e", seathru_file_path)
seathru_mono_e2e = importlib.util.module_from_spec(spec)
sys.modules["seathru_mono_e2e"] = seathru_mono_e2e

# 执行模块，此时sys.path已经包含backend目录
spec.loader.exec_module(seathru_mono_e2e)

# 从动态导入的模块中获取函数
process_image_with_seathru = seathru_mono_e2e.process_image_with_seathru

from core.config import settings
from services.quality_service import calculate_quality_metrics

class RepairService:
    """图像修复服务"""
    
    def __init__(self):
        # 使用相对于backend目录的路径
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
        self.output_dir = os.path.join(os.getcwd(), "outputs")
        self.models_dir = os.path.join(os.getcwd(), "models")
        
        # 确保目录存在
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_uploaded_image(self, uploaded_file) -> str:
        """保存上传的图像文件"""
        # 生成唯一文件名
        file_ext = uploaded_file.filename.split(".")[-1].lower()
        file_name = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(self.upload_dir, file_name)
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(uploaded_file.file.read())
        
        return file_path, file_name
    
    def repair_single_image(self, image_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """修复单张图像"""
        start_time = time.time()
        
        # 生成输出文件路径
        file_name = os.path.basename(image_path)
        output_file_name = f"repaired_{uuid.uuid4()}_{file_name}"
        output_path = os.path.join(self.output_dir, output_file_name)
        
        # 调用现有修复算法
        try:
            process_image_with_seathru(
                image_path=image_path,
                model_name=params.get("model_name", settings.DEFAULT_MODEL_NAME),
                output_path=output_path,
                size=params.get("size", settings.MAX_IMAGE_SIZE),
                depth_scale=params.get("depth_scale", 10.0),
                depth_offset=params.get("depth_offset", 2.0),
                no_cuda=True,  # 暂时使用CPU模式，后续可配置
                save_depth=params.get("save_depth", False),
                save_intermediate=params.get("save_intermediate", False)
            )
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 计算质量指标
            original_image = Image.open(image_path).convert("RGB")
            repaired_image = Image.open(output_path).convert("RGB")
            
            quality_metrics = calculate_quality_metrics(
                np.array(original_image),
                np.array(repaired_image)
            )
            
            # 构建返回结果
            result = {
                "original_url": f"/uploads/{os.path.basename(image_path)}",
                "repaired_url": f"/outputs/{output_file_name}",
                "quality_metrics": quality_metrics,
                "processing_time": round(processing_time, 2)
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"图像修复失败: {str(e)}")
    
    def repair_batch_images(self, image_paths: list, params: Dict[str, Any]) -> Dict[str, Any]:
        """批量修复图像"""
        results = []
        errors = []
        
        for image_path in image_paths:
            try:
                result = self.repair_single_image(image_path, params)
                results.append(result)
            except Exception as e:
                errors.append({
                    "image_path": image_path,
                    "error": str(e)
                })
        
        return {
            "results": results,
            "errors": errors,
            "total": len(image_paths),
            "success": len(results),
            "failed": len(errors)
        }

# 创建修复服务实例
repair_service = RepairService()

# 修复图像的便捷函数
def repair_image(image_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """修复图像的便捷函数"""
    return repair_service.repair_single_image(image_path, params)
