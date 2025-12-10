# services/quality_service.py
import numpy as np
from typing import Dict, Any, Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

class QualityService:
    """图像质量评估服务"""
    
    def __init__(self):
        pass
    
    def calculate_psnr(self, original: np.ndarray, repaired: np.ndarray) -> float:
        """计算峰值信噪比（PSNR）"""
        # 确保图像是相同大小
        if original.shape != repaired.shape:
            raise ValueError("原始图像和修复图像必须具有相同的尺寸")
        
        # 确保图像是uint8类型
        if original.dtype != np.uint8:
            original = original.astype(np.uint8)
        if repaired.dtype != np.uint8:
            repaired = repaired.astype(np.uint8)
        
        return peak_signal_noise_ratio(original, repaired, data_range=255)
    
    def calculate_ssim(self, original: np.ndarray, repaired: np.ndarray) -> float:
        """计算结构相似性（SSIM）"""
        # 确保图像是相同大小
        if original.shape != repaired.shape:
            raise ValueError("原始图像和修复图像必须具有相同的尺寸")
        
        # 确保图像是uint8类型
        if original.dtype != np.uint8:
            original = original.astype(np.uint8)
        if repaired.dtype != np.uint8:
            repaired = repaired.astype(np.uint8)
        
        # 对于RGB图像，使用multichannel参数
        if original.ndim == 3:
            return structural_similarity(original, repaired, win_size=11, data_range=255, multichannel=True, channel_axis=-1)
        else:
            return structural_similarity(original, repaired, win_size=11, data_range=255)
    
    def calculate_mse(self, original: np.ndarray, repaired: np.ndarray) -> float:
        """计算均方误差（MSE）"""
        # 确保图像是相同大小
        if original.shape != repaired.shape:
            raise ValueError("原始图像和修复图像必须具有相同的尺寸")
        
        # 确保图像是uint8类型
        if original.dtype != np.uint8:
            original = original.astype(np.uint8)
        if repaired.dtype != np.uint8:
            repaired = repaired.astype(np.uint8)
        
        return mean_squared_error(original, repaired)
    
    def calculate_uciqe(self, image: np.ndarray) -> Optional[float]:
        """计算水下图像质量指标（UIQE）
        参考：Underwater Image Quality Assessment in the Wild
        """
        try:
            # 确保图像是RGB类型
            if image.ndim != 3:
                return None
            
            # 转换为Lab颜色空间
            from skimage import color
            lab = color.rgb2lab(image)
            
            # 计算颜色fulness、对比度和饱和度
            l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
            
            # 颜色fulness
            chroma = np.sqrt(a**2 + b**2)
            c_avg = np.mean(chroma)
            c_std = np.std(chroma)
            
            # 对比度
            l_std = np.std(l)
            
            # 饱和度
            s = chroma / (l + 1e-10)
            s_avg = np.mean(s)
            
            # UIQE公式：0.4680 * c_std + 0.2745 * s_avg + 0.2576 * l_std
            uciqe = 0.4680 * c_std + 0.2745 * s_avg + 0.2576 * l_std
            return uciqe
        except Exception as e:
            # 如果计算失败，返回None
            return None
    
    def calculate_quality_metrics(self, original: np.ndarray, repaired: np.ndarray) -> Dict[str, float]:
        """计算所有质量指标
        
        Args:
            original: 原始图像（numpy数组，uint8类型）
            repaired: 修复后图像（numpy数组，uint8类型）
            
        Returns:
            包含所有质量指标的字典
        """
        metrics = {}
        
        # 计算PSNR
        metrics["psnr"] = round(self.calculate_psnr(original, repaired), 4)
        
        # 计算SSIM
        metrics["ssim"] = round(self.calculate_ssim(original, repaired), 4)
        
        # 计算MSE
        metrics["mse"] = round(self.calculate_mse(original, repaired), 4)
        
        # 计算修复后图像的UIQE
        uciqe = self.calculate_uciqe(repaired)
        if uciqe is not None:
            metrics["uciqe"] = round(uciqe, 4)
        
        return metrics

# 创建质量服务实例
quality_service = QualityService()

# 计算质量指标的便捷函数
def calculate_quality_metrics(original: np.ndarray, repaired: np.ndarray) -> Dict[str, float]:
    """计算图像质量指标的便捷函数"""
    return quality_service.calculate_quality_metrics(original, repaired)
