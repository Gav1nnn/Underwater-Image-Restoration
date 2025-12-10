<script setup>
import { ref, reactive } from 'vue'
import axios from 'axios'
import ImageCompare from './ImageCompare.vue'

// 状态管理
const isUploading = ref(false)
const isProcessing = ref(false)
const originalImage = ref(null)
const repairedImage = ref(null)
const qualityMetrics = ref(null)
const errorMessage = ref('')

// 修复参数
const repairParams = reactive({
  model_name: 'mono_1024x320',
  size: 1024,
  depth_scale: 10.0,
  depth_offset: 2.0,
  save_depth: false,
  save_intermediate: false
})

// 上传图像
const handleImageUpload = (uploadFile) => {
  const file = uploadFile.raw
  if (!file) return
  
  // 验证文件类型
  if (!file.type.startsWith('image/')) {
    errorMessage.value = '请上传有效的图像文件'
    return
  }
  
  // 读取图像文件
  const reader = new FileReader()
  reader.onload = (e) => {
    originalImage.value = e.target.result
    repairedImage.value = null
    qualityMetrics.value = null
    errorMessage.value = ''
  }
  reader.readAsDataURL(file)
}

// 修复图像
const repairImage = async () => {
  if (!originalImage.value) {
    errorMessage.value = '请先上传图像'
    return
  }
  
  try {
    isProcessing.value = true
    errorMessage.value = ''
    
    // 从DataURL创建Blob
    const response = await fetch(originalImage.value)
    const blob = await response.blob()
    
    // 创建FormData
    const formData = new FormData()
    formData.append('file', blob, 'uploaded_image.jpg')
    
    // 添加修复参数
    Object.entries(repairParams).forEach(([key, value]) => {
      formData.append(key, value.toString())
    })
    
    // 发送请求
    const result = await axios.post('http://localhost:8000/api/repair', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    // 处理响应
    if (result.data && result.data.repaired_url) {
      repairedImage.value = `http://localhost:8000${result.data.repaired_url}`
      qualityMetrics.value = result.data.quality_metrics
    }
    
  } catch (error) {
    console.error('修复图像失败:', error)
    errorMessage.value = `修复失败: ${error.message || '未知错误'}`
  } finally {
    isProcessing.value = false
  }
}

// 下载修复后的图像
const downloadImage = async () => {
  if (!repairedImage.value) return
  
  try {
    // 使用fetch获取图片数据
    const response = await fetch(repairedImage.value)
    const blob = await response.blob()
    
    // 创建Blob URL
    const blobUrl = URL.createObjectURL(blob)
    
    // 创建下载链接
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = 'repaired_image.jpg'
    
    // 触发下载
    link.click()
    
    // 释放Blob URL
    URL.revokeObjectURL(blobUrl)
  } catch (error) {
    console.error('下载图片失败:', error)
  }
}
</script>

<template>
  <div class="single-repair-container">
    <h2 class="section-title">单张图像修复</h2>
    
    <!-- 错误信息 -->
    <el-alert
      v-if="errorMessage"
      type="error"
      :title="errorMessage"
      show-icon
      closable
      @close="errorMessage = ''"
      class="error-alert"
    />
    
    <div class="repair-content">
      <!-- 左侧：上传和参数设置 -->
      <div class="left-panel">
        <!-- 图像上传 -->
        <el-card class="upload-card">
          <template #header>
            <div class="card-header">
              <span>图像上传</span>
            </div>
          </template>
          
          <el-upload
            class="image-uploader"
            action="#"
            :auto-upload="false"
            :on-change="handleImageUpload"
            :show-file-list="false"
            accept="image/*"
          >
            <div v-if="!originalImage" class="upload-placeholder">
              <el-icon class="upload-icon"><Plus /></el-icon>
              <div class="upload-text">点击或拖拽上传图像</div>
              <div class="upload-hint">支持 JPG、PNG 等格式</div>
            </div>
            <img v-else :src="originalImage" class="uploaded-image" />
          </el-upload>
        </el-card>
        
        <!-- 修复参数设置 -->
        <el-card class="params-card" v-if="originalImage">
          <template #header>
            <div class="card-header">
              <span>修复参数</span>
            </div>
          </template>
          
          <el-form :model="repairParams" label-width="120px" size="small">
            <el-form-item label="模型名称">
              <el-select v-model="repairParams.model_name" placeholder="选择模型">
                <el-option label="mono_1024x320" value="mono_1024x320" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="处理尺寸">
              <el-input-number
                v-model="repairParams.size"
                :min="512"
                :max="2048"
                :step="64"
                placeholder="最大处理尺寸"
              />
            </el-form-item>
            
            <el-form-item label="深度缩放">
              <el-slider
                v-model="repairParams.depth_scale"
                :min="1.0"
                :max="20.0"
                :step="0.5"
                :show-input="true"
              />
            </el-form-item>
            
            <el-form-item label="深度偏移">
              <el-slider
                v-model="repairParams.depth_offset"
                :min="0.0"
                :max="5.0"
                :step="0.1"
                :show-input="true"
              />
            </el-form-item>
            
            <el-form-item>
              <el-button
                type="primary"
                :loading="isProcessing"
                @click="repairImage"
                :disabled="isProcessing"
                size="large"
              >
                <el-icon v-if="isProcessing"><Loading /></el-icon>
                开始修复
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </div>
      
      <!-- 右侧：结果展示 -->
      <div class="right-panel">
        <el-card class="result-card">
          <template #header>
            <div class="card-header">
              <span>修复结果</span>
              <el-button
                v-if="repairedImage"
                type="success"
                size="small"
                @click="downloadImage"
              >
                <el-icon><Download /></el-icon>
                下载结果
              </el-button>
            </div>
          </template>
          
          <div v-if="!repairedImage" class="result-placeholder">
            <el-icon class="result-icon"><Picture /></el-icon>
            <div class="result-text">修复结果将显示在这里</div>
          </div>
          
          <div v-else class="result-content">
            <!-- 图像对比组件 -->
            <ImageCompare
              :original-image="originalImage"
              :repaired-image="repairedImage"
            />
            
            <!-- 质量指标 -->
            <div class="quality-section" v-if="qualityMetrics">
              <h3 class="quality-title">质量评估指标</h3>
              <div class="quality-metrics-grid">
                <div class="metric-item">
                  <div class="metric-label">PSNR</div>
                  <div class="metric-value">{{ qualityMetrics.psnr.toFixed(2) }}</div>
                  <div class="metric-desc">峰值信噪比（越高越好）</div>
                </div>
                <div class="metric-item">
                  <div class="metric-label">SSIM</div>
                  <div class="metric-value">{{ qualityMetrics.ssim.toFixed(4) }}</div>
                  <div class="metric-desc">结构相似性（越高越好）</div>
                </div>
                <div class="metric-item">
                  <div class="metric-label">MSE</div>
                  <div class="metric-value">{{ qualityMetrics.mse.toFixed(2) }}</div>
                  <div class="metric-desc">均方误差（越低越好）</div>
                </div>
                <div class="metric-item" v-if="qualityMetrics.uciqe">
                  <div class="metric-label">UIQE</div>
                  <div class="metric-value">{{ qualityMetrics.uciqe.toFixed(4) }}</div>
                  <div class="metric-desc">水下图像质量指标</div>
                </div>
              </div>
            </div>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<style scoped>
.single-repair-container {
  width: 100%;
}

.section-title {
  font-size: 1.25rem;
  margin-bottom: 1.5rem;
  color: #333;
  text-align: center;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #0056b3;
}

.error-alert {
  margin-bottom: 1.5rem;
}

.repair-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

/* 左侧面板 */
.left-panel {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.upload-card, .params-card, .result-card {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}

/* 上传组件 */
.image-uploader {
  width: 100%;
  text-align: center;
}

.upload-placeholder {
  border: 2px dashed #d9d9d9;
  border-radius: 8px;
  padding: 2rem;
  cursor: pointer;
  transition: all 0.3s;
}

.upload-placeholder:hover {
  border-color: #0056b3;
  background-color: #f0f8ff;
}

.upload-icon {
  font-size: 2.5rem;
  color: #0056b3;
  margin-bottom: 1rem;
}

.upload-text {
  font-size: 1rem;
  color: #606266;
  margin-bottom: 0.5rem;
}

.upload-hint {
  font-size: 0.875rem;
  color: #909399;
}

.uploaded-image {
  max-width: 100%;
  max-height: 300px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

/* 参数表单 */
.params-card {
  margin-top: auto;
}

/* 右侧面板 */
.right-panel {
  display: flex;
  flex-direction: column;
}

.result-card {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.result-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem;
  color: #909399;
  text-align: center;
  border: 2px dashed #d9d9d9;
  border-radius: 8px;
  min-height: 300px;
}

.result-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.result-text {
  font-size: 1rem;
}

.result-content {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* 质量指标 */
.quality-section {
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid #ebeef5;
}

.quality-title {
  font-size: 1.125rem;
  margin-bottom: 1rem;
  color: #303133;
}

.quality-metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
}

.metric-item {
  background-color: #f5f7fa;
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
}

.metric-label {
  font-size: 0.875rem;
  color: #606266;
  margin-bottom: 0.5rem;
}

.metric-value {
  font-size: 1.5rem;
  font-weight: 600;
  color: #0056b3;
  margin-bottom: 0.25rem;
}

.metric-desc {
  font-size: 0.75rem;
  color: #909399;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .repair-content {
    grid-template-columns: 1fr;
  }
  
  .left-panel, .right-panel {
    width: 100%;
  }
}
</style>