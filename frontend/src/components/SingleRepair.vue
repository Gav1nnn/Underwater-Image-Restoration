<script setup>
import { ref, reactive } from 'vue'
import axios from 'axios'
import ImageCompare from './ImageCompare.vue'
import { ElMessage } from 'element-plus'

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
    ElMessage.error('请上传有效的图像文件')
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
    ElMessage.warning('请先上传图像')
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
    const result = await axios.post('/api/repair', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    // 处理响应
    if (result.data && result.data.repaired_url) {
      // 兼容相对路径
      const url = result.data.repaired_url
      repairedImage.value = url.startsWith('http') ? url : `/api${url.replace('/api', '')}`
      qualityMetrics.value = result.data.quality_metrics
      ElMessage.success('修复完成')
    }
    
  } catch (error) {
    console.error('修复图像失败:', error)
    ElMessage.error(`修复失败: ${error.message || '未知错误'}`)
  } finally {
    isProcessing.value = false
  }
}

// 下载修复后的图像
const downloadImage = async () => {
  if (!repairedImage.value) return
  
  try {
    const response = await fetch(repairedImage.value)
    const blob = await response.blob()
    const blobUrl = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = `repaired_${Date.now()}.jpg`
    link.click()
    URL.revokeObjectURL(blobUrl)
  } catch (error) {
    ElMessage.error('下载图片失败')
  }
}
</script>

<template>
  <div class="single-repair-container">
    <h2 class="section-title">单张图像修复</h2>
    
    <div class="repair-content">
      <div class="left-panel">
        <el-card class="upload-card">
          <template #header>
            <div class="card-header">
              <span>图像上传</span>
              <el-button 
                v-if="originalImage" 
                type="danger" 
                link 
                @click="originalImage = null; repairedImage = null"
              >
                清除
              </el-button>
            </div>
          </template>
          
          <el-upload
            class="image-uploader"
            action="#"
            :auto-upload="false"
            :on-change="handleImageUpload"
            :show-file-list="false"
            accept="image/*"
            drag
          >
            <div v-if="!originalImage" class="upload-placeholder">
              <el-icon class="upload-icon"><Plus /></el-icon>
              <div class="upload-text">点击或拖拽上传图像</div>
              <div class="upload-hint">支持 JPG、PNG 等格式</div>
            </div>
            <img v-else :src="originalImage" class="uploaded-image" />
          </el-upload>
        </el-card>
        
        <transition name="el-zoom-in-top">
          <el-card class="params-card" v-if="originalImage">
            <template #header>
              <div class="card-header">
                <span>修复参数</span>
              </div>
            </template>
            
            <el-form :model="repairParams" label-width="60px" size="small" label-position="top">
              <el-form-item label="模型名称">
                <el-select v-model="repairParams.model_name" placeholder="选择模型" class="full-width">
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
                  class="full-width"
                  controls-position="right"
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
                  class="submit-btn"
                >
                  <el-icon v-if="isProcessing"><Loading /></el-icon>
                  {{ isProcessing ? '正在处理中...' : '开始修复' }}
                </el-button>
              </el-form-item>
            </el-form>
          </el-card>
        </transition>
      </div>
      
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
            <div class="result-text">等待任务开始...</div>
            <div class="result-hint">请在左侧添加图片，设置参数后点击开始</div>
          </div>
          
          <div v-else class="result-content">
            <ImageCompare
              :original-image="originalImage"
              :repaired-image="repairedImage"
            />
            
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
/* --- 布局容器 --- */
.single-repair-container {
  width: 100%;
}

.section-title {
  font-size: 1.5rem;
  margin-bottom: 2rem;
  color: #2c3e50;
  text-align: center;
  font-weight: 700;
  position: relative;
  padding-bottom: 10px;
}

.section-title::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 60px;
  height: 4px;
  background: linear-gradient(90deg, #0066cc, #00b4d8);
  border-radius: 2px;
}

.repair-content {
  display: grid;
  grid-template-columns: 350px 1fr;
  gap: 2rem;
  align-items: start;
}

/* --- 左侧面板 --- */
.left-panel {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* --- 卡片通用样式 --- */
.upload-card, .params-card, .result-card {
  border-radius: 12px;
  border: none;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  color: #2c3e50;
}

/* --- 上传区域 --- */
.image-uploader {
  width: 100%;
  text-align: center;
}

.image-uploader :deep(.el-upload) {
  width: 100%;
  display: block;
}

.image-uploader :deep(.el-upload-dragger) {
  padding: 0;
  border: none;
  background: transparent;
}

/* --- src/components/BatchRepair.vue 的 style 区域 --- */

.upload-placeholder {
  border: 2px dashed #a0cfff;
  background-color: #f0f7ff;
  border-radius: 16px;
  padding: 3rem 1rem; /* 保持原有的内边距 */
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 250px;
  
  /* 【关键修改点】添加 2px 外边距，防止上浮时顶部被切掉 */
  margin: 2px; 
}

/* 保持 hover 样式不变 */
.upload-placeholder:hover {
  border-color: #0066cc;
  background-color: #e6f1fc;
  transform: translateY(-2px); /* 因为有了 margin，现在上浮不会消失了 */
  box-shadow: 0 8px 16px rgba(0, 102, 204, 0.1);
}

.upload-icon {
  font-size: 3.5rem;
  color: #409eff;
  margin-bottom: 1rem;
  transition: transform 0.4s;
}

.upload-placeholder:hover .upload-icon {
  transform: scale(1.1) rotate(10deg);
  color: #0066cc;
}

.upload-text {
  font-size: 1rem;
  color: #506070;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.upload-hint {
  font-size: 0.8rem;
  color: #909399;
}

.uploaded-image {
  max-width: 100%;
  max-height: 300px;
  border-radius: 12px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
  display: block;
  margin: 0 auto;
}

/* --- 参数表单 --- */
.full-width { width: 100%; }
.submit-btn { width: 100%; font-weight: 600; letter-spacing: 1px; }

/* --- 右侧结果 --- */
.result-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 600px;
}

.result-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem;
  color: #909399;
  text-align: center;
  border: 2px dashed #e0e6ed;
  border-radius: 16px;
  background: #fafafa;
  flex: 1;
}

.result-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
  color: #cbd5e0;
}

/* 【新增】提示文字样式，与BatchRepair统一 */
.result-text {
  font-size: 1rem;
  color: #909399;
  margin-bottom: 0.5rem;
}

.result-hint {
  font-size: 0.85rem;
  color: #c0c4cc;
}

.result-content {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

/* --- 质量指标 --- */
.quality-section {
  margin-top: 1rem;
  padding-top: 1.5rem;
  border-top: 1px solid #edf2f7;
}

.quality-title {
  font-size: 1.1rem;
  margin-bottom: 1.2rem;
  color: #2c3e50;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.quality-title::before {
  content: '';
  display: block;
  width: 4px;
  height: 16px;
  background: #0066cc;
  border-radius: 2px;
}

.quality-metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1.2rem;
}

.metric-item {
  background: #ffffff;
  padding: 1.2rem;
  border-radius: 12px;
  text-align: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
  border: 1px solid #f0f4f8;
  transition: all 0.3s ease;
}

.metric-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 25px rgba(0, 102, 204, 0.15);
  border-color: #d9ecff;
}

.metric-label {
  font-size: 0.85rem;
  color: #64748b;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.metric-value {
  font-size: 1.6rem;
  font-weight: 700;
  background: linear-gradient(135deg, #0066cc 0%, #00b4d8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.25rem;
}

.metric-desc {
  font-size: 0.75rem;
  color: #94a3b8;
}

/* --- 响应式适配 --- */
@media (max-width: 1024px) {
  .repair-content {
    grid-template-columns: 1fr;
  }
  
  .left-panel, .right-panel {
    width: 100%;
  }

  .result-card {
    min-height: auto;
  }
}
</style>