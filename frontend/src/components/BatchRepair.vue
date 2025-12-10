<script setup>
import { ref, reactive } from 'vue'
import axios from 'axios'

// 状态管理
const isUploading = ref(false)
const isProcessing = ref(false)
const uploadedFiles = ref([])
const repairResults = ref([])
const errorMessage = ref('')
const batchProgress = ref(0)
const currentTaskId = ref(null)

// 修复参数
const repairParams = reactive({
  model_name: 'mono_1024x320',
  size: 1024,
  depth_scale: 10.0,
  depth_offset: 2.0,
  save_depth: false,
  save_intermediate: false
})

// 上传图像 - 处理单个文件添加
const handleImageUpload = (uploadFile) => {
  if (uploadFile.raw) {
    isUploading.value = true
    uploadedFiles.value = [...uploadedFiles.value, uploadFile]
    isUploading.value = false
  }
}

// 上传图像 - 处理文件列表变化
const handleFileListChange = (newFiles) => {
  uploadedFiles.value = newFiles
}

// 移除已上传的图像
const removeUploadedFile = (index) => {
  uploadedFiles.value.splice(index, 1)
}

// 清空已上传的图像
const clearUploadedFiles = () => {
  uploadedFiles.value = []
  repairResults.value = []
  errorMessage.value = ''
}

// 修复图像
const repairImages = async () => {
  if (uploadedFiles.value.length === 0) {
    errorMessage.value = '请先上传图像'
    return
  }
  
  try {
    isProcessing.value = true
    errorMessage.value = ''
    repairResults.value = []
    batchProgress.value = 0
    
    // 创建FormData
    const formData = new FormData()
    
    // 添加图像文件
    uploadedFiles.value.forEach(file => {
      formData.append('files', file.raw, file.name)
    })
    
    // 添加修复参数
    Object.entries(repairParams).forEach(([key, value]) => {
      formData.append(key, value.toString())
    })
    
    // 发送请求
    const result = await axios.post('http://localhost:8000/api/repair/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    // 保存任务ID
    currentTaskId.value = result.data.task_id
    
    // 查询任务状态
    await checkTaskStatus(result.data.task_id)
    
  } catch (error) {
    console.error('批量修复失败:', error)
    errorMessage.value = `批量修复失败: ${error.message || '未知错误'}`
    isProcessing.value = false
  }
}

// 查询任务状态
const checkTaskStatus = async (taskId) => {
  try {
    const result = await axios.get(`http://localhost:8000/api/task/${taskId}`)
    
    if (result.data.status === 'completed') {
      // 任务完成
      repairResults.value = result.data.results
      batchProgress.value = 100
      isProcessing.value = false
    } else if (result.data.status === 'failed') {
      // 任务失败
      errorMessage.value = `批量修复失败: ${result.data.error || '未知错误'}`
      isProcessing.value = false
    } else {
      // 任务进行中，继续查询
      batchProgress.value = result.data.progress
      setTimeout(() => {
        checkTaskStatus(taskId)
      }, 1000)
    }
  } catch (error) {
    console.error('查询任务状态失败:', error)
    errorMessage.value = `查询任务状态失败: ${error.message || '未知错误'}`
    isProcessing.value = false
  }
}

// 下载修复后的图像
const downloadRepairedImage = async (result) => {
  if (!result.repaired_url) return
  
  try {
    const imageUrl = `http://localhost:8000${result.repaired_url}`
    
    // 使用fetch获取图片数据
    const response = await fetch(imageUrl)
    const blob = await response.blob()
    
    // 创建Blob URL
    const blobUrl = URL.createObjectURL(blob)
    
    // 创建下载链接
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = `repaired_${Date.now()}.jpg`
    
    // 触发下载
    link.click()
    
    // 释放Blob URL
    URL.revokeObjectURL(blobUrl)
  } catch (error) {
    console.error('下载图片失败:', error)
  }
}

// 批量下载修复后的图像
const downloadAllResults = () => {
  repairResults.value.forEach((result, index) => {
    setTimeout(async () => {
      await downloadRepairedImage(result)
    }, index * 100) // 延迟下载，避免浏览器阻塞
  })
}
</script>

<template>
  <div class="batch-repair-container">
    <h2 class="section-title">批量图像修复</h2>
    
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
    
    <div class="batch-content">
      <!-- 左侧：上传和参数设置 -->
      <div class="left-panel">
        <!-- 图像上传 -->
        <el-card class="upload-card">
          <template #header>
            <div class="card-header">
              <span>图像上传</span>
              <el-button
                type="danger"
                size="small"
                @click="clearUploadedFiles"
                :disabled="uploadedFiles.length === 0"
              >
                清空
              </el-button>
            </div>
          </template>
          
          <el-upload
            class="batch-uploader"
            action="#"
            :auto-upload="false"
            :on-change="handleImageUpload"
            :file-list="uploadedFiles"
            :on-update:file-list="handleFileListChange"
            :multiple="true"
            :limit="10"
            accept="image/*"
            list-type="picture-card"
          >
            <div v-if="uploadedFiles.length < 10">
              <el-icon class="upload-icon"><Plus /></el-icon>
              <div class="upload-text">上传图像</div>
            </div>
          </el-upload>
        </el-card>
        
        <!-- 修复参数设置 -->
        <el-card class="params-card" v-if="uploadedFiles.length > 0">
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
                @click="repairImages"
                :disabled="isProcessing || uploadedFiles.length === 0"
                size="large"
              >
                <el-icon v-if="isProcessing"><Loading /></el-icon>
                开始批量修复
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
        
        <!-- 进度显示 -->
        <el-card class="progress-card" v-if="isProcessing">
          <template #header>
            <div class="card-header">
              <span>处理进度</span>
            </div>
          </template>
          
          <div class="progress-content">
            <el-progress
              :percentage="batchProgress"
              :status="batchProgress === 100 ? 'success' : 'active'"
              :stroke-width="20"
              format="percentage"
            />
            <div class="progress-text">
              正在处理 {{ uploadedFiles.length }} 张图像...
            </div>
          </div>
        </el-card>
      </div>
      
      <!-- 右侧：结果展示 -->
      <div class="right-panel">
        <el-card class="results-card">
          <template #header>
            <div class="card-header">
              <span>修复结果</span>
              <el-button
                v-if="repairResults.length > 0"
                type="success"
                size="small"
                @click="downloadAllResults"
              >
                批量下载
              </el-button>
            </div>
          </template>
          
          <div v-if="repairResults.length === 0" class="results-placeholder">
            <el-icon class="results-icon"><Picture /></el-icon>
            <div class="results-text">修复结果将显示在这里</div>
          </div>
          
          <div v-else class="results-grid">
            <div 
              v-for="(result, index) in repairResults" 
              :key="index"
              class="result-item"
            >
              <el-card class="result-card">
                <template #header>
                  <div class="result-card-header">
                    <span class="result-index">#{{ index + 1 }}</span>
                    <el-button
                      type="success"
                      size="small"
                      @click="downloadRepairedImage(result)"
                    >
                      下载
                    </el-button>
                  </div>
                </template>
                
                <div class="result-content">
                  <!-- 修复后图像 -->
                  <div class="result-images">
                    <img 
                      :src="`http://localhost:8000${result.repaired_url}`" 
                      class="repaired-image"
                      alt="修复后图像"
                      @click.stop
                      draggable="false"
                    />
                  </div>
                  
                  <!-- 质量指标 -->
                  <div class="quality-metrics">
                    <div class="metric-item">
                      <span class="metric-label">PSNR:</span>
                      <span class="metric-value">{{ result.quality_metrics.psnr.toFixed(2) }}</span>
                    </div>
                    <div class="metric-item">
                      <span class="metric-label">SSIM:</span>
                      <span class="metric-value">{{ result.quality_metrics.ssim.toFixed(4) }}</span>
                    </div>
                    <div class="metric-item">
                      <span class="metric-label">MSE:</span>
                      <span class="metric-value">{{ result.quality_metrics.mse.toFixed(2) }}</span>
                    </div>
                  </div>
                </div>
              </el-card>
            </div>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<style scoped>
.batch-repair-container {
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

.batch-content {
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

.upload-card, .params-card, .progress-card, .results-card {
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
.batch-uploader {
  width: 100%;
  text-align: center;
}

.upload-icon {
  font-size: 2rem;
  color: #0056b3;
  margin-bottom: 0.5rem;
}

.upload-text {
  font-size: 0.875rem;
  color: #606266;
}

.upload-list {
  margin-top: 1rem;
}

/* 参数表单 */
.params-card {
  margin-top: auto;
}

/* 进度卡片 */
.progress-content {
  text-align: center;
  padding: 1rem;
}

.progress-text {
  margin-top: 1rem;
  color: #606266;
}

/* 右侧面板 */
.right-panel {
  display: flex;
  flex-direction: column;
}

.results-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  max-height: calc(100vh - 200px);
  overflow: hidden;
}

.results-placeholder {
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

.results-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.results-text {
  font-size: 1rem;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1rem;
  overflow-y: auto;
  max-height: calc(100vh - 300px);
  padding: 0.5rem;
}

.result-item {
  display: flex;
  flex-direction: column;
}

.result-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.result-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  font-size: 0.875rem;
}

.result-index {
  color: #0056b3;
}

.result-content {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.result-images {
  width: 100%;
  height: 150px;
  overflow: hidden;
  border-radius: 4px;
  background-color: #f0f0f0;
}

.repaired-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.repaired-image:hover {
  transform: scale(1.05);
}

.quality-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  font-size: 0.875rem;
}

.metric-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: #f5f7fa;
  padding: 0.5rem;
  border-radius: 4px;
}

.metric-label {
  color: #606266;
  font-size: 0.75rem;
  margin-bottom: 0.25rem;
}

.metric-value {
  color: #0056b3;
  font-weight: 600;
  font-size: 0.875rem;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .batch-content {
    grid-template-columns: 1fr;
  }
  
  .left-panel, .right-panel {
    width: 100%;
  }
  
  .results-grid {
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  }
}
</style>