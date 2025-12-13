<script setup>
import { ref, reactive } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

// --- 状态管理 ---
const isUploading = ref(false)
const isProcessing = ref(false)
const uploadedFiles = ref([]) // 文件列表状态
const repairResults = ref([])
const batchProgress = ref(0)
const currentTaskId = ref(null)

// --- 修复参数 (与 SingleRepair 完全一致) ---
const repairParams = reactive({
  model_name: 'mono_1024x320',
  size: 1024,
  depth_scale: 10.0,
  depth_offset: 2.0,
  save_depth: false,
  save_intermediate: false
})

// --- 文件处理逻辑 ---
const handleFileChange = (uploadFile, uploadFiles) => {
  if (uploadFile.raw) {
    if (!uploadFile.raw.type.startsWith('image/')) {
      ElMessage.warning(`${uploadFile.name} 不是有效的图像文件`)
      const index = uploadedFiles.findIndex(f => f.uid === uploadFile.uid)
      if (index !== -1) uploadedFiles.splice(index, 1)
      return
    }
  }
}

const clearUploadedFiles = () => {
  uploadedFiles.value = []
  repairResults.value = []
  batchProgress.value = 0
}

const repairImages = async () => {
  if (uploadedFiles.value.length === 0) {
    ElMessage.warning('请先上传图像')
    return
  }
  
  try {
    isProcessing.value = true
    repairResults.value = []
    batchProgress.value = 0
    
    const formData = new FormData()
    uploadedFiles.value.forEach(file => {
      formData.append('files', file.raw, file.name)
    })
    Object.entries(repairParams).forEach(([key, value]) => {
      formData.append(key, value.toString())
    })
    
    // 发送请求
    const result = await axios.post('/api/repair/batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    
    currentTaskId.value = result.data.task_id
    await checkTaskStatus(result.data.task_id)
    
  } catch (error) {
    console.error('批量修复失败:', error)
    ElMessage.error(`批量修复失败: ${error.message || '未知错误'}`)
    isProcessing.value = false
  }
}

const checkTaskStatus = async (taskId) => {
  try {
    const result = await axios.get(`/api/task/${taskId}`)
    
    if (result.data.status === 'completed') {
      repairResults.value = result.data.results
      batchProgress.value = 100
      isProcessing.value = false
      ElMessage.success(`批量处理完成，共 ${result.data.results.length} 张`)
    } else if (result.data.status === 'failed') {
      ElMessage.error(`任务失败: ${result.data.error}`)
      isProcessing.value = false
    } else {
      batchProgress.value = result.data.progress
      setTimeout(() => checkTaskStatus(taskId), 1000)
    }
  } catch (error) {
    isProcessing.value = false
  }
}

const downloadRepairedImage = async (result) => {
  if (!result.repaired_url) return
  try {
    const imageUrl = result.repaired_url.startsWith('http') 
      ? result.repaired_url 
      : `/api${result.repaired_url.replace('/api', '')}`

    const response = await fetch(imageUrl)
    const blob = await response.blob()
    const blobUrl = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = `repaired_${Date.now()}.jpg`
    link.click()
    URL.revokeObjectURL(blobUrl)
  } catch (error) {
    ElMessage.error('下载失败')
  }
}

const downloadAllResults = () => {
  repairResults.value.forEach((result, index) => {
    setTimeout(async () => {
      await downloadRepairedImage(result)
    }, index * 200)
  })
}
</script>

<template>
  <div class="batch-repair-container">
    <h2 class="section-title">批量图像修复</h2>
    
    <div class="repair-content">
      <div class="left-panel">
        <el-card class="upload-card">
          <template #header>
            <div class="card-header">
              <span>图像队列 ({{ uploadedFiles.length }})</span>
              <el-button 
                type="danger" 
                link 
                @click="clearUploadedFiles"
                :disabled="uploadedFiles.length === 0"
              >
                清空
              </el-button>
            </div>
          </template>
          
          <el-upload
            v-model:file-list="uploadedFiles"
            class="image-uploader"
            action="#"
            :auto-upload="false"
            :on-change="handleFileChange"
            :multiple="true"
            accept="image/*"
            drag
            list-type="picture"
          >
            <div class="upload-placeholder">
              <el-icon class="upload-icon"><Plus /></el-icon>
              <div class="upload-text">点击或拖拽上传图像</div>
              <div class="upload-hint">支持批量上传 JPG、PNG 格式</div>
            </div>
          </el-upload>
        </el-card>
        
        <transition name="el-zoom-in-top">
          <el-card class="params-card" v-if="uploadedFiles.length > 0">
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

        <transition name="el-fade-in">
          <el-card v-if="isProcessing || batchProgress > 0" class="progress-card">
            <div class="progress-info">
              <span>处理进度</span>
              <span>{{ batchProgress }}%</span>
            </div>
            <el-progress :percentage="batchProgress" :show-text="false" status="success" />
          </el-card>
        </transition>
      </div>
      
      <div class="right-panel">
        <el-card class="result-card">
          <template #header>
            <div class="card-header">
              <span>处理结果</span>
              <el-button
                v-if="repairResults.length > 0"
                type="success"
                size="small"
                @click="downloadAllResults"
              >
                <el-icon><Download /></el-icon>
                批量下载
              </el-button>
            </div>
          </template>
          
          <div v-if="repairResults.length === 0" class="result-placeholder">
            <el-icon class="result-icon"><Files /></el-icon>
            <div class="result-text">等待任务开始...</div>
            <div class="result-hint">请在左侧添加图片，设置参数后点击开始</div>
          </div>
          
          <div v-else class="results-grid">
            <div 
              v-for="(result, index) in repairResults" 
              :key="index"
              class="grid-item"
            >
              <div class="image-wrapper">
                <img 
                  :src="result.repaired_url.startsWith('http') ? result.repaired_url : `/api${result.repaired_url.replace('/api', '')}`" 
                  loading="lazy"
                />
                <div class="overlay">
                  <el-button circle type="primary" @click="downloadRepairedImage(result)">
                    <el-icon><Download /></el-icon>
                  </el-button>
                </div>
              </div>
              <div class="metrics-mini">
                <span>PSNR: {{ result.quality_metrics.psnr.toFixed(1) }}</span>
                <span>SSIM: {{ result.quality_metrics.ssim.toFixed(2) }}</span>
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
.batch-repair-container { width: 100%; }

.section-title {
  font-size: 1.5rem; margin-bottom: 2rem;
  color: #2c3e50; text-align: center; font-weight: 700;
  position: relative; padding-bottom: 10px;
}
.section-title::after {
  content: ''; position: absolute; bottom: 0; left: 50%;
  transform: translateX(-50%); width: 60px; height: 4px;
  background: linear-gradient(90deg, #0066cc, #00b4d8);
  border-radius: 2px;
}

/* Grid 布局：严格复刻 SingleRepair (左 350px) */
.repair-content {
  display: grid;
  grid-template-columns: 350px 1fr;
  gap: 2rem; align-items: start;
}

.left-panel { display: flex; flex-direction: column; gap: 1.5rem; }
.right-panel { display: flex; flex-direction: column; height: 100%; }

/* --- 卡片样式 --- */
.upload-card, .params-card, .result-card, .progress-card {
  border-radius: 12px; border: none;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}
.card-header {
  display: flex; justify-content: space-between;
  align-items: center; font-weight: 600; color: #2c3e50;
}

/* --- 上传区域 (严格复刻 SingleRepair 的 CSS) --- */
.image-uploader { width: 100%; text-align: center; }

.image-uploader :deep(.el-upload) { width: 100%; display: block; }
.image-uploader :deep(.el-upload-dragger) { padding: 0; border: none; background: transparent; }

/* 你的 SingleRepair 用的样式 */
.upload-placeholder {
  border: 2px dashed #a0cfff; /* 浅蓝虚线 */
  background-color: #f0f7ff;  /* 浅蓝背景 */
  border-radius: 16px;
  padding: 3rem 1rem;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 250px; /* 保持与 SingleRepair 一样的高度 */
  
  /* 【关键】添加与 SingleRepair 相同的 margin */
  margin: 2px;
}

.upload-placeholder:hover {
  border-color: #0066cc;
  background-color: #e6f1fc;
  transform: translateY(-2px);
  box-shadow: 0 8px 16px rgba(0, 102, 204, 0.1);
}

.upload-icon {
  font-size: 3.5rem; color: #409eff;
  margin-bottom: 1rem;
  transition: transform 0.4s;
}

.upload-placeholder:hover .upload-icon {
  transform: scale(1.1) rotate(10deg);
  color: #0066cc;
}

.upload-text {
  font-size: 1rem; color: #506070; margin-bottom: 0.5rem; font-weight: 500;
}
.upload-hint { font-size: 0.8rem; color: #909399; }

/* --- 按钮样式 (与 SingleRepair 保持一致) --- */
.full-width { width: 100%; }
.submit-btn { width: 100%; font-weight: 600; letter-spacing: 1px; }

/* --- 右侧结果 (Grid 布局) --- */
.result-card { min-height: 600px; }

.result-placeholder {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 4rem; color: #909399; text-align: center;
  border: 2px dashed #e0e6ed; border-radius: 16px;
  background: #fafafa; flex: 1;
}
.result-icon { font-size: 4rem; margin-bottom: 1rem; color: #cbd5e0; }
.result-text { font-size: 1rem; color: #909399; margin-bottom: 0.5rem; }
.result-hint { font-size: 0.85rem; color: #c0c4cc; }

/* Grid 结果网格 */
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1.5rem; padding: 1rem 0;
}
.grid-item {
  background: white; border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  transition: transform 0.3s ease;
  border: 1px solid #f0f0f0;
}
.grid-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 20px rgba(0, 50, 100, 0.15);
}
.image-wrapper {
  position: relative; aspect-ratio: 4/3;
  background: #f5f7fa; overflow: hidden;
}
.image-wrapper img { width: 100%; height: 100%; object-fit: cover; }
.overlay {
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.4);
  display: flex; align-items: center; justify-content: center;
  opacity: 0; transition: opacity 0.3s;
}
.image-wrapper:hover .overlay { opacity: 1; }
.metrics-mini {
  padding: 0.8rem; display: flex;
  justify-content: space-between; font-size: 0.75rem;
  background: #fff; color: #606266;
  border-top: 1px solid #f0f0f0;
}
.progress-info {
  display: flex; justify-content: space-between; margin-bottom: 8px; color: #606266;
}

@media (max-width: 1024px) {
  .repair-content { grid-template-columns: 1fr; }
}
</style>