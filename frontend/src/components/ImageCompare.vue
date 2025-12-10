<script setup>
import { ref, onMounted, watch } from 'vue'

// 组件属性
const props = defineProps({
  originalImage: {
    type: String,
    required: true
  },
  repairedImage: {
    type: String,
    required: true
  }
})

// 状态管理
const isDragging = ref(false)
const sliderPosition = ref(50) // 滑块位置，百分比
const containerWidth = ref(0)
const containerHeight = ref(0)

// 原始图像加载完成事件
const originalImageLoaded = (event) => {
  updateContainerSize(event.target)
}

// 更新容器尺寸
const updateContainerSize = (imgElement) => {
  containerWidth.value = imgElement.naturalWidth
  containerHeight.value = imgElement.naturalHeight
}

// 鼠标按下事件
const handleMouseDown = () => {
  isDragging.value = true
}

// 鼠标移动事件
const handleMouseMove = (event) => {
  if (!isDragging.value) return
  
  const container = event.currentTarget
  const rect = container.getBoundingClientRect()
  const x = event.clientX - rect.left
  const position = (x / rect.width) * 100
  
  // 限制滑块位置在0-100之间
  sliderPosition.value = Math.max(0, Math.min(100, position))
}

// 鼠标释放事件
const handleMouseUp = () => {
  isDragging.value = false
}

// 监听图像变化
watch([() => props.originalImage, () => props.repairedImage], () => {
  // 重置滑块位置
  sliderPosition.value = 50
})

// 组件挂载完成
onMounted(() => {
  // 添加全局鼠标事件监听器
  document.addEventListener('mouseup', handleMouseUp)
  document.addEventListener('mousemove', handleMouseMove)
})

// 组件卸载前清理
const cleanup = () => {
  document.removeEventListener('mouseup', handleMouseUp)
  document.removeEventListener('mousemove', handleMouseMove)
}

defineExpose({
  cleanup
})
</script>

<template>
  <div class="image-compare-container">
    <h3 class="compare-title">修复前后对比</h3>
    
    <div 
      class="compare-wrapper"
      @mousedown="handleMouseDown"
      @mousemove="handleMouseMove"
    >
      <!-- 原始图像 -->
      <div class="image-container original">
        <img 
          :src="originalImage" 
          class="compare-image" 
          @load="originalImageLoaded"
          alt="原始图像"
          @click.stop
          draggable="false"
        />
        <div class="image-label">原始图像</div>
      </div>
      
      <!-- 修复后图像 -->
      <div 
        class="image-container repaired"
        :style="{
          clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`
        }"
      >
        <img 
          :src="repairedImage" 
          class="compare-image"
          alt="修复后图像"
          @click.stop
          draggable="false"
        />
        <div class="image-label">修复后图像</div>
      </div>
      
      <!-- 滑块 -->
      <div 
        class="slider"
        :style="{
          left: `${sliderPosition}%`
        }"
      >
        <div class="slider-handle">
          <el-icon class="slider-icon"><SwitchButton /></el-icon>
        </div>
        <div class="slider-line"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.image-compare-container {
  width: 100%;
  margin: 0 auto;
}

.compare-title {
  font-size: 1rem;
  color: #303133;
  margin-bottom: 1rem;
  text-align: center;
  font-weight: 600;
}

.compare-wrapper {
  position: relative;
  width: 100%;
  max-width: 600px;
  margin: 0 auto;
  cursor: col-resize;
  background-color: #f0f0f0;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.image-container {
  position: relative;
  width: 100%;
  height: auto;
  overflow: hidden;
}

.original {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1;
}

.repaired {
  position: relative;
  z-index: 2;
}

.compare-image {
  display: block;
  width: 100%;
  height: auto;
  max-height: 500px;
  object-fit: contain;
  user-select: none;
}

.image-label {
  position: absolute;
  bottom: 10px;
  padding: 5px 15px;
  background-color: rgba(0, 0, 0, 0.6);
  color: white;
  font-size: 0.875rem;
  font-weight: 500;
  border-radius: 4px;
  z-index: 10;
}

.original .image-label {
  left: 10px;
}

.repaired .image-label {
  right: 10px;
}

.slider {
  position: absolute;
  top: 0;
  height: 100%;
  width: 2px;
  background-color: rgba(255, 255, 255, 0.8);
  z-index: 100;
  transform: translateX(-50%);
  transition: all 0.1s ease;
}

.slider-line {
  position: absolute;
  top: 0;
  left: 50%;
  width: 2px;
  height: 100%;
  background-color: #0056b3;
  transform: translateX(-50%);
  box-shadow: 0 0 10px rgba(0, 86, 179, 0.5);
}

.slider-handle {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 36px;
  height: 36px;
  background-color: white;
  border: 3px solid #0056b3;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  cursor: col-resize;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
}

.slider-handle:hover {
  background-color: #f0f8ff;
  transform: translate(-50%, -50%) scale(1.1);
  box-shadow: 0 4px 15px rgba(0, 86, 179, 0.4);
}

.slider-icon {
  font-size: 1.25rem;
  color: #0056b3;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .compare-wrapper {
    max-width: 100%;
  }
  
  .compare-image {
    max-height: 300px;
  }
}
</style>