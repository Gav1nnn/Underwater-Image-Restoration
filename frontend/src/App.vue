<script setup>
import { ref } from 'vue'
import SingleRepair from './components/SingleRepair.vue'
import BatchRepair from './components/BatchRepair.vue'

const activeTab = ref('single')
</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-content">
        <div class="logo-area">
          <span class="logo-icon">🌊</span>
          <h1 class="app-title">DeepSee <span class="subtitle">水下图像修复系统</span></h1>
        </div>
        
        <nav class="app-nav">
          <el-menu
            :default-active="activeTab"
            mode="horizontal"
            @select="(key) => activeTab = key"
            class="transparent-menu"
            :ellipsis="false"
          >
            <el-menu-item index="single">
              <el-icon><Monitor /></el-icon>单张精修
            </el-menu-item>
            <el-menu-item index="batch">
              <el-icon><Files /></el-icon>批量处理
            </el-menu-item>
          </el-menu>
        </nav>
      </div>
    </header>

    <main class="app-main">
      <transition name="fade" mode="out-in">
        <div class="content-wrapper">
          <keep-alive>
            <component :is="activeTab === 'single' ? SingleRepair : BatchRepair" />
          </keep-alive>
        </div>
      </transition>
    </main>

    <footer class="app-footer">
      <p>&copy; 2025 DeepSee Restoration System | 基于 Sea-Thru + Monodepth2 算法</p>
    </footer>
  </div>
</template>

<style scoped>
/* --- 全局容器布局 (粘性页脚核心) --- */
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh; /* 强制占满视口高度 */
  /* 这里不需要设背景，因为 main.js 引入的 style.css 里的 body 已经设了全局背景 */
}

/* --- 导航栏美化 --- */
.app-header {
  background: rgba(255, 255, 255, 0.85); /* 半透明白 */
  backdrop-filter: blur(12px);           /* 毛玻璃模糊 */
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); /* 柔和阴影 */
  padding: 0 2rem;
}

.header-content {
  max-width: 1280px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 70px;
}

.logo-area {
  display: flex;
  align-items: center;
  gap: 10px;
  user-select: none;
}

.logo-icon {
  font-size: 2rem;
  animation: float 3s ease-in-out infinite; /* 浮动动画 */
}

.app-title {
  font-size: 1.5rem;
  font-weight: 800;
  /* 渐变文字效果 */
  background: linear-gradient(120deg, #0066cc, #00b4d8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
}

.subtitle {
  font-size: 1.2rem;
  color: #666;
  font-weight: 400;
  margin-left: 8px;
  -webkit-text-fill-color: #666; /* 重置 subtitle 的颜色 */
}

/* --- 菜单透明化处理 --- */
.transparent-menu {
  background: transparent !important;
  border-bottom: none !important;
  width: 300px;
  justify-content: flex-end;
}

/* --- 主要内容区域 (自动填充剩余空间) --- */
.app-main {
  flex: 1; /* 关键：把 footer 挤到底部 */
  padding: 2rem;
  max-width: 1280px;
  margin: 0 auto;
  width: 100%;
  display: flex;          /* 确保子元素能撑开 */
  flex-direction: column;
}

.content-wrapper {
  flex: 1;
  width: 100%;
}

/* --- 页脚美化与居中 --- */
.app-footer {
  text-align: center;
  padding: 1.5rem;
  color: #909399;
  font-size: 0.85rem;
  letter-spacing: 0.5px;
  /* 页脚背景微调，使其融入整体 */
  background: rgba(255, 255, 255, 0.3);
  backdrop-filter: blur(5px);
  margin-top: auto; /* 双重保险，确保在底部 */
}

/* --- 动画效果 --- */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.fade-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

/* 浮动动画关键帧 */
@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

/* 响应式调整 */
@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    height: auto;
    padding: 1rem 0;
    gap: 1rem;
  }
  
  .transparent-menu {
    width: 100%;
    justify-content: center;
  }
}
</style>