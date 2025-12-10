<script setup>
import { ref } from 'vue'
import SingleRepair from './components/SingleRepair.vue'
import BatchRepair from './components/BatchRepair.vue'

const activeTab = ref('single')
</script>

<template>
  <div class="app-container">
    <!-- 导航栏 -->
    <header class="app-header">
      <h1 class="app-title">🌊 水下图像修复系统</h1>
      <nav class="app-nav">
        <el-menu
          :default-active="activeTab"
          mode="horizontal"
          @select="(key) => activeTab = key"
          background-color="#0056b3"
          text-color="#fff"
          active-text-color="#ffd700"
        >
          <el-menu-item index="single">单张修复</el-menu-item>
          <el-menu-item index="batch">批量修复</el-menu-item>
        </el-menu>
      </nav>
    </header>

    <!-- 主要内容区域 -->
    <main class="app-main">
      <div class="content-wrapper">
        <!-- 单张修复 -->
        <SingleRepair v-if="activeTab === 'single'" />
        <!-- 批量修复 -->
        <BatchRepair v-else />
      </div>
    </main>

    <!-- 页脚 -->
    <footer class="app-footer">
      <p>&copy; 2025 水下图像修复系统 | 基于 Sea-Thru + Monodepth2 算法</p>
    </footer>
  </div>
</template>

<style>
/* 全局样式重置 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background-color: #f5f7fa;
  color: #333;
}

/* 应用容器 */
.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 导航栏 */
.app-header {
  background-color: #0056b3;
  color: white;
  padding: 1rem 2rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.app-title {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
  text-align: center;
}

.app-nav {
  max-width: 1200px;
  margin: 0 auto;
}

/* 主要内容区域 */
.app-main {
  flex: 1;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

.content-wrapper {
  background: white;
  border-radius: 8px;
  padding: 2rem;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
}

/* 页脚 */
.app-footer {
  background-color: #f0f2f5;
  color: #606266;
  text-align: center;
  padding: 1rem;
  margin-top: auto;
  font-size: 0.9rem;
}
</style>
