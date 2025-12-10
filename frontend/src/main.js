import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import axios from 'axios'
import './style.css'
import App from './App.vue'

const app = createApp(App)

// 配置Axios
app.config.globalProperties.$axios = axios

// 配置Element Plus
app.use(ElementPlus)

app.mount('#app')
