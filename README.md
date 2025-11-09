# 未来搭子 AI陪伴智能体

24小时温暖陪伴的智能AI助手，提供个性化、情感化的对话体验。

## ✨ 核心功能

- **情感理解**：识别28种情绪类别，深度理解用户情绪状态
- **长期记忆**：记住用户的重要信息（行程、偏好、事件），在对话中自然提及
- **角色自适应**：7种角色类型、6种沟通风格、4种语调特点，动态切换
- **智能对话**：支持文本和图片输入，生成个性化回复

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `env.example` 文件为 `.env` 并填入你的配置：

```bash
cp env.example .env
```

然后编辑 `.env` 文件，填入你的 Gemini API 密钥：

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. 运行应用

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问应用

- 聊天界面：http://localhost:8000/chat
- 监控面板：http://localhost:8000/monitor

## 📋 系统要求

- Python 3.8+
- 至少 4GB RAM（用于运行AI模型）
- 网络连接（用于下载预训练模型和调用Gemini API）

## 📝 注意事项

1. **首次运行**：系统会自动下载所需的预训练模型到 `models_cache/` 目录
2. **数据库**：首次运行会自动创建 SQLite 数据库和 Qdrant 向量数据库
3. **API密钥**：需要在 `.env` 文件中配置 Gemini API 密钥

## 🏗️ 技术栈

- **后端框架**：FastAPI
- **AI模型**：Gemini 2.5 Flash, GoEmotions
- **数据库**：SQLite, Qdrant
- **工作流引擎**：LangGraph
- **向量嵌入**：SentenceTransformer

## 📄 许可证

MIT License

