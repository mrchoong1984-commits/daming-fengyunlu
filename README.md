# 大明风云录 Gemini 后端

这是为《大明风云录》手机网页准备的 Python/FastAPI 后端，调用 Google Gemini API。

## Render 配置
- Runtime / Language: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Environment Variables
在 Render 的 Environment 页面添加：
- `GEMINI_API_KEY` = 你的 Google AI Studio API 密钥（只填在服务器环境变量，不要提交到 GitHub）
- `GEMINI_MODEL` = `gemini-2.5-flash`

## 检查
部署完成后打开 `https://你的服务网址/health`，应该显示 `{"ok":true}`。
游戏接口为 `POST /api/turn`。
