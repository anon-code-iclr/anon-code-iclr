"""
gemini-3-flash-preview图片分析示例
使用 DMXAPI 调用 gemini-3-flash-preview 模型进行图片内容识别
"""

from google import genai
from google.genai import types

# ===========================================
# DMXAPI 配置
# ===========================================
API_KEY = "sk-*******************************************"
BASE_URL = "https://www.dmxapi.cn"

# 初始化客户端
client = genai.Client(
    api_key=API_KEY,
    http_options={'base_url': BASE_URL}
)

# ===========================================
# 读取图片
# ===========================================
# 支持的图片格式: JPEG, PNG, GIF, WebP
# 对应的 mime_type:
#   - JPEG: "image/jpeg"
#   - PNG:  "image/png"
#   - GIF:  "image/gif"
#   - WebP: "image/webp"

image_path = "test/example.jpg"  # 替换为你的图片路径
with open(image_path, "rb") as f:
    image_data = f.read()

# ===========================================
# 调用模型
# ===========================================
# media_resolution 参数说明:
#   - media_resolution_low:    280 tokens, 适合快速处理
#   - media_resolution_medium: 560 tokens, 平衡质量与速度
#   - media_resolution_high:   1120 tokens, 最佳质量, 适合细节识别

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Content(
            parts=[
                types.Part(text="这张图片里有什么?"),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=image_data
                    ),
                    media_resolution={"level": "media_resolution_medium"}
                )
            ]
        )
    ]
)

# ===========================================
# 输出结果
# ===========================================
print(response.text)