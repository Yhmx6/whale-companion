FROM python:3.12-slim

# 中文字体：导出图的"AI生成"标识需要它，否则会渲染成方块
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0 PORT=8848 \
    ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3 \
    ARK_IMAGE_MODEL=doubao-seedream-5-0-pro-260628 \
    ARK_VLM_MODEL=doubao-seed-2-1-lite-260915

EXPOSE 8848
CMD ["python", "-m", "app.server"]
