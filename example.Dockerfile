# 使用现有的基础镜像
FROM base/python:ubuntu22.04-cuda11.8-python3.9

## 设置代理 (确保网络环境)
#ENV all_proxy=http://192.168.0.64:7890

# 更新并安装必要的依赖
RUN apt-get update \
    # 在这里安装你需要的依赖，比如 git、python 等 \
    && apt-get install libcudnn8 \
    && apt-get install git -y \
    && apt-get install ffmpeg -y \
    && apt-get clean

# 设置工作目录
WORKDIR /app/faster-whisper

# 设置 GPU 使用
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
# 设置访问密钥
ENV WHISPER-SECRET-KEY=sk-whisper

# 克隆并直接重命名为 faster-whisper
RUN cd /app && git clone https://github.com/zxd147/faster-whisper-api.git faster-whisper

# 复制本地文件到容器内
COPY ./faster_whisper/ ./faster_whisper/
COPY ./models/faster-whisper-large-v2/ ./models/faster-whisper-large-v2/

RUN pip install -r ./requirements.txt

# 映射端口
EXPOSE 8001

# 容器启动时默认执行的命令
CMD ["python", "whisper_api.py"]

