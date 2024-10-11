# 使用官方Python运行时作为父镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录下的文件到容器的/app目录
COPY . /app

# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露Streamlit默认端口
EXPOSE 8501

# 运行Streamlit应用
CMD ["streamlit", "run", "web.py"]