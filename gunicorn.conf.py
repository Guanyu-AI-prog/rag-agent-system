# Gunicorn配置文件
import multiprocessing

# 绑定地址和端口
bind = "0.0.0.0:8001"

# Worker数量，建议 (2 x CPU核心数) + 1
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# 超时设置
timeout = 120
keepalive = 5
graceful_timeout = 30

# 进程文件
pidfile = "/tmp/rag-api.pid"
daemon = False

# 日志配置
accesslog = "/root/langchain_rag_code/logs/gunicorn_access.log"
errorlog = "/root/langchain_rag_code/logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# 内存管理
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# 安全设置
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# 性能优化
tmp_upload_dir = None