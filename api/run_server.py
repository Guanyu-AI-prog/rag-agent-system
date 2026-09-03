#!/usr/bin/env python3
"""RAG API 启动入口 — 修复 api.py main() 不启动 uvicorn 的问题"""
import sys, os
sys.path.insert(0, "/root/langchain_rag_code/core")
sys.path.insert(0, "/root/langchain_rag_code/infra")
sys.path.insert(0, "/root/langchain_rag_code/api")
os.chdir("/root/langchain_rag_code/api")

import api, uvicorn
uvicorn.run(api.app, host="0.0.0.0", port=8001, log_level="info", access_log=False)
