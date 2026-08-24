#!/usr/bin/env python3
"""
RAG系统配置管理工具
显示当前配置，支持快速修改关键参数
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import Config
except ImportError:
    print("错误: 无法导入config模块，请确保在项目目录中运行")
    sys.exit(1)

def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """加载.env文件内容"""
    env_vars = {}
    env_file = Path(env_path)
    
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
    return env_vars

def show_current_config():
    """显示当前配置"""
    print("=" * 50)
    print("RAG系统当前配置")
    print("=" * 50)
    
    # 从环境变量读取（会覆盖.env文件）
    env_vars = load_env_file()
    
    config_items = [
        ("模型配置", [
            ("LLM模型", "LLM_MODEL", Config.LLM_MODEL),
            ("嵌入模型", "EMBED_MODEL", Config.EMBED_MODEL),
            ("重排序模型", "RERANK_MODEL", Config.RERANK_MODEL),
            ("启用重排序", "USE_RERANK", str(Config.USE_RERANK)),
        ]),
        ("检索配置", [
            ("召回文档数", "RETRIEVAL_K", str(Config.RETRIEVAL_K)),
            ("分块大小", "CHUNK_SIZE", str(Config.CHUNK_SIZE)),
            ("分块重叠", "CHUNK_OVERLAP", str(Config.CHUNK_OVERLAP)),
        ]),
        ("服务配置", [
            ("API端口", "API_PORT", str(Config.API_PORT)),
            ("最大线程", "MAX_WORKERS", str(Config.MAX_WORKERS)),
            ("缓存时间(秒)", "CACHE_TTL", str(Config.CACHE_TTL)),
            ("查询超时(秒)", "QUERY_TIMEOUT", str(Config.QUERY_TIMEOUT)),
        ]),
        ("路径配置", [
            ("向量库路径", "VECTOR_DB_PATH", Config.VECTOR_DB_PATH),
            ("文档路径", "DATA_DIR", Config.DATA_DIR),
            ("日志文件", "LOG_FILE", Config.LOG_FILE),
            ("日志级别", "LOG_LEVEL", Config.LOG_LEVEL),
        ]),
    ]
    
    for category, items in config_items:
        print(f"\n{category}:")
        for display_name, env_key, value in items:
            # 隐藏敏感信息
            if "KEY" in env_key or "SECRET" in env_key:
                if value and len(value) > 8:
                    value = value[:8] + "..." + value[-4:]
            print(f"  {display_name:15} = {value}")
    
    print("\n" + "=" * 50)
    print("配置文件路径: .env")
    print("完整配置文档: config.py")
    print("=" * 50)

def update_config(key: str, value: str):
    """更新.env文件中的配置"""
    env_file = Path(".env")
    lines = []
    key_found = False
    
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
    # 查找并更新键
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and '=' in stripped:
            existing_key, _ = stripped.split('=', 1)
            if existing_key.strip() == key:
                new_lines.append(f"{key}={value}\n")
                key_found = True
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # 如果没找到，添加新行
    if not key_found:
        new_lines.append(f"\n# 通过配置工具添加\n{key}={value}\n")
    
    # 写入文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ 配置已更新: {key}={value}")
    print("请重启API服务使配置生效")

def interactive_mode():
    """交互式配置模式"""
    print("RAG系统配置工具 - 交互模式")
    print("=" * 40)
    
    while True:
        print("\n可配置项:")
        print("1. LLM_MODEL      - LLM模型名称")
        print("2. EMBED_MODEL    - 嵌入模型")
        print("3. TOP_K          - 召回文档数")
        print("4. CHUNK_SIZE     - 分块大小")
        print("5. API_PORT       - API端口")
        print("6. LOG_LEVEL      - 日志级别")
        print("7. USE_RERANK     - 启用重排序")
        print("8. 查看当前配置")
        print("0. 退出")
        
        choice = input("\n请选择 (0-8): ").strip()
        
        if choice == '0':
            break
        elif choice == '8':
            show_current_config()
        elif choice in ['1', '2', '3', '4', '5', '6', '7']:
            key_map = {
                '1': 'LLM_MODEL',
                '2': 'EMBED_MODEL',
                '3': 'RETRIEVAL_K',
                '4': 'CHUNK_SIZE',
                '5': 'API_PORT',
                '6': 'LOG_LEVEL',
                '7': 'USE_RERANK',
            }
            key = key_map[choice]
            current = getattr(Config, key, '')
            new_value = input(f"输入新值 (当前: {current}): ").strip()
            
            if new_value:
                update_config(key, new_value)
        else:
            print("无效选择")

def main():
    parser = argparse.ArgumentParser(description="RAG系统配置管理工具")
    parser.add_argument('--show', action='store_true', help='显示当前配置')
    parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='设置配置项')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互式配置')
    
    args = parser.parse_args()
    
    if args.set:
        key, value = args.set
        update_config(key, value)
    elif args.interactive:
        interactive_mode()
    else:
        show_current_config()

if __name__ == "__main__":
    main()