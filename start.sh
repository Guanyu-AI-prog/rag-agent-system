#!/bin/bash
# RAG Agent 系统快速启动脚本
# 用法: ./start.sh [install|build|api|agent|test|clean|help]

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Python 环境：优先使用项目 venv
if [ -x "$ROOT_DIR/venv/bin/python" ]; then
    PYTHON_CMD="$ROOT_DIR/venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    error "未找到 Python，请安装 Python 3.8+"
    exit 1
fi

check_env() {
    if [ ! -f "$ROOT_DIR/.env" ]; then
        warning "未找到 .env 文件"
        if [ -f "$ROOT_DIR/.env.example" ]; then
            cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
            warning "已从 .env.example 创建 .env，请填入 API 密钥后重试"
        fi
        exit 1
    fi
}

install_deps() {
    info "安装 Python 依赖..."
    $PYTHON_CMD -m pip install -r "$ROOT_DIR/requirements.txt"
    success "依赖安装完成"
}

build_vectors() {
    check_env
    info "构建向量数据库..."
    if [ ! -d "$ROOT_DIR/data" ]; then
        warning "请先创建 data/ 目录并放入知识库文档（.txt/.md/.jsonl/.csv）"
        exit 1
    fi
    $PYTHON_CMD "$ROOT_DIR/vector/build_vectors.py" --force
    success "向量库构建完成"
}

run_api() {
    # RAG 模式（simple_rag.py 后端），默认端口 8001
    check_env
    info "启动 RAG API 服务: http://localhost:8001 (文档: /docs)"
    $PYTHON_CMD "$ROOT_DIR/api/api.py"
}

run_agent() {
    # Agent 模式（dx_agent.py 后端），默认端口 8002
    check_env
    info "启动 Agent API 服务: http://localhost:8002 (文档: /docs)"
    $PYTHON_CMD "$ROOT_DIR/api/dx_agent_api.py"
}

test_query() {
    check_env
    if [ ! -d "$ROOT_DIR/vector_db" ] && [ ! -d "$ROOT_DIR/api/vector_db" ]; then
        warning "向量库不存在，请先运行: ./start.sh build"
        exit 1
    fi
    info "测试查询: 有哪些套餐？"
    $PYTHON_CMD "$ROOT_DIR/core/simple_rag.py" "有哪些套餐？"
}

clean_files() {
    info "清理生成文件..."
    rm -rf "$ROOT_DIR/vector_db" "$ROOT_DIR/api/vector_db" 2>/dev/null || true
    find "$ROOT_DIR" -path "$ROOT_DIR/venv" -prune -o -type d -name "__pycache__" -print0 2>/dev/null | xargs -0 rm -rf 2>/dev/null || true
    success "清理完成"
}

show_help() {
    echo ""
    echo "RAG Agent 系统启动脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  install    安装依赖 (requirements.txt)"
    echo "  build      构建向量数据库"
    echo "  api        启动 RAG API 服务（端口 8001，simple_rag 后端）"
    echo "  agent      启动 Agent API 服务（端口 8002，dx_agent 后端）"
    echo "  test       命令行测试查询"
    echo "  clean      清理向量库与缓存"
    echo "  help       显示此帮助"
    echo ""
}

case "${1:-help}" in
    install) install_deps ;;
    build)   build_vectors ;;
    api)     run_api ;;
    agent)   run_agent ;;
    test)    test_query ;;
    clean)   clean_files ;;
    help|--help|-h) show_help ;;
    *) error "未知命令: $1"; show_help; exit 1 ;;
esac
