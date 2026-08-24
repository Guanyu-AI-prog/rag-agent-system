#!/bin/bash
# RAG系统快速启动脚本
# 用法: ./start.sh [build|run|test|clean]

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python() {
    info "检查Python环境..."
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        error "未找到Python，请安装Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    success "找到 $PYTHON_VERSION"
}

# 检查环境变量
check_env() {
    info "检查环境配置..."
    if [ ! -f .env ]; then
        warning "未找到 .env 文件"
        if [ -f .env.example ]; then
            info "从 .env.example 创建 .env 文件..."
            cp .env.example .env
            warning "请编辑 .env 文件，填入你的API密钥"
            warning "然后重新运行此脚本"
            exit 1
        else
            error "未找到 .env.example 文件"
            exit 1
        fi
    fi
    
    # 检查关键配置
    source .env
    if [ -z "$SILICONFLOW_API_KEY" ] || [ "$SILICONFLOW_API_KEY" = "sk-your-siliconflow-api-key-here" ]; then
        error "请在 .env 文件中配置 SILICONFLOW_API_KEY"
        exit 1
    fi
    
    success "环境配置检查通过"
}

# 安装依赖
install_deps() {
    info "安装Python依赖..."
    if [ -f requirements.txt ]; then
        $PYTHON_CMD -m pip install -r requirements.txt --quiet
        success "依赖安装完成"
    else
        warning "未找到 requirements.txt 文件"
    fi
}

# 构建向量库
build_vectors() {
    info "构建向量数据库..."
    if [ ! -d "data" ]; then
        warning "创建 data 目录..."
        mkdir -p data
        info "请将知识库文档（.txt或.md格式）放入 data/ 目录"
        info "然后重新运行构建"
        return 1
    fi
    
    # 检查是否有文档文件
    if [ -z "$(ls -A data/*.txt data/*.md 2>/dev/null)" ]; then
        warning "data/ 目录中没有 .txt 或 .md 文件"
        info "请添加知识库文档到 data/ 目录"
        return 1
    fi
    
    $PYTHON_CMD build_vectors.py --force
    success "向量库构建完成"
}

# 启动服务
run_server() {
    info "启动RAG API服务..."
    info "服务地址: http://localhost:8001"
    info "API文档: http://localhost:8001/docs"
    info "按 Ctrl+C 停止服务"
    echo ""
    
    $PYTHON_CMD api.py
}

# 测试查询
test_query() {
    info "测试RAG查询..."
    
    # 检查向量库是否存在
    if [ ! -d "vector_db" ]; then
        warning "向量库不存在，请先运行: ./start.sh build"
        return 1
    fi
    
    # 测试查询
    TEST_QUERY="有哪些套餐？"
    info "测试查询: $TEST_QUERY"
    echo ""
    
    $PYTHON_CMD workflow_langchain.py "$TEST_QUERY"
}

# 清理文件
clean_files() {
    info "清理生成文件..."
    
    # 删除向量库
    if [ -d "vector_db" ]; then
        rm -rf vector_db
        success "已删除 vector_db/"
    fi
    
    # 删除缓存文件
    rm -f .langchain.db
    rm -f *.log
    
    # 删除__pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    success "清理完成"
}

# 显示帮助
show_help() {
    echo ""
    echo "RAG系统快速启动脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  build    构建向量数据库"
    echo "  run      启动API服务"
    echo "  test     测试查询"
    echo "  install  安装依赖"
    echo "  clean    清理生成文件"
    echo "  all      安装依赖并构建向量库"
    echo "  help     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 install  # 安装依赖"
    echo "  $0 build    # 构建向量库"
    echo "  $0 run      # 启动服务"
    echo "  $0 test     # 测试查询"
    echo ""
}

# 主函数
main() {
    echo ""
    echo "========================================"
    echo "   LangChain RAG 系统启动脚本"
    echo "========================================"
    echo ""
    
    check_python
    
    case "${1:-help}" in
        "build")
            check_env
            build_vectors
            ;;
        "run")
            check_env
            run_server
            ;;
        "test")
            check_env
            test_query
            ;;
        "install")
            install_deps
            ;;
        "clean")
            clean_files
            ;;
        "all")
            check_env
            install_deps
            build_vectors
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"