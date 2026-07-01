#!/usr/bin/env python3
"""
启动 API 服务
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import create_app
import uvicorn


def main():
    """启动 API 服务"""
    app = create_app()
    
    print("=" * 50)
    print("足球滚球预测系统 API")
    print("=" * 50)
    print("访问 http://localhost:8000/docs 查看 API 文档")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
