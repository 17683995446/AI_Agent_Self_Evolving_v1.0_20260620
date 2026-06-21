"""
pytest全局配置 —— 统一测试路径管理

O5：测试秩序统一
- 消除所有测试文件顶部的 sys.path.insert 副作用
- 统一通过 conftest.py 管理路径
- 所有测试统一使用 pytest 框架
"""

import os
import sys

# 将项目根目录添加到 sys.path（仅在此处执行一次）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
