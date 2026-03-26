# Python 语法练习集

本目录包含全面的 Python 语法练习，涵盖基础到高级特性。每个练习都生成对应的 `.out` 输出文件用于验证。

## 练习列表

### 基础语法 (v1)
- `fibonacci.py` - 斐波那契数列（循环）
- `squares.py` - 列表推导平方数

### 其他解法 (v2)
- `fibonacci_recursive.py` - 斐波那契递归（缓存）
- `fibonacci_generator.py` - 斐波那契生成器
- `map_filter.py` - map/filter 高阶函数
- `class_example.py` - 简单类定义

### 高级语法 (v3)
- `exception_handling.py` - 异常处理
- `decorator.py` - 装饰器
- `context_manager.py` - 上下文管理器
- `comp_vs_gen.py` - 列表推导 vs 生成器

### 数据结构与工具 (v4)
- `dataclass_example.py` - dataclass
- `property_example.py` - property 属性
- `json_example.py` - JSON 序列化
- `argparse_example.py` - 命令行解析

### 标准库进阶 (v5)
- `asyncio_example.py` - 异步编程
- `typing_example.py` - 类型提示
- `unittest_example.py` - 单元测试
- `re_example.py` - 正则表达式
- `pathlib_example.py` - 路径操作
- `collections_example.py` - collections 模块
- `itertools_example.py` - itertools 模块
- `functools_example.py` - functools 模块

### 并发编程 (v6)
- `threading_example.py` - 多线程
- `multiprocessing_example.py` - 多进程
- `sqlite_example.py` - SQLite 数据库
- `logging_example.py` - 日志记录
- `subprocess_example.py` - 子进程
- `enum_example.py` - 枚举类型
- `futures_example.py` - concurrent.futures
- `datetime_example.py` - 日期时间

### 高级特性 (v7)
- `async_context.py` - 异步上下文管理器
- `generator_coroutine.py` - 生成器协程
- `typing_advanced.py` - 类型提示高级
- `contextlib_advanced.py` - contextlib 高级
- `singledispatch_example.py` - 泛型函数
- `profile_example.py` - 性能分析

### 设计模式 (v8)
- `singleton_module.py`, `singleton_decorator.py`, `singleton_metaclass.py` - 单例模式
- `factory_pattern.py` - 工厂模式
- `descriptor.py` - 描述符
- `metaclass_logger.py` - 元类
- `semaphore_example.py` - 信号量
- `queue_example.py` - 线程队列
- `asyncio_queue.py` - 异步队列
- `http_client.py` - HTTP 客户端

### 高级 OOP (v9)
- `abc_example.py` - 抽象基类
- `decorator_with_args.py` - 带参数装饰器
- `custom_iterator.py` - 自定义迭代器
- `dynamic_attrs.py` - 动态属性
- `magic_methods.py` - 魔术方法
- `memoryview_example.py` - 内存视图
- `doctest_example.py` - 文档测试
- `config_example.py` - 配置文件解析
- `random_example.py` - 随机数

### 内存与上下文 (v10)
- `slots_example.py` - __slots__
- `weakref_example.py` - 弱引用
- `contextvars_example.py` - 上下文变量
- `partialmethod_example.py` - partialmethod
- `dataclass_advanced.py` - dataclass 高级
- `async_iterator.py` - 异步迭代器
- `struct_example.py` - 二进制结构
- `reprlib_example.py` - reprlib
- `array_example.py` - 数组

### 标准库全面覆盖 (v11)
- `protocol_example.py` - typing.Protocol
- `async_contextlib.py` - 异步上下文管理器
- `cache_example.py` - functools.cache
- `importlib_example.py` - 动态导入
- `pathlib_advanced.py` - pathlib 高级
- `shutil_example.py` - shutil 文件操作
- `tempfile_example.py` - 临时文件
- `glob_example.py` - glob 匹配
- `argparse_advanced.py` - 子命令
- `logging_config.py` - logging 配置
- `datetime_tz.py` - 时区处理
- `hashlib_example.py` - 哈希
- `secrets_example.py` - 安全随机数
- `uuid_example.py` - UUID
- `base64_example.py` - Base64
- `json_custom.py` - JSON 自定义编码

## 验证

每个 `.py` 文件都有对应的 `.out` 输出文件，可通过以下命令验证：

```bash
for py in *.py; do
    python "$py" > /dev/null 2>&1 || echo "Failed: $py"
done
```

## Git 标签

- `python-exercise-v1` 至 `v11` 对应各批次提交。
