"""自定义顶点示例：通过继承 framework.vertex.Vertex 实现自定义顶点。

两种自定义顶点：
    SanitizeVertex —— 写入前清洗文本空白，并额外统计词数(自定义属性)
    ReportVertex   —— 出边触发前把所有输入合并为一份报告

关键点：
    1. 覆盖 `set`：在调用 super().set() 之前做自定义处理；
       注意必须调用 super().set() 来完成真实写入、输入计数与 READY 推进。
    2. 覆盖 `prepare_outputs`：出边触发前运行的自定义逻辑。
    3. 坑：不要在 prepare_outputs 里调用 self.set() 写数据 —— 那会再次
       把状态改成 READY，导致执行器重复处理该顶点(死循环)。
       应使用下面封装的 _store() 直接写入数据存储，不触发状态机。
"""

from framework.vertex import Vertex


class SanitizeVertex(Vertex):
    """自定义顶点：清洗文本空白，并记录每个数据键的单词数。"""

    def __init__(self, vertex_id, **kwargs):
        # 先把用户配置与初始数据传给父类
        super().__init__(vertex_id, **kwargs)
        # 自定义属性：data_id:tags -> 单词数
        self.word_counts = {}

    async def _store(self, data, data_id="default", tags=None):
        """直接写入数据存储，不推进状态机。

        仅在 prepare_outputs 内部使用，避免把 PROCESSING 重置回 READY。
        """
        key = self._make_key(data_id, tags)
        async with self._lock:
            self._data_store[key] = data
        return data

    async def set(self, data, data_id="default", tags=None):
        """写入前的自定义处理：压缩多余空白并统计词数。

        注意：只有经边(Edge)写入的数据才会走这里；
        initial_data 在 __init__ 里直接写入数据存储，不经过 set()。
        因此示例中 input 顶点的初始数据未被清洗，processor 收到的数据才会被清洗。
        """
        if isinstance(data, str):
            # 1) 自定义转换：把 "  Hello   world!  " 压成 "Hello world!"
            data = " ".join(data.split())
            # 2) 记录统计信息到自定义属性
            key = f"{data_id}:{','.join(sorted(tags or []))}"
            self.word_counts[key] = len(data.split())
        # 3) 调用父类完成真实写入、计数与状态推进
        return await super().set(data, data_id, tags)

    async def prepare_outputs(self):
        """出边触发前，把统计信息合入数据存储。"""
        # 汇总所有键的单词数
        stats = {"total_words": sum(self.word_counts.values())}
        # 用 _store 写入，避免触发状态机(见类注释中的"坑")
        await self._store(stats, "stats", ["summary"])
        # 仍调用父类实现(父类会执行 on_ready 脚本钩子，如有)
        await super().prepare_outputs()


class ReportVertex(Vertex):
    """自定义顶点：把收到的所有字符串数据合并为一份报告。"""

    async def prepare_outputs(self):
        # 读取顶点当前全部数据
        all_data = await self.get_all_data()
        parts = []
        for key in sorted(all_data.keys()):
            val = all_data[key]
            if isinstance(val, str):
                label = f"{key[0]}:{','.join(key[1])}"
                parts.append(f"[{label}] {val}")
        combined = "\n".join(parts)
        # 直接写入数据存储，不触发状态机
        await self._store(combined, "report", ["final"])
        await super().prepare_outputs()
