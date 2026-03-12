# [START] BASE-RULE
# version: 0.0.1
# 上下文：定义匹配规则的抽象基类。

class BaseRule:
    
    # [START] BASE-RULE-MATCH
    # version: 0.0.1
    # 上下文：由规则处理器分发文本时调用。
    # 输入参数：text (str)
    # 输出参数：执行结果汇总 (str)
    async def match_and_execute(self, text: str) -> str:
        raise NotImplementedError
    # [END] BASE-RULE-MATCH
# [END] BASE-RULE