# S1 AI Discussion Report

# [Ox Alpha被认领，GLM-5.3-Flash上线|大模型讨论专楼](https://stage1st.com/2b/thread-2275806-1-1.html)

【AI/LLM 趋势】
- 8月28日下午8点：zcode更新送3亿token，反映国产卡集群算力宽松。
- 8月28日下午9点：Qwen 3.8 Flash Next可在DGX Spark通过sglang/vllm部署，NVFP4量化模型发布；dsh发布破坏性更新致启动失败。
- 8月29日上午2点：pi扩展组件包（审批/余额/状态监视）发布于GitHub/NPM。
- 8月29日上午3点：梁文锋称DeepSeek V4正式版flash及多模态足以称成功（涨价后）。
- 8月29日上午11点：GLM 5.3 Flash、Hy4 preview、Qwen 3.8 Flash Next架构对比信息（chenyedgg）。
- 8月29日下午12点：opencode上线Ling-3.0-flash-Fin（蚂蚁集团金融模型）。
- 8月29日下午2点：OpenAI与Cursor解约（Cursor被SpaceX收购）；deepseek-search-mcp提供DS原生搜索需装。
- 8月29日下午4点：dsh 0.1.2 agent team功能引关注（类agent间通信）。
- 8月29日下午6点：cpa反代可转GPT/Gemini请求免梯（zjf实践）。
- 8月29日下午7点：警告dsh agent team易拉128子agent烧干余额。

【用户观点】
- CCauchy：梁文谷涨价后性价比≈gpt5x(不开fast)，上下文翻倍则选DS。
- tonyunreal：hy4比梁文谷0813贵比GLM便宜，能力居中。
- 七氷：gpt-5.6-sol+Gemini-3.7-flash子代理，plus额度用不完。
- 星野あさみ：vllm跑Qwen3.8 FN速度是sglang两倍，但GPU功耗仅50W。
- 舞以/UmarIbnLaAhad：第三方评测不靠谱，一句话生成玩具不实用，多轮才是关键。
- 杀人鲸：梁子涨价后性价比仍顶级。
- cscbzcbz：盼友商支棱让梁子降价；开源百花齐放。
- xiaohanne/scikirbypoke：华子给卡或梁子故意限API保训练。
- chenyedgg：qwen3.8 FN架构潜力最好，GLM5.3≈V4F，Hy4≈GLM5.3。
- nxmonitor/浪费喝咖啡：workbuddy卡，codebuddy cli更好；HY4在codebuddy差。
- Azcarlo/御姐贾：DS搜索差且贵，不如exa/tavily；cherry 2.0默认ExaMCP贵。
- andychen：一句话生成如赛博许愿；头脑风暴智能体实用。
- qwased：dsh alpha前别用防烧余额；不知需求可用grill me。

【关键论点】
- 性价比与选择：DS V4涨价仍具性价比；按场景选HY4/GLM/Qwen。
- 评测与实用落差：第三方评测不可信，agent实用看多轮与子代理协作。
- Agent演进：中心化树状向监管+网状；子代理外包可行（agy/ds flash）。
- 搜索能力：DS原生搜索效果差、计费贵，推荐MCP或第三方。
- 本地部署：Qwen3.8 FN量化版涌现，消费级显卡可期。
- 工具生态：zcode送量、pi扩展、dsh迭代快但稳定性风险（破坏性更新/烧余额）。
- 获取途径：白嫖英伟达DS、zcode token、cpa反代免梯。

---

# [《时代》AI百大人物出炉 黄仁勋 梁文峰落选](https://stage1st.com/2b/thread-2288779-1-1.html)

【AI/LLM 趋势】
- 8月29日下午1点：《时代》2026 TIME100 AI榜发布，黄仁勋、梁文峰落选，11华人上榜（吴泳铭、李飞飞登封面），重心转应用层与Agent。
- 8月29日下午2点：DeepSeek V3.2（年初）、V4（4月）、V4F0731模型进展被提及。
- 8月29日下午3点：KIMI K3发布冲击美股；DeepSeek V4 Flash/Pro上线未掀水花但全球用至涨价；KIMI威胁闭源溢价，0731威胁闭源总营收。
- 8月29日下午4点：帕丽斯·希尔顿因推动《DEFIANCE法案》（AI生成露骨内容受害者起诉权）入选。
- 8月29日下午5点：榜单国籍统计中国籍8人、英国6人、阿三籍5人。
- 8月29日下午6点：梁孟松低调专注芯片制造；算裔阿三超中国，美国独占60+人。

【用户观点】
- 多名用户讥讽榜单“野鸡”/政治化/分猪肉，落选非黄仁勋梁文峰损失而是榜单傻逼。
- 质疑华人仅11人过低，调侃阿三含量高；封面帕丽斯成分可疑，或为推动AI上市抬咖造势。
- 部分反驳“没新活”说，列举DeepSeek今年模型进展；TIME在AI领域不权威不应严肃讨论。

【关键论点】
- 榜单漏选头部华人反映其政治化/圈子化，不代表技术实力。
- 国产模型（DeepSeek、KIMI）今年实质冲击美股与闭源生态，使用率与影响力真实。
- 榜单涵盖芯片、机器人、Agent及安全法案，侧重应用与合规。

---

# [16G显卡+qwen3.8 27B上下文200K，个人经验总结](https://stage1st.com/2b/thread-2288655-1-1.html)

【AI/LLM 趋势】
- 8月29日上午10点：zktz 用 ollama + gemma4 31b（48G A6000，200k上下文）。
- 8月29日下午3点：qwased 荐 qwen3.8-27b-vision q6/q8 + MTP（tb4+4开262k）；zktz 确认命令；qwased 补 --spec-draft-n-max 2/3。
- 8月29日下午6点：llamacpp rocm 版支持 A 卡类似部署。

【用户观点】
- zktz：A6000 跑 gemma4 31b 256k 爆显存，控200k慢、识图爆、单并发；求方案。
- qwased：换 q6/q8 视觉回显存，tb4+4开262k+MTP；A卡用 llamacpp rocm。
- ljwlwd：疑 N/A 卡无别，问能否同操。

【关键论点】
- 48G 显存放 qwen3.8-27b-vision q6_k + MTP 可支262k上下文与视觉，解爆显存提速。
- 命令参数：`-m qwen3.8-27b-vision-q6_k.gguf --spec-type draft-mtp -c 262144 --cache-type-k q8_0 --cache-type-v q8_0 -n-gpu-layers 99`，草稿 `--spec-draft-n-max 2/3`。
- A 卡借 llamacpp rocm 版可达类似效果。

---

# [和AI辩（圣）经，其乐无穷，收获颇深](https://stage1st.com/2b/thread-2288716-1-1.html)

## 【AI/LLM 趋势】
- 8月28日下午9点：cheatdeath1942 质疑用AI辩经不如看宗教思想史。
- 8月28日下午10点：fneasag 与AI辩经问成佛状态，AI最终没招。
- 8月28日下午11点：吉黑尽阵 称AI方便，懒得看砖头书。
- 8月29日上午1点：绝地潜兵 提议让AI完善新圣经修bug；1242599693 与AI辩经问唯心主义他者等。
- 8月29日上午3点：灰狼 用AI问敏感问题及40k设定讨论。

## 【用户观点】
- cheatdeath1942：诘问术古已有，AI未必比经典懂。
- fneasag：AI对成佛状态无法用现思维解释。
- 吉黑尽阵：AI方便；成佛问题不可能答。
- 缪斯替：诺斯替求“润”；AI修圣经趋自然神论/无神论。
- 绝地潜兵：AI写新圣经修复愚昧。
- 1242599693：AI问唯心他者、上帝与看不见的手。
- 灰狼：AI宜问敏感/跨设定。
- martinoy：宗教是意识形态工具，辩经难改观念，传教靠信任；常见无神论框架。
- 吉黑尽阵（#31）：核心矛盾为神善不可理解致行善/作恶难辨。

## 【关键论点】
- AI辩经方便但有限：终极宗教问题难答，不替代经典。
- 宗教辩经实质：意识形态承载，核心在信任非逻辑；耶稣屠杀矛盾凸显神善不可知。
- AI生成圣经：修正bug会滑向自然神论/无神论。
- AI作工具：可安全探讨敏感或虚构设定话题。

---
