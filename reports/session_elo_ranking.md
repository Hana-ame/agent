# DSH Session ELO 排行

> 生成：2026-10-28 04:49　样本：1920 个 session（有最终 assistant 结论）

## 方法

- 每个 session 取最后一个 assistant text 作为结论
- 确定性质量分：内容量/具体性 30%、与用户诉求匹配 25%、结构 20%、行动性 15%、完成状态 10%
- 质量分两两对局 → Elo（初始 1500，K=16，3 轮全对局）
- **这是启发式排序，不是 LLM 语义评判**；ELO 只在本语料内可比

## 样本

- state 中 session 总数：2570
- 可评价（有最终 assistant 文本）：1920；跳过 650（无 transcript/结论）
- 按类型：subagent 1025、top-level 878、fork 17
- 工作区 Top：/mnt/e/knowledge-base 1387、/tmp 361、/home/lumin/code/dsh-plan-decomposition 27、/mnt/e/paper 21、/home/lumin 19、/mnt/d/WorkPlace/peerdrive 11、/mnt/c/Users/lumin/Downloads/agent 9、/home/lumin/peerdrive 8

## Top 30

| # | Elo | 质量分 | Session | 标题 | 类型 | turns/tools | 字数 |
|---:|---:|---:|---|---|---|---:|---:|
| 1 | 4014 | 0.92 | `45d88cac-1a06-43d5-9f6f-` | 【任务性质】只读架构体检。 | subagent | 1/82 | 19426 |
| 2 | 3982 | 0.91 | `7be3a1a1-232f-4939-a56f-` | You are doing READ-ONLY research | subagent | 1/69 | 18085 |
| 3 | 3975 | 0.89 | `a6784d8b-0273-41ab-b92c-` | 你是知识库研究助手。工作目 | subagent | 1/25 | 7055 |
| 4 | 3953 | 0.89 | `76ccf864-8476-43f3-9266-` | 你是知识库研究助手。工作目 | subagent | 1/9 | 16277 |
| 5 | 3935 | 0.88 | `session-03010d82-3f9c-40` | 解读·递归自改进天花板 | top-level | 1/36 | 11759 |
| 6 | 3918 | 0.86 | `session-327665b2-face-40` | 调查一下kb有什么，如果我要 | top-level | 1/24 | 7010 |
| 7 | 3912 | 0.92 | `d6069d0c-5b5b-4962-abb3-` | 只读调研任务。禁止使用 web_se | subagent | 1/72 | 26441 |
| 8 | 3897 | 0.86 | `session-4ede6071-ce78-4d` | repo 盘点 8/9：go-server 家族等 8 个 | top-level | 2/62 | 14281 |
| 9 | 3891 | 0.86 | `session-8732bc91-059d-49` | 读三个 .py 的真实内容 | top-level | 1/5 | 3151 |
| 10 | 3859 | 0.88 | `session-3163650b-4774-42` | 排查refactor主干CI红灯 | top-level | 1/37 | 3658 |
| 11 | 3853 | 0.88 | `session-286e585e-3671-48` | 手动升级 our-free-model 1.4.5 → 1.4.6 | top-level | 1/33 | 3237 |
| 12 | 3852 | 0.85 | `session-bf8bb234-b792-45` | repo 盘点 9/9：proxy 家族等 7 个 | top-level | 3/76 | 17370 |
| 13 | 3850 | 0.89 | `session-20067144-c407-48` | gh 全量盘点 B：our-free-model 的 1.4.5/1.4.6 来… | top-level | 1/42 | 6435 |
| 14 | 3844 | 0.84 | `session-0c0e38eb-7873-4e` | repo 盘点 7/9：minmus 等 8 个 | top-level | 2/50 | 10330 |
| 15 | 3841 | 0.85 | `session-0ef78d28-8a1f-40` | 抓取·S1帖2275806 9月讨论 | top-level | 1/33 | 8091 |
| 16 | 3835 | 0.81 | `57830ef2-63f8-4fe4-bb3e-` | You are doing READ-ONLY web | subagent | 1/64 | 22387 |
| 17 | 3834 | 0.84 | `ca307174-5a01-4564-a6ac-` | READ-ONLY research task. Do not | subagent | 1/81 | 29901 |
| 18 | 3829 | 0.86 | `session-dfd5887f-3338-45` | 调查DSH对话选项卡扩展 | top-level | 1/83 | 5856 |
| 19 | 3814 | 0.90 | `9847d2c0-f3c0-47df-8c21-` | 你是只读代码体检员。任务： | subagent | 2/44 | 31805 |
| 20 | 3811 | 0.83 | `737e54ec-5e93-4f40-bd79-` | 你是知识库研究助手。工作目 | subagent | 1/8 | 5835 |
| 21 | 3805 | 0.81 | `f7668f8d-3b00-4c8d-98a5-` | 你是研究助手。本任务主角是 | subagent | 1/46 | 5789 |
| 22 | 3793 | 0.88 | `session-033eb70e-8ce5-4f` | repo 盘点 3/9：live2d-viewer 等 8 个 | top-level | 2/71 | 7997 |
| 23 | 3787 | 0.83 | `session-7ec6a1b5-b372-43` | peerdrive G402终案+体检(ECH进程内/#370/#371已合并) | top-level | 13/980 | 4445 |
| 24 | 3766 | 0.89 | `bb3f8040-27bf-42e5-a280-` | <system-reminder> You are teammate "sur- | subagent | 2/73 | 25262 |
| 25 | 3764 | 0.88 | `session-c14c76fe-d935-48` | repo 盘点 1/9：peerdrive 等 8 个 | top-level | 2/63 | 12029 |
| 26 | 3744 | 0.80 | `session-341fa85a-f2c1-44` | 8443 端口访问超时排查（逐跳实测） | top-level | 6/89 | 2579 |
| 27 | 3741 | 0.80 | `session-734d23cc-80da-43` | https://bbs.imoutolove.me/read.php?tid-2 | top-level | 4/70 | 14090 |
| 28 | 3739 | 0.80 | `session-a62dfcde-121d-4b` | 查证·enchanted-station.com(重派) | top-level | 1/10 | 3814 |
| 29 | 3727 | 0.80 | `session-d609dc9f-42ed-40` | 升级 dsh-our-free-model 插件到 v1.4.6 | top-level | 2/82 | 3492 |
| 30 | 3722 | 0.87 | `session-a7fb7ac2-3420-41` | repo 盘点 4/9：s3proxy 等 8 个 | top-level | 2/67 | 16110 |

## Top 15：top-level（人开的会话）

| # | 总排名 | Elo | 质量分 | Session | 标题 | 字数 |
|---:|---:|---:|---:|---|---|---:|
| 1 | 5 | 3935 | 0.88 | `session-03010d82-3f9c-40` | 解读·递归自改进天花板 | 11759 |
| 2 | 6 | 3918 | 0.86 | `session-327665b2-face-40` | 调查一下kb有什么，如果我要 | 7010 |
| 3 | 8 | 3897 | 0.86 | `session-4ede6071-ce78-4d` | repo 盘点 8/9：go-server 家族等 8 个 | 14281 |
| 4 | 9 | 3891 | 0.86 | `session-8732bc91-059d-49` | 读三个 .py 的真实内容 | 3151 |
| 5 | 10 | 3859 | 0.88 | `session-3163650b-4774-42` | 排查refactor主干CI红灯 | 3658 |
| 6 | 11 | 3853 | 0.88 | `session-286e585e-3671-48` | 手动升级 our-free-model 1.4.5 → 1.4.6 | 3237 |
| 7 | 12 | 3852 | 0.85 | `session-bf8bb234-b792-45` | repo 盘点 9/9：proxy 家族等 7 个 | 17370 |
| 8 | 13 | 3850 | 0.89 | `session-20067144-c407-48` | gh 全量盘点 B：our-free-model 的 1.4.5/1.4.6 来… | 6435 |
| 9 | 14 | 3844 | 0.84 | `session-0c0e38eb-7873-4e` | repo 盘点 7/9：minmus 等 8 个 | 10330 |
| 10 | 15 | 3841 | 0.85 | `session-0ef78d28-8a1f-40` | 抓取·S1帖2275806 9月讨论 | 8091 |
| 11 | 18 | 3829 | 0.86 | `session-dfd5887f-3338-45` | 调查DSH对话选项卡扩展 | 5856 |
| 12 | 22 | 3793 | 0.88 | `session-033eb70e-8ce5-4f` | repo 盘点 3/9：live2d-viewer 等 8 个 | 7997 |
| 13 | 23 | 3787 | 0.83 | `session-7ec6a1b5-b372-43` | peerdrive G402终案+体检(ECH进程内/#370/#371已合并) | 4445 |
| 14 | 25 | 3764 | 0.88 | `session-c14c76fe-d935-48` | repo 盘点 1/9：peerdrive 等 8 个 | 12029 |
| 15 | 26 | 3744 | 0.80 | `session-341fa85a-f2c1-44` | 8443 端口访问超时排查（逐跳实测） | 2579 |

## Top 10：subagent

| 总排名 | Elo | 质量分 | Session | 标题 | 字数 |
|---:|---:|---:|---|---|---:|
| 1 | 4014 | 0.92 | `45d88cac-1a06-43d5-9f6f-` | 【任务性质】只读架构体检。 | 19426 |
| 2 | 3982 | 0.91 | `7be3a1a1-232f-4939-a56f-` | You are doing READ-ONLY research | 18085 |
| 3 | 3975 | 0.89 | `a6784d8b-0273-41ab-b92c-` | 你是知识库研究助手。工作目 | 7055 |
| 4 | 3953 | 0.89 | `76ccf864-8476-43f3-9266-` | 你是知识库研究助手。工作目 | 16277 |
| 7 | 3912 | 0.92 | `d6069d0c-5b5b-4962-abb3-` | 只读调研任务。禁止使用 web_se | 26441 |
| 16 | 3835 | 0.81 | `57830ef2-63f8-4fe4-bb3e-` | You are doing READ-ONLY web | 22387 |
| 17 | 3834 | 0.84 | `ca307174-5a01-4564-a6ac-` | READ-ONLY research task. Do not | 29901 |
| 19 | 3814 | 0.90 | `9847d2c0-f3c0-47df-8c21-` | 你是只读代码体检员。任务： | 31805 |
| 20 | 3811 | 0.83 | `737e54ec-5e93-4f40-bd79-` | 你是知识库研究助手。工作目 | 5835 |
| 21 | 3805 | 0.81 | `f7668f8d-3b00-4c8d-98a5-` | 你是研究助手。本任务主角是 | 5789 |

## Bottom 10

| # | Elo | 质量分 | Session | 标题 | 字数 |
|---:|---:|---:|---|---|---:|
| 1911 | -856 | 0.09 | `87ce9682-56af-4b71-94d2-` | 你是 B 站转录节点撰写 subagent | 65 |
| 1912 | -870 | 0.08 | `2c839f2a-a9c7-4821-ac43-` | 你是 B 站转录节点撰写 subagent | 62 |
| 1913 | -874 | 0.11 | `session-b5c7eb8b-86ff-42` | 稿件B批2：参考文献IEEE化 | 20 |
| 1914 | -904 | 0.09 | `c9ebabd7-c0ff-473e-87b6-` | 你是 B 站转录节点撰写 subagent | 48 |
| 1915 | -916 | 0.10 | `b1e8f076-3eba-4a2e-8451-` | 你的工作目录是 /mnt/e/knowledge-b | 62 |
| 1916 | -916 | 0.11 | `session-ed70f2d8-73e5-49` | K3改稿②重做 | 27 |
| 1917 | -984 | 0.06 | `fb7299dc-6341-4ab4-bc0d-` | 你在审计一个本地个人知识库 | 30 |
| 1918 | -992 | 0.08 | `f990a84e-e208-4a45-b599-` | 你是B站转录节点撰写 subagent。 | 29 |
| 1919 | -1002 | 0.07 | `2fc257c4-99a3-47c8-892b-` | You are investigating GitHub repos | 2 |
| 1920 | -1016 | 0.06 | `513d1e91-eb43-4f4d-87d9-` | 你负责本地文件视察任务【KB i | 25 |

## 注意

- Top 30 混合了 subagent 和 top-level；只看人开的会话请看 Top 15 top-level
- Elo 绝对值会超过常见 0–3000（3 轮全对局累积），只看相对名次，不要当绝对分
- 结论为空/极短的 session 会被压低；这是数据缺失，不一定是回答差
- 启发式偏好长、有代码/路径/数字、结构清晰的回答，存在长度偏置
- 要语义级评判，需要网关恢复后用 LLM judge 重排

