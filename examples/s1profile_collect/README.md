# S1 Profile Collect —— 用 vertex-edge graph 实现 chatto「去 stage1st 收集帖子」任务

把 chatto-bot 的 **s1profile 每日任务**(`plugins/s1parse.py` + `plugins/s1profile.py`)用
**vertex-edge agent** 的方式实现成一条 **edge script + graph**:

```
forum(source, URL) ─e_collect─▶ posts_pool ─e_enrich─▶ profiles(sink)
```

## 任务逻辑(与 chatto 原实现一一对应)

1. **e_collect**(`stage="collect"`):抓配置版块(默认 forum-157 外野)最新 N 个帖的**全部楼层**,
   按 uid 归并 → `by_uid`。对应 `s1parse.forum_to_json` + `s1profile._collect` 的前半段。
2. **e_enrich**(`stage="enrich"`):对每个发言 uid 再补抓其「发的主题」
   (`home.php?mod=space&uid=<uid>&do=thread&view=me`),把 uid 自己的主题帖楼层并入。
   对应 `s1profile._collect` 的后半段。
3. **post_process(e_enrich)**:与已有档案 `data/profiles/<uid>.json` 合并,
   **按 (uid, page, 正文前40字) 去重**(与 `s1profile._merge_posts` 一致),写回 JSON。

全程透明透传:两条边 `prompt=""` + `model="default"`,**0 token、不调用 LLM**。

## 运行

```bash
# 在项目根目录(默认抓最新 3 帖,输出到本目录 data/profiles,不动 chatto 真实档案)
python examples/s1profile_collect/run.py

# 抓更多帖 / 换版块 / 直接更新 chatto 真实档案
python examples/s1profile_collect/run.py --max-threads 15
python examples/s1profile_collect/run.py --url "https://stage1st.com/2b/forum-157-1.html" \
    --profile-dir /mnt/d/WorkPlace/chatto-bot/data/profiles
```

## Graph(`config.json`)

- **forum**(source):`initial_data` 提供版块 URL(channel `url`)。
- **posts_pool**(中间顶点):暂存 e_collect 产出的 `by_uid`。
- **profiles**(sink):接收 e_enrich 的最终结果。
- 两条边共用同一个 edge script `s1profile_collect.py`,用 `settings.stage` 区分阶段;
  中间顶点数据按 `channel="url"` 存取(pipeline 用 `source_vertex.fetch_data(channel)` 取数)。

## Edge script(`s1profile_collect.py`)

- 解析层(镜像 `s1parse`):楼层 `div[id^="post_"]`、uid 取 `a[href*="uid="]`、正文 `td.t_f`、
  时间支持相对/绝对、版块清单只选 `a.xst`。
- 收集层(镜像 `s1profile._collect`):`collect_forum_posts`(版块按 uid 归并)+
  `enrich_own_threads`(每人 do=thread 补抓)。
- 档案层(镜像 `s1profile._merge_posts/_save_profile`):`merge_into_profiles` 去重合并写回。

## 踩坑记

- stage1st 匿名态 `do=reply` 回帖列表要登录,只抓「帖子楼层 + 用户发的主题」,不依赖登录。
- stage1st 是境内站,`trust_env=False` 直连,不跟随环境代理(否则页面错乱/时间全旧)。
- **修正了 chatto 原实现的 bug**:`s1profile._merge_posts` 去重键用 `p["uid"]`,但存进档案的
  帖子没有 uid 字段 → 第二次运行会 `KeyError`。本实现用档案自身的 uid 做键,重复收集幂等。

## 测试

```bash
python -m pytest tests/test_s1profile_collect.py -q   # 离线(罐头 HTML):8 passed
```
