# 验收 Checklist: 修改 agent666 — 使用 upload 上传详细回复后回复摘要+链接

## 默认条目
- [x] 代码无语法错误，能正常运行
- [x] 所有新增/修改文件已保存
- [x] 执行相关测试，已经全部通过
- [x] 停止前已 `git add` 并 `git commit`
- [x] 停止前已 `git push`（如有远程）
- [x] 已向 Board 666 回复执行结果

## 本次任务条目
- [x] `.opencode/agents/Auto666.md` — 工作流已添加"生成报告 → 上传 → 回复摘要"步骤
- [x] `loop.py` — prompt 模板已更新，反映新的工作流
- [x] `reports/` 目录已创建（或确认存在）
- [x] 验证 upload.py 脚本路径存在且可执行
- [x] 验证 moonchan.py reply 命令可用
- [x] upload 上传详细报告后，不再回复完整详文本，而是回复摘要+链接
