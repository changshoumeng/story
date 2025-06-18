# 故事创作AI助手规划

## 架构目标
- 基于 LazyLLM multi-agent 框架实现在线小说创作系统
- 故事总字数目标 5 万字以上
- 提供可追踪的日志输出
- 通过 WebModule 提供交互界面

## Agent 角色
1. **StoryPlanner**：解析用户概念，输出故事设定
2. **OutlineDesigner**：根据设定生成章节大纲
3. **ContentWriter**：按照大纲逐章创作内容
4. **QualityReviewer**：检查内容质量、连贯性
5. **Editor**：对章节进行最终润色

## 流程概述
1. 系统初始化后先由 StoryPlanner 生成故事设定
2. OutlineDesigner 根据设定生成 25-30 章的大纲
3. 对于每一章循环执行：
   - ContentWriter 依据大纲创作初稿
   - QualityReviewer 给出评估与修改意见
   - Editor 输出润色后的最终文本
   - 保存章节并累积字数，当达到 5 万字停止
4. 全流程在日志中记录阶段、字数和进度
5. 最终生成的章节列表和统计信息返回给用户

## 日志方案
- 使用 `logging` 模块，日志级别 INFO
- 关键阶段均输出：阶段名称、说明、当前字数、总章节数
- 错误信息记录为 ERROR 级别

## 运行方式
1. 设置 `LAZYLLM_OPENAI_API_KEY` 指向可用的 OpenAI API 密钥
2. 可选变量 `LAZYLLM_BASE_URL` 与 `NOVEL_TARGET_WORDS` 控制接口地址与目标字数
3. 运行 `python story_online.py` 后访问日志中显示的端口范围

## 与 README 的对比与反思
README 中已经给出系统架构与主要组件。本实现遵循其设定，采用相同的 Agent 分工和流程。不过在实现上加入了 `logging` 记录，以及将核心流程封装为函数以便 Web 调用。若后续使用结果显示上下文管理或代理协作不足，可继续增加摘要生成 Agent 或更细致的状态管理功能。
