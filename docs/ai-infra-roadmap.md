# CacheFlow-Nano：面向 RAG / Agent 的 SLO 感知推理服务路线

## 1. 项目定位

### 一句话描述

在 Nano-vLLM 上构建一个面向企业 RAG、客服机器人和 Agent 工作负载的推理服务：复用重复的系统提示词、工具描述和文档前缀，并在请求竞争 GPU 时兼顾缓存命中率、TTFT 与公平性。

### 要解决的实际问题

企业推理请求经常共享很长的前缀：

- 客服请求共享系统提示词、服务规范和产品知识；
- RAG 请求共享部分检索文档或模板；
- Agent 请求共享工具 schema 和历史上下文；
- 多租户流量同时到来时，长 prompt 会阻塞短请求，造成 P99 TTFT 恶化。

单纯增加吞吐不能解决这些问题。项目需要回答：

1. KV Cache 能否跨请求安全复用？
2. 缓存空间不足时应该淘汰谁？
3. 调度器怎样利用缓存亲和性，同时避免长请求或低优先级租户饥饿？
4. 优化是否改善了真实 workload 下的 goodput，而不只是离线 tokens/s？

### 最终作品形态

项目完成时应包含：

- 一个可流式返回结果的 OpenAI-compatible HTTP 服务；
- 可插拔 KV Cache 淘汰策略：基线 FIFO、LRU、prefix-aware；
- token-budget chunked-prefill 调度与带 aging 的优先级调度；
- 面向重复前缀的 workload generator 和 trace replay；
- TTFT、TPOT、P50/P95/P99、goodput、cache hit、preemption 指标；
- 至少一个经 profiling 后实现的 Triton 融合算子；
- 正确性测试、压力测试、性能报告、架构图和 5 分钟演示。

## 2. 为什么选择这个项目

这个题目不是重新包装一个聊天 Demo。它能同时证明招聘中最重要的四类能力：

| 能力 | 项目证据 |
| --- | --- |
| 推理引擎 | Scheduler、Sequence、Paged KV Cache、Chunked Prefill |
| 系统工程 | 异步服务、背压、SLO、可观测性、故障处理 |
| 性能优化 | workload 设计、profiling、对照实验、Triton Kernel |
| 工程质量 | 接口抽象、测试、CI、文档、可复现 benchmark |

当前仓库已经具备 Prefix Cache、基本 Chunked Prefill、Tensor Parallel 和 CUDA Graph，因此项目重点不是照抄已有功能，而是把它们扩展为可测量、可配置、可服务化的系统。

## 3. 范围约束

### 必做范围

- 单机单卡可完整运行；
- Qwen3-0.6B 作为开发模型；
- 所有优化都有 reference correctness test；
- 所有性能结论写明 GPU、软件版本、模型、输入输出长度与并发；
- 不预先承诺“提升 30%”，先测基线再制定目标；
- 每一阶段保持主分支可运行。

### 暂不纳入首版

- 自研 Attention Kernel；
- 跨节点 KV Cache 传输；
- 完整 Kubernetes 平台；
- 支持大量模型架构；
- 先做界面再做引擎。

这些内容会显著扩大范围，却不一定提高首份 AI Infra 简历的说服力。

## 4. 12 周实施路线

默认投入为每周 10～15 小时。时间不足时优先完成 M0～M5；M6 是冲刺项。

### M0：读懂基线并锁定实验环境（第 1 周）

#### 学习内容

- Prefill 与 Decode 的数据流和计算特征；
- Sequence 状态机；
- block table、physical block、ref count；
- CUDA Graph 的适用条件；
- TTFT、TPOT、ITL、throughput、goodput 的定义。

#### 代码阅读顺序

1. `nanovllm/llm.py`
2. `nanovllm/engine/llm_engine.py`
3. `nanovllm/engine/scheduler.py`
4. `nanovllm/engine/block_manager.py`
5. `nanovllm/engine/model_runner.py`
6. `nanovllm/layers/attention.py`

#### 实施任务

- [ ] 画出一条请求从 `add_request` 到返回 token 的调用链；
- [ ] 给 Scheduler、BlockManager 写最小单元测试；
- [ ] 固化环境：GPU、驱动、CUDA、PyTorch、Triton、模型 commit；
- [ ] 重构 `bench.py`，让随机种子、并发、输入/输出长度成为参数；
- [ ] 输出机器可读的 JSON 结果，而不只是终端日志。

#### 验收门槛

- 相同随机种子可以复现实验；
- 能口述一次 prefill、decode、preemption 和 block 回收；
- 基线结果至少重复 5 次，报告均值、标准差和冷启动处理方式。

#### 反馈材料

- 一张请求链路图；
- 一份 `baseline.json`；
- 你最不确定的三个源码问题。

### M1：可观测性与真实 workload（第 2～3 周）

#### 学习内容

- 开环与闭环压测；
- 到达率、并发数与排队延迟；
- 分位数指标及 coordinated omission；
- Zipf 分布和真实共享前缀分布。

#### 实施任务

- [ ] 为 Sequence 记录 arrival、scheduled、first-token、finished 时间；
- [ ] 记录 prompt/output tokens、cached tokens、preemptions；
- [ ] 构造三类 workload：随机前缀、客服共享前缀、Agent 工具前缀；
- [ ] 支持 prefix pool、prefix length、共享比例、请求率和并发度；
- [ ] 输出 TTFT、TPOT、E2E 的 P50/P95/P99；
- [ ] 定义 goodput：同时满足 TTFT 与 TPOT SLO 的请求速率。

#### 验收门槛

- 共享比例为 0 时，cache hit 应接近 0；
- 共享比例上升时，命中率趋势必须能解释；
- 每个指标能追溯到原始 request-level 记录；
- workload generator 与引擎解耦，可用于对比 vLLM。

### M2：可插拔 KV Cache 策略（第 4～5 周）

#### 学习内容

- vLLM/SGLang 的 prefix caching 思路；
- block hash 链、引用计数和碰撞校验；
- LRU、LFU、cost-aware eviction；
- 多租户缓存污染与配额。

#### 实施任务

- [ ] 抽象 `CachePolicy`，让分配机制与淘汰策略解耦；
- [ ] 保留当前策略作为 baseline；
- [ ] 实现 LRU；
- [ ] 实现 prefix-aware cost 策略，例如 `reuse_probability × recompute_tokens`；
- [ ] 增加 tenant_id，并实现简单软配额；
- [ ] 增加 hash collision、共享引用、回收和压力测试；
- [ ] 统计 block-level hit、token-level hit、eviction 和 wasted cache。

#### 关键不变量

- `ref_count > 0` 的 block 不能被淘汰；
- 一个 physical block 只能表示一份确定的 token prefix；
- 异常、取消和 preemption 后不能泄漏 block；
- 缓存命中不能改变模型输出。

#### 验收门槛

- 运行随机状态机测试后，free + used block 总数恒定；
- 相同输入、相同采样条件下，开启和关闭缓存结果一致；
- 至少找到一种 workload，使新策略明显优于 LRU；
- 同时公开新策略退化的 workload，不隐藏负面结果。

### M3：SLO 感知调度器（第 6～7 周）

#### 学习内容

- FCFS、SRPT、priority aging；
- head-of-line blocking；
- chunked prefill token budget；
- 吞吐、TTFT、TPOT 与公平性的冲突。

#### 实施任务

- [ ] 将当前“仅第一个请求允许 chunk”的逻辑改成明确的 token-budget scheduler；
- [ ] 给 Request 增加 priority、deadline、tenant_id；
- [ ] 实现 FCFS 基线；
- [ ] 实现带 aging 的优先级调度，确保低优先级请求不会永久饥饿；
- [ ] 将 cached token savings 纳入调度分数；
- [ ] 增加取消、超时和 admission control；
- [ ] 构造短请求与长 prompt 混合 workload。

#### 验收门槛

- 在过载条件下不会 OOM；
- 无请求永久饥饿；
- 报告必须同时展示 throughput、P99 TTFT、P99 TPOT、goodput；
- 给出“缓存亲和”和“公平性”之间的至少一个反例。

### M4：在线服务化（第 8 周）

#### 学习内容

- OpenAI Chat Completions 的最小协议；
- SSE streaming；
- backpressure、disconnect、request cancellation；
- readiness 与 liveness 的区别。

#### 实施任务

- [ ] 将离线 `generate` 拆成异步提交与增量取回；
- [ ] 增加 `/v1/chat/completions`；
- [ ] 支持 streaming 和 client disconnect；
- [ ] 增加 `/healthz`、`/readyz`、`/metrics`；
- [ ] 支持 request_id、tenant_id、priority；
- [ ] 用固定并发压测服务，确认断连后资源被回收。

#### 验收门槛

- 可以用标准 OpenAI Python client 请求；
- 首 token 能被流式返回；
- 客户端中断后 Sequence 与 KV Block 均能释放；
- 服务过载时明确拒绝或排队，不能无界积压。

### M5：Profiling 驱动的 Triton 优化（第 9～10 周）

#### 学习内容

- GPU memory hierarchy、warp、occupancy；
- arithmetic intensity 与 Roofline；
- Nsight Systems/Compute；
- Triton program、block size、mask 和 autotune。

#### 实施任务

- [ ] 先 profile 端到端请求，列出耗时、launch 次数和同步点；
- [ ] 从 `RMSNorm` 或 `Add + RMSNorm` 选择一个真实热点；
- [ ] 写 PyTorch reference 和 correctness matrix；
- [ ] 实现 Triton Kernel，并覆盖不同 batch、token、hidden size；
- [ ] 对比 eager、`torch.compile` 和 Triton；
- [ ] 把 Kernel 接回真实模型，再测端到端收益。

#### 验收门槛

- 覆盖 FP16/BF16、非 2 次幂 shape 和边界条件；
- 给出数值误差标准；
- 同时报告 microbenchmark 和 end-to-end；
- 如果端到端没有收益，能用数据说明原因，而不是只展示微基准。

### M6：对标、开源与面试包装（第 11～12 周）

#### 实施任务

- [ ] 使用同一 workload 对比 Nano-vLLM baseline、CacheFlow-Nano、vLLM；
- [ ] 至少覆盖低并发、高并发、短 prompt、长共享前缀四种场景；
- [ ] 增加 CI：单元测试、静态检查和无 GPU 测试；
- [ ] 完成 architecture、benchmark、design-decisions 文档；
- [ ] 录制 5 分钟 demo；
- [ ] 从项目中拆出一个通用修复，尝试向上游提交 PR；
- [ ] 写一页简历项目描述和 10 个深挖问题答案。

#### Definition of Done

- 新机器按照 README 可以复现实验；
- 性能数据附原始 JSON 和生成脚本；
- 至少 20 个有意义的测试；
- 所有性能结论包含硬件、版本、workload 和基线；
- 能解释三次失败的优化尝试；
- 有一个公开 PR、Issue、技术文章或设计讨论作为外部证据。

## 5. 硬件不足时的替代路线

### 只有一张消费级 GPU

- 以 Qwen3-0.6B 开发；
- 重点做 Scheduler、Cache Policy、服务化和 trace replay；
- Kernel 只测设备支持的 FP16/BF16；
- 分布式部分写 simulator，不伪造多卡结果。

### 有两张 GPU

- 增加 TP=2 对照；
- 记录 NCCL collective 时间；
- 研究 batch size 对计算/通信比例的影响；
- 不必立即做跨节点。

### 暂时没有 NVIDIA GPU

- 先完成 Scheduler、BlockManager 的纯 CPU 状态机测试；
- 把 workload、metrics、policy simulator 做完；
- GPU benchmark 明确标为待补，不使用估算数据冒充实测。

## 6. 每周协作与实时反馈方式

每周只维护一个当前 milestone。完成一个 checkpoint 后，把下面模板直接发给 Codex：

```text
阶段：M1 / 可观测性
本周 commit：<hash 或 diff>
运行环境：<GPU、CUDA、PyTorch、模型>
完成项：<最多 5 条>
核心数据：<baseline 与当前结果，附命令或 JSON>
失败/异常：<现象、最小复现、已排除项>
我的解释：<为什么出现这个结果>
希望评审：<代码 / 实验设计 / 原理 / 下一步>
投入时间：<小时>
```

反馈时按以下顺序处理：

1. 先检查正确性和实验有效性；
2. 再做代码 review；
3. 再判断是否达到阶段验收门槛；
4. 通过后才解锁下一阶段；
5. 如果实际结果推翻原计划，就修改路线，不为赶进度跳过证据。

## 7. 第一次反馈前的具体任务

先完成一个 2～4 小时的小 checkpoint，不要直接开始改调度器：

- [ ] 阅读 `scheduler.py`、`block_manager.py` 和 `sequence.py`；
- [ ] 手画或用 Mermaid 画出 waiting → running → finished / preempted 状态变化；
- [ ] 回答下面 5 个问题：
  1. 为什么 `can_allocate` 不缓存最后一个不完整 block？
  2. `ref_count` 与 `used_block_ids` 分别表达什么？
  3. 当前什么情况下会发生 preemption？
  4. 当前 chunked prefill 为什么只允许队首请求使用剩余 token budget？
  5. `hash_blocks` 在什么时候才能安全地登记一个 block？
- [ ] 运行一次现有 `bench.py`，记录完整环境和结果；
- [ ] 不修改现有的 `example.py` 本地改动。

第一次反馈不要求答案全对。它的目的，是判断后续应该先补 Transformer/CUDA 基础，还是直接进入引擎实验。

## 8. 最终简历表述模板

不要现在填写提升数字。项目结束后，用真实数据替换占位符：

> 设计并实现面向 RAG/Agent 共享前缀负载的 SLO 感知 LLM 推理服务，扩展 Nano-vLLM 的 Paged KV Cache 与 Chunked Prefill；实现 prefix-aware 缓存淘汰、带 aging 的优先级调度及 OpenAI-compatible 流式 API。在 `<GPU/model/workload>` 下，相对 `<baseline>` 将 token cache hit 从 `<A>` 提升至 `<B>`，P99 TTFT 降低 `<C>`，SLO goodput 提升 `<D>`；使用 Nsight 定位 `<瓶颈>` 并通过 Triton `<算子>` 获得 `<micro/end-to-end>` 加速。项目包含可复现实验、正确性测试和 `<上游 PR/Issue>`。

