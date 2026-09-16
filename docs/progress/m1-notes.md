# M1 学习记录

## M1-0：Scheduler

### 问题 1：为什么本轮一旦成功调度 Prefill，就不会在同一个 Batch 中继续调度 Decode？

我的回答：当前 Scheduler 优先尝试调度 waiting 队列。如果本轮至少成功调度了一个 Prefill 请求，scheduled_seqs 就不为空，代码会立即返回 `(scheduled_seqs, True)`，因此同一个 Batch 不会再进入 Decode 调度。但如果 waiting 队列中的请求因为 KV Cache 不足等原因无法执行，scheduled_seqs 仍为空，Scheduler 仍可能继续调度 running 队列中的 Decode 请求。

### 问题 2：为什么只有 Batch 中第一个请求允许部分 Prefill？

我的回答：代码通过 `remaining < num_tokens and scheduled_seqs` 限制部分 Prefill。当 scheduled_seqs 为空时，说明当前请求是本 Batch 第一个请求，即使剩余 Token Budget 放不下整个 Prompt，也可以使用 `min(num_tokens, remaining)` 调度其中一部分；如果 Batch 中已经调度了其他请求，当前请求无法完整放入剩余 Budget 时就直接停止。这样把 Chunked Prefill 限制为单请求 Batch，避免同一 Batch 同时包含完整 Prefill 和部分 Prefill，使状态更新和调度行为更简单、可预测，但代价是部分剩余 Token Budget 可能无法被利用。这是当前 Scheduler 的简化策略，并不是模型执行层面的必然限制。

### 问题 3：一次 preemption 会修改哪些状态和数据结构？

我的回答：Scheduler 会先通过 `pop()` 或 `popleft()` 将被抢占的 Sequence 从 running 队列移除。`preempt()` 随后把它的 status 改成 WAITING，把 is_prefill 改回 True，并调用 BlockManager.deallocate() 释放它引用的物理 Block。释放过程中，各 Block 的 ref_count 会减一；ref_count 变为 0 的 Block 会从 used_block_ids 移除并加入 free_block_ids。Sequence 的 block_table 会被清空，num_cached_tokens 会被清零，最后 Sequence 会通过 appendleft() 放到 waiting 队首。已经生成的 token_ids 不会被删除，之后需要重新计算这些 Token 对应的 KV Cache。