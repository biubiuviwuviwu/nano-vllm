# M0 源码阅读与环境记录

> 第一次学习只编辑这一个文件。不会写的地方保留“暂不理解”即可。

## 1. 我的运行环境

执行命令：

```powershell
python --version
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

输出：

```text
请粘贴终端输出
```

## 2. example.py 运行结果

执行命令：

```powershell
python example.py
```

结果或错误：

```text
请粘贴成功摘要，或者从 Traceback 开始粘贴完整错误
```

## 3. Sequence 状态机

### 我看到的三种状态

- WAITING：
- RUNNING：
- FINISHED：

### 请求状态变化

```text
请按照自己的理解补全：

add_request
  →
  →
  →
```

## 4. BlockManager 概念

### `free_block_ids` 表示什么？

我的理解：

### `used_block_ids` 表示什么？

我的理解：

### `ref_count` 表示什么？

我的理解：

### `hash_to_block_id` 表示什么？

我的理解：

## 5. 五个源码问题

### 问题 1：为什么 `can_allocate` 不缓存最后一个不完整 Block？

我的回答：

### 问题 2：`ref_count` 与 `used_block_ids` 分别表达什么？

我的回答：

### 问题 3：当前什么情况下会发生 preemption？

我的回答：

### 问题 4：当前 Chunked Prefill 为什么只允许队首请求使用剩余 Token Budget？

我的回答：

### 问题 5：`hash_blocks` 在什么时候才能安全登记一个 Block？

我的回答：

## 6. 我最不理解的地方

1. 
2. 
3. 

## 7. 本次投入时间

- 开始时间：
- 结束时间：
- 大约用时：

