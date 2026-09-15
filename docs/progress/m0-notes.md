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
(.venv311) root@LAPTOP-26T3VEES:/mnt/e/Project/Nano-vllm# python --version
Python 3.11.15
(.venv311) root@LAPTOP-26T3VEES:/mnt/e/Project/Nano-vllm# python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_availabl
e()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
torch= 2.5.1+cu124
cuda= True
gpu= NVIDIA GeForce RTX 3060 Laptop GPU
```

## 2. example.py 运行结果

执行命令：

```powershell
python example.py
```

结果或错误：

```text
Generating: 100%|████████████████████████████████████████████████████████████████████████████| 3/3 [00:25<00:00,  8.38s/it, Prefill=6tok/s, Decode=28tok/s]


Prompt: '<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n'
Completion: "<think>\nOkay, the user wants me to introduce myself. Let me start by recalling who I am. I'm an AI assistant designed to help with various tasks like answering questions, providing information, and offering support. I should mention my capabilities to give them an idea of what I can do.\n\nI need to keep the tone friendly and approachable. Maybe start with a greeting, then explain my purpose. It's important to highlight that I'm here to assist them, not just answer questions. Also, I should mention my availability, like being available 24/7. \n\nWait, should I include any specific features or tools I can use? Probably not, but I should make sure the introduction covers the main points without overcomplicating. Keep it concise and clear. Let me check if there's anything else I need to add. No, that should cover it. Now, put it all together in a natural way.\n</think>\n\nHello! I'm an AI assistant designed to help with a wide range of tasks, from answering questions to providing information, and offering support. I'm available 24/7 to assist you with whatever you need. Let me know how I can help!<|im_end|>"


Prompt: '<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n'
Completion: "<think>\nOkay, so I need to list all the prime numbers between 100. Let me think about how to approach this. First, I remember that a prime number is a number greater than 1 that has no positive divisors other than 1 and itself. So, starting from 100, I need to check each number to see if it's prime.\n\nLet me start by recalling some prime numbers. The first few primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97. These are all primes up to 100. But wait, I need to make sure that every number up to 100 is checked, not just the ones I already know. \n\nLet me start counting from 100. The next prime after 97 is 101, which is outside the range"


Prompt: '<|im_start|>user\n你是什么模型？<|im_end|>\n<|im_start|>assistant\n'
Completion: '<think>\n嗯，用户问：“你是什么模型？”我需要先确定用户的问题意图。可能他们想了解我的基本信息，或者想确认我的角色。作为AI助手，我应该提供一个清晰的回答，同时保持友好和专业的形象。\n\n首先，我应该明确说明我是基于什么技术构建的，比如基于Transformer架构，使用了大量预训练数据，这样用户能了解技术基础。然后，我需要回答模型的目的，比如帮助用户解决问题，提供信息等。\n\n另外，用户可能对模型的用途感兴趣，所以可以简单介绍一下应用场景，比如学术、商业或日常交流。同时，要确保回答简洁明了，避免使用过于专业的术语，让用户容易理解。\n\n最后，检查回答是否符合所有要求，确保没有遗漏任何关键点，同时保持自然流畅的表达。\n</think>\n\n我是基于Transformer架构构建的AI模型，基于大量预训练数据进行优化。我的主要功能包括回答问题、提供信息、协助学习和工作等。如果您有任何问题或需要帮助，请随时告诉我！<|im_end|>'
```

## 3. Sequence 状态机

### 我看到的三种状态

- WAITING：表示 Sequence 正在等待调度。它可能从未执行过 Prefill，也可能只完成了部分 Prefill，还可能是被抢占后重新回到等待队列。
- RUNNING：表示 Prompt 已经完成 Prefill，Sequence 已进入运行队列，可以参与后续 Decode
- FINISHED： 表示 Sequence 因生成 EOS 或达到 max_tokens 而结束，并已经释放关联的 KV Cache Block

### 请求状态变化

```text
请按照自己的理解补全：
add_request -> waiting -> 进入prefill阶段 -> prefill阶段完成 ->进入decode阶段 ->running -> 直到decode结束 ->finished

running -> preempt ->waiting
```

## 4. BlockManager 概念

### `free_block_ids` 表示什么？

我的理解：当前没有被任何活跃 Sequence 引用、可以被重新分配的 Block。

### `used_block_ids` 表示什么？

我的理解：正在被seq引用的block

### `ref_count` 表示什么？

我的理解：当前同时有多少个活跃 Sequence 引用这个物理 Block。

### `hash_to_block_id` 表示什么？

我的理解：将hash 值转化为 block_id

## 5. 五个源码问题

### 问题 1：为什么 `can_allocate` 不缓存最后一个不完整 Block？

我的回答：如果多个 Sequence 共享它，不同请求的后续写入可能互相影响，使 Token、Hash 和 KV Cache 内容失去一致性。因此当前实现只复用稳定的完整 Block。

### 问题 2：`ref_count` 与 `used_block_ids` 分别表达什么？

我的回答：ref_count 表示当前同时引用该物理 Block 的活跃 Sequence 数量，used_block_ids表示的是哪些block已经被活跃的seq所引用
### 问题 3：当前什么情况下会发生 preemption？

我的回答：当处于running队首的seq进行decode 阶段发现KVCache不够用时，会抢占处于running队尾的seq

### 问题 4：当前 Chunked Prefill 为什么只允许队首请求使用剩余 Token Budget？

我的回答：scheduled_seqs 为空时，第一个请求可以只执行一部分 Prefill；
Batch 中已经存在请求时，后续请求如果无法完整放入剩余 Token Budget，就不再切分；
这样保证一个 Batch 最多只有一个部分 Prefill 请求。

### 问题 5：`hash_blocks` 在什么时候才能安全登记一个 Block？

我的回答：当 Block 已经填满，并且该 Block 对应 Token 的 Forward 已经完成、K/V 已写入 KV Cache 后，才能安全登记。整个请求不需要结束。

## 6. 我最不理解的地方

1. continuous batching体现在哪里？
  答：当前 generate() 会在开始时一次性加入全部请求，因此它实现了动态批处理核心，但还不能在运行过程中持续接收新的在线请求。每轮batch都重新调度，每次调度都是动态组成batch，每次请求完成都立即移除
2. 
3. 

## 7. 本次投入时间

- 开始时间：
- 结束时间：
- 大约用时：2h

