# 从这里开始：CacheFlow-Nano 零基础操作指南

这份文档只带你完成第一次学习，不要求你预先会 Git、CUDA 或项目开发。

## 0. 先认识你现在所在的位置

当前项目文件夹是：

```text
E:\Project\Nano-vllm
```

它已经是一个 Git 仓库，并且连接到了原始 Nano-vLLM 项目：

```text
当前分支：main
远程仓库：https://github.com/GeeeekExplorer/nano-vllm.git
```

因此：

- 不要运行 `git init`；
- 不需要重新下载 Nano-vLLM；
- Git 仓库可以理解为“能够记录每一次代码修改的项目文件夹”；
- 后续会创建一个自己的学习分支，但第一次阅读源码不需要立即创建。

当前工作区里已经存在一些本地改动：

```text
example.py    已修改
.wheels/      未被 Git 跟踪
models/       未被 Git 跟踪
run.bat       未被 Git 跟踪
docs/         未被 Git 跟踪
```

不要删除或覆盖这些内容，也暂时不要执行 `git add .`，否则本地模型可能被错误加入 Git。

## 1. 在哪里打开路线图

在 Codex 回复中点击下面这个文件链接：

```text
E:\Project\Nano-vllm\docs\ai-infra-roadmap.md
```

文件会在 Codex 的文件面板中打开。它是完整的 12 周路线，不需要一次读完。

第一次只阅读其中两节：

1. `M0：读懂基线并锁定实验环境`；
2. `第一次反馈前的具体任务`。

本文件 `START-HERE.md` 是操作手册，`ai-infra-roadmap.md` 是项目总路线。

## 2. 第一次需要打开哪些文件

按下面顺序打开，每次只看一个：

### 文件 1：`nanovllm/engine/sequence.py`

完整路径：

```text
E:\Project\Nano-vllm\nanovllm\engine\sequence.py
```

暂时不要编辑。阅读时寻找：

- `SequenceStatus` 有哪些状态；
- `Sequence.__init__` 保存了哪些信息；
- `append_token` 做了什么；
- `num_blocks` 如何计算；
- `block(i)` 如何取得第 i 个逻辑块。

把答案写入：

```text
E:\Project\Nano-vllm\docs\progress\m0-notes.md
```

### 文件 2：`nanovllm/engine/block_manager.py`

完整路径：

```text
E:\Project\Nano-vllm\nanovllm\engine\block_manager.py
```

暂时不要编辑。重点理解：

- `blocks`：全部物理 KV Block；
- `free_block_ids`：当前空闲 Block；
- `used_block_ids`：当前正在使用的 Block；
- `hash_to_block_id`：可复用前缀到物理 Block 的索引；
- `ref_count`：有多少 Sequence 正在引用该 Block；
- `allocate`、`deallocate`、`hash_blocks` 的调用关系。

### 文件 3：`nanovllm/engine/scheduler.py`

完整路径：

```text
E:\Project\Nano-vllm\nanovllm\engine\scheduler.py
```

暂时不要编辑。重点看 `schedule()` 的两个部分：

```text
prefill 调度
    ↓
如果本轮没有 prefill
    ↓
decode 调度
```

阅读时跟踪一个请求：

```text
add()
  → waiting
  → allocate KV Block
  → prefill
  → running
  → 多轮 decode
  → finished
  → deallocate KV Block
```

### 文件 4：`nanovllm/engine/llm_engine.py`

完整路径：

```text
E:\Project\Nano-vllm\nanovllm\engine\llm_engine.py
```

重点看：

- `add_request()` 如何创建 Sequence；
- `step()` 如何调用 Scheduler 和 ModelRunner；
- `generate()` 为什么会不断调用 `step()`。

### 文件 5：`bench.py`

完整路径：

```text
E:\Project\Nano-vllm\bench.py
```

这是 M0 后半段才会编辑的文件。第一次阅读源码时先不改它。

## 3. 如何在 Codex 中打开文件

有三种方式，选择其中一种即可：

### 方法 A：点击 Codex 回复中的文件链接

这是最简单的方法。点击文件名后，它会在 Codex 文件面板中打开。

### 方法 B：使用项目文件列表

在 Codex 项目窗口的文件列表中依次展开：

```text
Nano-vllm
  └─ nanovllm
      └─ engine
          ├─ sequence.py
          ├─ block_manager.py
          ├─ scheduler.py
          └─ llm_engine.py
```

### 方法 C：直接告诉 Codex

你可以发送：

```text
帮我打开 sequence.py，并从 SequenceStatus 开始带我阅读。
```

## 4. 第一次如何运行项目

Nano-vLLM 不能直接使用当前 Windows PowerShell 中的 Python。已经验证可用的推理环境位于：

```text
WSL 发行版：nano-ubuntu
虚拟环境：/root/nano-vllm/.venv311
Python：3.11.15
PyTorch：2.5.1+cu124
GPU：NVIDIA GeForce RTX 3060 Laptop GPU
```

### 第一步：打开终端并进入正确的 WSL

在 Codex 中打开 Terminal 面板。最初看到的通常是 Windows PowerShell：

```text
PS E:\Project\Nano-vllm>
```

在 PowerShell 中输入：

```powershell
wsl -d nano-ubuntu -u root
```

进入成功后，提示符会变成类似：

```text
root@电脑名:/mnt/e/Project/Nano-vllm#
```

从这一步开始，下面的命令都是 Linux 命令，不要再复制 `PS>` 等提示符。

### 第二步：进入项目并激活虚拟环境

```bash
cd /mnt/e/Project/Nano-vllm
```

```bash
source /root/nano-vllm/.venv311/bin/activate
```

成功后，提示符开头通常会出现：

```text
(.venv311)
```

检查当前 Python：

```bash
which python
python --version
```

应该得到：

```text
/root/nano-vllm/.venv311/bin/python
Python 3.11.15
```

### 第三步：确认 PyTorch 和 GPU

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

已验证的结果是：

```text
torch= 2.5.1+cu124
cuda= True
gpu= NVIDIA GeForce RTX 3060 Laptop GPU
```

把终端输出复制到 `m0-notes.md`。如果结果不同，不要自己反复安装依赖，直接把错误原文发给 Codex。

### 第四步：运行最小示例

确认提示符仍以 `(.venv311)` 开头，然后运行：

```bash
python example.py
```

当前 `example.py` 已经指向项目内的 `models/Qwen3-0.6B`。实际验证中，RTX 3060 6GB 可以成功完成三个 prompt 的生成。

第一次运行可能包含模型加载、编译或显存初始化，因此较慢是正常现象。需要中止时按 `Ctrl+C`。

成功标准：

- 没有 Python traceback；
- GPU 可用；
- 三个 prompt 都产生了输出。

如果运行成功，把耗时和输出摘要写进 `m0-notes.md`。如果失败，复制从 `Traceback` 开始的完整错误。

### 第五步：暂时不要直接运行 `bench.py`

当前 `bench.py` 使用的是：

```python
~/huggingface/Qwen3-0.6B/
```

而你当前模型位于：

```text
E:\Project\Nano-vllm\models\Qwen3-0.6B
```

因此第一次先运行 `example.py` 验证环境。等源码阅读完成后，我们再一起把 `bench.py` 改造成可以指定模型、随机种子和输出 JSON 的版本。

### 第六步：退出 WSL

运行完成后可以输入：

```bash
deactivate
exit
```

终端将回到 Windows PowerShell。

## 5. 第一次允许编辑什么

第一次学习只编辑一个文件：

```text
E:\Project\Nano-vllm\docs\progress\m0-notes.md
```

暂时不要编辑：

```text
nanovllm/engine/sequence.py
nanovllm/engine/block_manager.py
nanovllm/engine/scheduler.py
nanovllm/engine/llm_engine.py
```

原因是先建立正确的执行模型，再开始修改核心逻辑。否则程序即使能运行，也很难判断行为是否正确。

## 6. Git 到底什么时候使用

### 现在不需要创建 Git 仓库

项目已经是 Git 仓库，不运行：

```powershell
git init
```

### 什么时候创建自己的分支

完成源码阅读、准备第一次修改 `bench.py` 前，再创建学习分支：

```powershell
git switch -c codex/cacheflow-learning
```

“分支”可以理解为一条独立修改路线。这样你的学习代码不会直接混入 `main`。

创建分支不会自动删除当前本地文件，但由于当前已经存在本地改动，执行前最好先把 `git status` 的结果发给 Codex确认。

### 第一次提交时不要使用 `git add .`

以后修改了 `bench.py` 和学习文档，只添加明确的文件：

```powershell
git add bench.py docs\progress\m0-notes.md
```

查看将要提交的内容：

```powershell
git diff --cached
```

确认无误后提交：

```powershell
git commit -m "bench: add reproducible baseline"
```

暂时不要把下面这些目录加入提交：

```text
models/
.wheels/
```

模型权重通常很大，不应该直接提交到普通 Git 仓库。

## 7. 你今天只需要完成这些

- [ ] 打开 `sequence.py`；
- [ ] 打开 `block_manager.py`；
- [ ] 打开 `scheduler.py`；
- [ ] 在 `m0-notes.md` 中填写初步理解；
- [ ] 在终端确认 Python 和 CUDA；
- [ ] 运行 `python example.py`；
- [ ] 把结果或完整错误发给 Codex。

不要求今天运行 benchmark，不要求写 CUDA，也不要求提交 Git。

## 8. 第一次反馈可以这样发

```text
我完成了 START-HERE 的第一次任务。

Python/CUDA 检查结果：
<粘贴输出>

example.py 运行结果：
<成功摘要或完整错误>

m0-notes.md：
<请检查这个文件>

我最不理解的是：
1. ...
2. ...
```

收到反馈后，下一步才是一起修改 `bench.py`。届时会逐行说明创建哪些参数、为什么需要 JSON，以及怎样进行第一次 Git 提交。
