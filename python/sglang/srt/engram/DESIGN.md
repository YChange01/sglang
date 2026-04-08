# Engram Remote Embedding Pool via UB/URMA

## 1. 目标

仿照论文 [arXiv:2603.10087](https://arxiv.org/abs/2603.10087) 的 CXL 方案，
基于 **SGLang + yuanrong-datasystem (UB/URMA)** 实现 Engram embedding 表的
跨节点小包读取，实现推理时 GPU 显存卸载 + 异步预取。

## 2. 设计思路

### 2.1 论文方案 vs 本方案

| 维度 | 论文 (CXL) | 本方案 (UB/URMA) |
|------|-----------|-----------------|
| 存储 | Samsung CXL DIMM | yuanrong-datasystem Worker 共享内存 |
| 传输 | load/store via DAX mmap | KVClient → URMA urma_write |
| GPU传输 | CUDA kernel 注册 CXL 为 host mem | torch CPU→GPU copy / DsTensorClient |
| 框架 | SGLang ModelRunner | SGLang ModelRunner (相同) |
| 场景 | Engram 稀疏只读查找 (~5KB/token/layer) | 相同 |

### 2.2 核心设计：异步预取与计算重叠

```
时间线 →
──────────────────────────────────────────────
Transformer L0-L1:  [======= GPU 计算 =======]
Engram Prefetch:       [KVClient.get() via URMA]  ← 与上面并行
                                               ↓ wait
Engram Layer:                              [gather + project]
Transformer L3+:                             [======= GPU 计算 =======]
──────────────────────────────────────────────
```

## 3. 文件变更清单

### 新增文件（6个核心 + 4个测试 + 1个benchmark）

```
sglang/python/sglang/srt/engram/
├── __init__.py                  # 包入口
├── engram_config.py             # 配置 dataclass (EngramPoolConfig)
├── engram_pool.py               # 远程池客户端 (封装 KVClient)
├── engram_prefetcher.py         # 异步预取引擎 (ThreadPoolExecutor)
├── engram_embedding.py          # RemoteEngramEmbedding 层
├── engram_utils.py              # N-gram 哈希索引计算工具
└── DESIGN.md                    # 本文档

sglang/test/srt/engram/
├── __init__.py
├── test_engram_pool.py          # 单元测试: 池连接/读写
├── test_engram_prefetch.py      # 集成测试: 预取流水线
└── test_engram_e2e.py           # E2E测试: 输出一致性

sglang/benchmark/engram/
└── bench_urma_small_packet.py   # 性能基准: 延迟/吞吐/预取开销
```

### 修改的现有文件（3个）

#### 1. `server_args.py` — 新增 Engram 配置参数

```diff
+ engram_pool_enabled: bool = False
+ engram_pool_hosts: str = ""           # Worker IP 列表 (逗号分隔)
+ engram_pool_ports: str = ""           # Worker 端口列表
+ engram_pool_etcd: str = ""            # ETCD 地址
+ engram_pool_timeout_ms: int = 60000   # 连接超时
+ engram_pool_req_timeout_ms: int = 5000  # 请求超时
+ engram_pool_prefetch_ahead: int = 2   # 预取提前层数
+ engram_pool_num_workers: int = 4      # 预取线程数
+ engram_pool_chunk_size: int = 1       # 每个KV存多少行
+ engram_pool_key_prefix: str = "engram"
+ engram_pool_shard_strategy: str = "hash"
+ engram_pool_cross_node: bool = False
```

#### 2. `forward_batch_info.py` — ForwardBatch 新增字段

```diff
+ from typing import Any  # 新增导入
  ...
  class ForwardBatch:
      ...
+     engram_prefetch_request: Optional[Any] = None
```

#### 3. `model_runner.py` — 集成初始化 + 预取触发

**修改点 A**: `maybe_init_ngram_embedding()` 末尾新增:
```python
# 初始化 Engram 远程池
self.engram_pool = None
self.engram_prefetcher = None
self.engram_embedding_module = None
self._maybe_init_engram_pool()
```

**新增方法 `_maybe_init_engram_pool()`**:
- 从 server_args 构建 EngramPoolConfig
- 创建 EngramPool 并连接 Workers
- 创建 EngramPrefetcher
- 找到 RemoteEngramEmbedding 模块并注入 prefetcher
- Rank 0 将权重加载到远程池，然后释放 GPU 显存

**新增方法 `_load_engram_tables_to_pool()`**:
- 遍历 N-gram 子表
- 调用 pool.load_table() 写入远程 Worker
- 调用 module.offload_oe_table() 释放 GPU 显存

**修改点 B**: `_forward_raw()` 中，在 decode/extend dispatch 之前:
```python
# 触发 Engram 异步预取 (与后续 Transformer 计算重叠)
if self.engram_embedding_module is not None:
    forward_batch.engram_prefetch_request = (
        self.engram_embedding_module.trigger_prefetch(forward_batch)
    )
```

## 4. 各模块详解

### 4.1 EngramPoolConfig (engram_config.py)

配置类，支持从 `server_args` 构建。关键配置:
- Worker 连接信息 (hosts/ports)
- 表结构 (num_tables/embedding_dim/table_capacity)
- 传输参数 (timeout/chunk_size)
- 预取策略 (prefetch_ahead/num_prefetch_workers)
- 分片策略 (hash/range)

### 4.2 EngramPool (engram_pool.py)

远程池客户端，封装 `yr.datasystem.KVClient`。

**存储格式**:
- Key: `{prefix}:t{table_id}:c{chunk_id}`
- Value: 连续 `chunk_size` 行 embedding 的 float32 字节

**分片**: `(table_id * 1000003 + chunk_id) % num_workers`

**核心方法**:
- `load_table(table_id, weights)` — 将 PyTorch tensor 写入远程池
- `batch_lookup(table_id, row_ids)` — 同步批量查找，返回 numpy
- `batch_lookup_async(table_id, row_ids)` — 异步版本，返回 Future

**优化**:
- 去重 chunk_id，合并同 Worker 请求
- ThreadPoolExecutor 并行读不同 Worker
- 遵守 KVClient 2000-key 批量限制

### 4.3 EngramPrefetcher (engram_prefetcher.py)

异步预取引擎，管理 prefetch 生命周期。

**API**:
- `prefetch_async(table_indices, seq_len, device)` → `PrefetchRequest`
- `wait_and_gather(request)` → `torch.Tensor [num_tables, seq_len, embed_dim]`

**流程**:
1. 对每个子表启动 `pool.batch_lookup_async()`
2. 返回 PrefetchRequest (持有 Future 列表)
3. 模型继续做 Transformer 计算
4. 在 Engram 层之前调用 `wait_and_gather()`:
   - 等待所有 Future 完成
   - 组装 numpy → torch.Tensor → .to(device)

### 4.4 RemoteEngramEmbedding (engram_embedding.py)

替代原有 `NgramEmbedding`，区别:
- **不持有** 大 embedding 表 (`oe_embeder`)
- 保留 word_embeder (标准词表 embedding) + oe_projection (投影矩阵)
- `trigger_prefetch()` — 计算 N-gram 哈希索引，触发异步预取
- `forward()` — 使用预取结果做 BMM 投影

**权重加载兼容**: `load_weight()` 保持与 NgramEmbedding 相同接口

### 4.5 engram_utils.py

N-gram 哈希索引计算工具:
- `compute_ngram_hash_indices()` — 复用 SGLang JIT kernel
- `split_indices_by_table()` — 将统一索引拆分为 per-table local row IDs

## 5. 数据流全景

```
Input: ForwardBatch { input_ids, req_pool_indices, seq_lens, ... }
                |
                v
   ┌─ ModelRunner._forward_raw() ─────────────────────────┐
   │                                                       │
   │  1. trigger_prefetch(forward_batch)                   │
   │     ├─ compute_n_gram_ids(input_ids) → ngram_ids     │
   │     ├─ split_indices_by_table(ngram_ids) → per-table │
   │     └─ pool.batch_lookup_async() × N tables          │
   │         → PrefetchRequest (Future 列表)              │
   │         → 存入 forward_batch.engram_prefetch_request  │
   │                                                       │
   │  2. model.forward(input_ids, forward_batch)           │
   │     ├─ RemoteEngramEmbedding.forward()               │
   │     │   ├─ word_embeder(input_ids) → word_hidden     │
   │     │   ├─ prefetcher.wait_and_gather(request)       │
   │     │   │   └─ future.result() × N tables            │
   │     │   │   └─ numpy → torch.Tensor → GPU            │
   │     │   ├─ torch.bmm(oe_hidden, oe_projection)       │
   │     │   └─ mean(word_hidden, projected)               │
   │     ├─ Transformer Layer 0 ...                        │
   │     ├─ Transformer Layer 1 ...                        │
   │     └─ ... (N layers)                                 │
   │                                                       │
   │  3. sample(logits) → next_token_ids                   │
   └───────────────────────────────────────────────────────┘
```

## 6. 启动方式

```bash
# 1. 启动 yuanrong-datasystem Workers
dscli start -w 10.0.0.1:18482
dscli start -w 10.0.0.2:18482

# 2. 启动 SGLang 并启用 Engram 池
python -m sglang.launch_server \
    --model-path <model_with_ngram_embedding> \
    --engram-pool-enabled \
    --engram-pool-hosts "10.0.0.1,10.0.0.2" \
    --engram-pool-ports "18482,18482" \
    --chunked-prefill-size 8192
```

## 7. 测试方案

| 级别 | 文件 | 内容 | 环境要求 |
|------|------|------|---------|
| 单元 | test_engram_pool.py | Pool 连接/读写/分片 | Mock (无需硬件) |
| 集成 | test_engram_prefetch.py | 预取重叠/并发/GPU输出 | Mock + 可选 GPU |
| E2E | test_engram_e2e.py | 输出一致性对比 | Mock |
| Live | test_engram_pool.py (Live) | 真实 Worker 延迟测试 | ENGRAM_LIVE_TEST=1 + Worker |
| Bench | bench_urma_small_packet.py | 延迟/吞吐/预取开销 | Worker + 可选 UB 硬件 |

## 8. 后续优化方向

1. **GPU Direct**: 使用 DsTensorClient.async_mget_h2d() 直接传到 GPU，跳过 CPU 中转
2. **热点缓存**: 本地 LRU 缓存高频 embedding 行，减少远程读取
3. **流式预取**: 按子表流水线式 gather，不等所有表完成
4. **CUDA Kernel**: 自定义 CUDA kernel 做 scatter-gather，避免 numpy 中间格式
5. **KV Cache 共存**: 在同一个 yuanrong-datasystem 池中管理 Engram + KV Cache
