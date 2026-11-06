# ASD：算法实现与使用

## 1. 算法定义

ASD 在推测解码的目标验证阶段放宽 token ID 完全匹配条件，通过局部 regret 门限、请求累计预算和每轮近似接受数量控制接受前缀。

设一轮草稿包含 K 个 token，位置 i 从 0 开始。草稿候选为 d_i，目标模型在相同前缀下的首选为 t_i，目标 logits 为 z_i。

```text
regret_i = max(0, z_i[t_i] - z_i[d_i])
value_i  = K - i
```

regret 使用目标模型的 logit 差。温度为 1 的 softmax 下，该值等于目标首选概率与候选概率的对数比；实现直接使用 logits 计算。

### 接受条件

候选 ID 与目标首选相同时，按精确匹配接受，代价为零。ID 不同时，需要同时满足：

```text
regret_i / value_i <= g
本轮截至当前位置的累计 regret <= max(B - 请求已用预算, 0) + 1e-9
本轮截至当前位置的 mismatch 数 <= M
```

其中：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| B | 每个请求整个生成过程的累计 regret 预算 | 8 |
| g | 单位置 regret 与剩余草稿长度之比的上限 | 0.25 |
| M | 每轮最多近似接受的 mismatch 数 | 2 |

从位置 0 开始顺序检查，遇到首个不满足条件的位置即停止。接受连续草稿前缀后，在拒绝位置输出目标 recovery token；全部草稿接受时输出目标 bonus token。

只对实际接受的 mismatch 逐个扣除预算。请求累计预算跨轮保留，M 每轮重新计数。精确匹配、recovery 和 bonus 不消耗近似接受预算。局部分母使用本轮实际草稿长度 K，最大配置长度不足时使用当前块的实际长度。

### 边界行为

- B、g、M 中任意一项为零时，退化为严格 token ID 匹配。
- logit 相同而 token ID 不同仍属于 mismatch，占用 M；其 regret 可以为零。
- 请求重排或抢占后保留累计预算；请求结束释放账本身份，新请求初始化为零。
- 丢弃的验证行和未完成 prefill 的行不提交累计代价。
- EOS、停止字符串和输出长度上限由框架的停止流程处理。

规则参考 `Kissmetothemoon/ASD` 提交 `9426ce7b3b8cafb2d34f675a5bb9db2ebf0259f6` 中的 `src/asd/budget.py`，采用默认后缀价值 `K-i`。当前设备实现使用 FP32，作者 Python 参考在提取 logits 后使用 Python 浮点运算；极接近阈值时可能出现数值边界差异。

## 2. 实现结构

ASD 接受策略与草稿生成器分别维护。验证器接收单链候选、目标 logits、位置和请求身份，不读取草稿模型权重、隐藏层配置或模型专用参数。

```text
草稿生成器
    ↓ 候选 token
目标模型验证
    ↓ logits、位置、请求身份
通用 Triton top-1 / logit 差计算
    ↓ 目标首选、每位置 regret
ASD 前缀选择与请求预算更新
    ↓ 接受前缀、recovery/bonus、输出数量
框架更新生成状态与 KV cache
```

### 核心文件

路径相对于本仓库根目录。

| 文件 | 作用 |
|---|---|
| `vllm_ascend/envs.py` | 注册 ASD 总开关和 B/g/M 环境变量。 |
| `vllm_ascend/sample/asd.py` | 参数校验、单链接口校验、请求 ID 与预算槽位管理。 |
| `vllm_ascend/ops/triton/spec_decode/logit_score.py` | 通用完整词表 top-1、候选 logit 差和草稿 argmax。 |
| `vllm_ascend/ops/triton/spec_decode/asd_score.py` | 评分函数的兼容导入入口。 |
| `vllm_ascend/ops/triton/spec_decode/asd.py` | 输入收集、V1 打包、前缀选择、恢复位置、输出计数和设备预算提交。 |
| `vllm_ascend/sample/asd_rejection_sampler.py` | V1 sampler 适配。 |
| `vllm_ascend/worker/v2/spec_decode/asd.py` | V2 sampler 适配。 |
| `vllm_ascend/worker/model_runner_v1.py` | V1 sampler 安装、请求生命周期和丢弃行处理。 |
| `vllm_ascend/worker/v2/model_runner.py` | V2 sampler 安装、实际请求状态绑定和账本生命周期。 |
| `vllm_ascend/platform.py` | 配置初始化时检查推测解码、单链和完整目标词表条件。 |

V1 已经分别提取 target、bonus 和草稿行，因此评分使用 `candidate_offset=0`。V2 的输入行对应当前 token，其 logits 预测下一个 token，因此评分使用 `candidate_offset=1`。两种适配最终调用同一个前缀选择内核。

### Triton 执行范围

完整词表 top-1、regret、输入收集、V1 打包、连续前缀选择、recovery/bonus 放置、padding 和预算数值更新使用 Triton。Torch 用于张量分配，Python 管理请求身份及整数元数据；模型前向和框架已有的 logits processors、logprobs 处理保持原执行路径。

词表评分使用每行一个 program、4096 元素分块扫描，采用最小 token ID 处理最大值平局。行数作为运行时参数，减少随批次变化产生的编译版本。草稿 argmax 使用单独入口，不分配用不到的 regret 输出。

## 3. 配置方式

ASD 在服务启动时通过环境变量配置，修改参数后需要重启服务。

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `VLLM_ASCEND_ENABLE_ASD` | `0` | `1` 启用 ASD，`0` 使用严格验证器。 |
| `VLLM_ASCEND_ASD_RISK_BUDGET` | `8` | 请求累计预算 B。 |
| `VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE` | `0.25` | 局部门限 g。 |
| `VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK` | `2` | 每轮 mismatch 上限 M。 |
| `VLLM_USE_V2_MODEL_RUNNER` | 由运行环境决定 | `1` 使用 V2 runner，`0` 使用 V1 runner。 |

B 和 g 必须是可用 FP32 表示的有限非负数，M 必须为非负整数。客户端请求须使用 `temperature=0`。ASD 要求完整目标词表 logits、连续单链布局和 `rejection_sample_method="strict"`；不接受分叉 token tree、`probabilistic` 或 `synthetic` 验证配置。

`VLLM_USE_V1=1` 表示使用 vLLM V1 引擎，`VLLM_USE_V2_MODEL_RUNNER=1` 表示选择该引擎内的 V2 runner，两者可以同时设置。

## 4. DFlash＋ASD 启动示例

### 环境和模型

当前配套版本为 vLLM `v0.20.2` 与 vLLM-Ascend `v0.20.2rc1`，并包含相应的 DFlash V2 接口适配。需要安装本仓库及配套 vLLM 代码，并加载匹配的 CANN、ATB 环境。

配套 vLLM 修改保存在 `patches/vllm-0.20.2-asd.patch`，基线提交为 `bc150f50299199599673614f80d12a196f377655`。在该版本的 vLLM 源码目录中先执行 `git apply --check /实际路径/vllm-ascend/patches/vllm-0.20.2-asd.patch`，检查通过后执行 `git apply /实际路径/vllm-ascend/patches/vllm-0.20.2-asd.patch`。

### 启动脚本

示例脚本：`examples/spec_decode/run_dflash_asd.sh`。先激活已安装本项目的 Python 环境并加载 CANN、ATB，然后从仓库根目录执行：

```bash
bash examples/spec_decode/run_dflash_asd.sh \
  /home/ma-user/work/models/Qwen3-32B \
  /home/ma-user/work/models/qwen3-32b-dflash-en-zh
```

脚本默认开启 ASD，使用 V2、K15、TP2、BF16、前缀缓存和 `FULL_DECODE_ONLY`；默认 B=8、g=.25、M=2。逻辑设备默认为 `0,1`，服务监听 `127.0.0.1:8815`。目标与草稿模型路径是两个必填参数。

通过环境变量调整参数，例如 K8：

```bash
ASD_K=8 \
VLLM_ASCEND_ASD_RISK_BUDGET=8 \
VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE=0.25 \
VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK=2 \
bash examples/spec_decode/run_dflash_asd.sh /path/to/target /path/to/draft
```

只打印配置和完整命令，不加载模型：

```bash
bash examples/spec_decode/run_dflash_asd.sh --dry-run /path/to/target /path/to/draft
bash examples/spec_decode/run_dflash_asd.sh --help
```

常用附加变量：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `ASD_PYTHON` | `python` | 当前已配置环境的 Python 可执行文件 |
| `ASD_K` | `15` | 草稿候选数量 |
| `ASD_TP` | `2` | 目标模型 TP |
| `ASD_DRAFT_TP` | 与 `ASD_TP` 相同 | 草稿模型 TP |
| `ASD_PORT` | `8815` | HTTP 端口 |
| `ASD_MAX_NUM_SEQS` | `16` | 服务调度容量 |
| `ASD_MAX_BATCHED_TOKENS` | `528` | 批次 token 数上限 |

完整变量列表见脚本 `--help`。脚本在前台运行，日志输出到终端。

本机也可以使用已经配置好的项目环境包装器：

```bash
bash /home/ma-user/work/src/v0202/vllm-ascend-official-asd/run_env.sh env \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  bash /home/ma-user/work/src/v0202/vllm-ascend-official-asd/vllm-ascend/examples/spec_decode/run_dflash_asd.sh \
  /home/ma-user/work/models/Qwen3-32B \
  /home/ma-user/work/models/qwen3-32b-dflash-en-zh
```

包装器会清理继承的环境，所以自定义 `ASD_K` 等参数应放在 `run_env.sh env` 后面。`VLLM_USE_V2_MODEL_RUNNER` 已存在时，脚本会保留其值；本机包装器调用中显式设置为 1。

当前 DFlash V2 注意力路径支持 K≤15。K15 表示 15 个草稿候选，对应 16 个验证位置。`max_num_batched_tokens` 至少覆盖 `max_num_seqs × (K+1)`。

`FULL_DECODE_ONLY` 用于目标解码和 DFlash 草稿查询的图执行，prefill 可使用普通路径。在 Ascend 上实际执行 ACL/NPUGraph，配置键沿用 `cudagraph_mode`。

DFlash 配置的 `target_layer_ids` 使用从零开始的 decoder 层号，读取隐藏状态时转换为加 1 后的槽位；显式 `eagle_aux_hidden_state_layer_ids` 已是槽位，不再转换。

### HTTP 请求

```bash
curl --fail http://127.0.0.1:8815/health

curl --fail http://127.0.0.1:8815/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-32b-dflash-asd",
    "messages": [{"role": "user", "content": "计算 17 × 23，给出简短过程。"}],
    "temperature": 0,
    "top_p": 1,
    "seed": 0,
    "max_tokens": 1024
  }'
```

对于已应用聊天模板的输入，可以使用 `/v1/completions`，传入 `prompt` 并设置 `add_special_tokens=false`，避免重复添加特殊 token。客户端不需要传入 ASD 参数。

`--max-num-seqs 16` 控制服务调度容量；并发请求数量由客户端控制。

### 开关与参数调整

- 关闭近似接受：启动时设置 `VLLM_ASCEND_ENABLE_ASD=0`。
- 切换 V1：设置 `VLLM_USE_V2_MODEL_RUNNER=0`。
- 修改草稿长度：调整 `num_speculative_tokens`。
- 修改 B/g/M：调整对应环境变量后重启服务。
- 检查严格退化：保持 ASD 开启，将 B、g、M 中任意一项设置为零。

关闭 ASD 与零预算经过不同的实现路径：前者调用框架严格验证器，后者使用 ASD 内核执行严格匹配。

## 5. 生效检查和指标

V2 初始化日志包含：

```text
ASD enabled (Qwen reference): B=8.0, g=0.25, M=2; Triton FP32 verification
```

V1 初始化日志包含：

```text
Triton ASD enabled (official Qwen rule): B=8.0, g=0.25, M=2
```

V2 的采样器类为 `vllm_ascend.worker.v2.spec_decode.asd.ASDRejectionSampler`，V1 为 `vllm_ascend.sample.asd_rejection_sampler.ASDRejectionSampler`。

服务 `/metrics` 提供推测解码计数。取正式请求前后的差值：

```text
草稿接受率 = Δspec_decode_num_accepted_tokens_total / Δspec_decode_num_draft_tokens_total
纯草稿接受长度 = Δspec_decode_num_accepted_tokens_total / Δspec_decode_num_drafts_total
```

`drafts` 表示起草轮次数。接受计数不包含 recovery、bonus 和 padding。部分已有统计将纯草稿接受长度加 1，比较结果时应明确口径。

## 6. 验证代码

| 测试文件 | 覆盖范围 |
|---|---|
| `tests/ut/ops/test_asd_triton.py` | 作者参考规则、regret、平局、V1 接口、预算、前缀、恢复位置及终止边界。 |
| `tests/ut/ops/test_asd_v2.py` | V2 sampler、批次/位置对齐、DFlash 层号、输入准备和图相关检查。 |
| `tests/ut/ops/test_asd_score_optimized.py` | 优化评分一致性、数据类型、非连续输入、NaN/Inf、动态行数与图重放。 |
| `tests/ut/ops/test_asd_decoupling.py` | 单链接口约束、草稿与 ASD 的依赖边界、共享评分函数保持一致。 |

参考实现和评分历史版本位于 `tests/ut/ops/fixtures/`，附有来源、提交号、内容哈希和许可。

在配置好的 NPU 环境中，从仓库根目录执行：

```bash
python -m pytest --confcutdir=tests/ut/ops \
  tests/ut/ops/test_asd_triton.py \
  tests/ut/ops/test_asd_v2.py \
  tests/ut/ops/test_asd_score_optimized.py \
  tests/ut/ops/test_asd_decoupling.py -q
```
