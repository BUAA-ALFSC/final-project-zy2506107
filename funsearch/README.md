# FunSearch MindSpore （Qwen-7B） 复现实验

> 课程大作业提交包：使用 FunSearch函数搜索框架复现 online bin packing（在线装箱问题）实验，并将大语言模型后端接入 MindSpore/MindFormers下的 Qwen-7B模型。

## 项目状态

- Framework：MindSpore 2.4.10
- Model backend：MindFormers Qwen-7B `/completions`（补全接口）
- Platform：ModelArts Ascend 910B
- Problem：online bin packing（在线装箱问题）

本项目目标是课程复现，不声称得到原论文级别的数学发现。实验重点是跑通 FunSearch的程序搜索链路，并验证 Qwen-7B能够通过 MindSpore/MindFormers 后端参与优先级函数生成，冰一次解决组合优化问题。

## 关键结果

| 指标                  | 数值                                          |
| ------------------------- | ------------------------------------------- |
| Dataset             | OR3 online bin packing benchmark（在线装箱基准）    |
| Saved samples       | 91                                          |
| Qwen raw samples    | 90                                          |
| Qwen HTTP 200 calls | 90                                          |
| Baseline score      | -500.0                                      |
| Best score          | -212.0                                      |
| Best function       | `results/qwen_real_formal/best_priority.py` |

score（分数）定义为负的平均使用箱数，因此越大越好。baseline（基线函数）为 `-500.0`，本次正式实验最佳结果为 `-212.0`，对应 best-fit（最佳适配）类启发式：

```python
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return -np.abs(bins - item)
```

## 目录结构

```text
.
├── README.md
├── MANIFEST.txt
├── FunSearch_MindSpore_Qwen7B_ModelArts_Experiment.ipynb
├── funsearch.pptx
├── src/
│   ├── bin_packing_utils.py
│   ├── funsearch_bin_packing_local_llm.py
│   ├── implementation_config.py
│   ├── implementation_evaluator.py
│   ├── implementation_funsearch.py
│   ├── implementation_sampler.py
│   ├── llm-server/llm_server_mindspore_qwen.py
│   ├── research/qwen/predict_qwen_7b_smoke.yaml
│   ├── run_qwen_funsearch_formal.sh
│   └── start_qwen_server_newenv.sh
└── results/qwen_real_formal/
    ├── summary.json
    ├── scores.csv
    ├── best_priority.py
    ├── qwen_health.json
    ├── qwen_raw_samples.jsonl
    ├── qwen_server_tail.log
    └── funsearch_qwen_real_formal/
```

## 快速查看方式

推荐先打开：

```text
FunSearch_MindSpore_Qwen7B_ModelArts_Experiment.ipynb
```

该 Notebook为完整复现说明，包含：

- 环境检查
- 结果文件检查
- Qwen-7B 后端证据
- 评估器复算
- 基线函数与 最佳优先级函数对比
- 分数曲线分析
- 正式实验运行命令

Notebook 默认不会启动长时间 Qwen-7B实验，避免误触发高成本算力；长任务命令以 Markdown和安全开关形式保留。

## 环境要求

正式复现实验建议使用 ModelArts（华为云 AI 开发平台）Ascend 环境：

- Python 3.10
- MindSpore 2.4.10
- CANN 8.0.0
- MindFormers compatible with Qwen-7B（通义千问 7B 模型）
- Ascend 910B 或同级 NPU（神经网络处理器）

Notebook 的结果复核部分只需要常规 Python 环境：

- `numpy`
- `matplotlib`，用于绘制分数曲线，缺失时不影响核心复算

## 模型权重

Qwen-7B（通义千问 7B 模型）权重约 29GB，不放入提交包。正式实验使用 OBS（对象存储服务）挂载路径：

```text
/model/mindformers_models/qwen_7b/
```

期望文件：

```text
qwen_7b_base.ckpt
qwen.tiktoken
```

配置文件位于：

```text
src/research/qwen/predict_qwen_7b_smoke.yaml
```

其中 `checkpoint_name_or_path` 和 `vocab_file` 指向上述 OBS 挂载路径。

## 正式实验运行方式

进入提交包或 ModelArts 工作目录后，先启动 Qwen-7B server（服务端）：

```bash
bash src/start_qwen_server_newenv.sh
```

确认 `/health`（健康检查接口）和 `/completions`（补全接口）可用后，运行 FunSearch（函数搜索框架）正式实验：

```bash
bash src/run_qwen_funsearch_formal.sh
```

正式实验结果会写入 `results/qwen_real_formal/` 或脚本指定的输出目录。由于 Qwen-7B（通义千问 7B 模型）启动和推理成本较高，建议优先用 Notebook 的离线复算部分核验提交结果。

## 方法概述

FunSearch将 LLM生成的程序片段放入评估器中执行，根据分数保留更优程序，再把高分程序作为 提示词的一部分继续迭代。我们复现时选择 online bin packing（在线装箱问题）：物品按顺序到达，算法必须立即选择一个箱子，目标是尽量减少最终使用的箱子数量。

在代码实现上：

- `src/llm-server/llm_server_mindspore_qwen.py` 提供 MindSpore/MindFormers Qwen-7B `/completions`（补全接口）。
- `src/funsearch_bin_packing_local_llm.py` 将 FunSearch sampler（采样器）连接到本地 Qwen-7B server（服务端）。
- `src/bin_packing_utils.py` 提供 OR3 benchmark（基准数据）。
- `results/qwen_real_formal/` 保存正式实验结果和可复核证据。

## 可复核证据

提交包保留了以下证据文件：

- `summary.json`：正式实验结构化摘要。
- `scores.csv`：每个 sample（样本）的 score（分数）。
- `qwen_raw_samples.jsonl`：Qwen-7B（通义千问 7B 模型）原始输出和裁剪后的函数体。
- `qwen_health.json`：Qwen-7B server（服务端）健康检查结果。
- `qwen_server_tail.log`：服务端日志尾部。
- `best_priority.py`：本次实验得到的最佳 priority（优先级函数）。

## 局限性

- 本次实验得到的是 best-fit最佳适配类启发式，属于课程复现结果，不等同于原论文在 cap set（无三项算术级数集合问题）中发现的新构造。
- Qwen-7B（通义千问 7B 模型）的 instruct-style output（指令式输出）经常包含自然语言说明，因此代码中加入了裁剪逻辑。
- `-212.0` 是当前设置下的最佳分数平台值；追加采样没有突破该平台，说明模型发现能力受 prompt、采样策略、模型规模和 benchmark共同限制。

## 引用

```text
Romera-Paredes B, Barekatain M, Novikov A, et al. Mathematical discoveries from program search with large language models[J]. Nature, 2024, 625: 468-475.
```
