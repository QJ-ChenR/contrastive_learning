# DNA encoder 连续相似度训练

本轮实现以已有 ModernBERT DNA encoder 为起点，使用化学相似度的**差值回归（Margin-MSE）**训练 BGC 向量。输入保持完整 DNA，推理时每个 BGC 独立得到一个向量。运行入口为 [`scripts/train_dna_encoder.py`](../scripts/train_dna_encoder.py)，首轮配置为 [`configs/dna_margin_mse.json`](../configs/dna_margin_mse.json)。

训练、验证、恢复和显式测试评估脚本已经提供。服务器原始 checkpoint 已通过 LoRA/BF16/FA2 的完整 DNA 资源探测，最长训练样本为 231,984 bp / 52,424 tokens；此前双重重算配置下峰值已分配显存为 3.66 GiB。2026-11-26 用户反馈 frozen/head 运行正常，但 LoRA 两小时后仍在第 1 轮 160/908 三元组。下方新增性能配置利用显存减少重算，**尚待服务器增强 probe 验证实际速度和峰值**；已通过的软件测试不构成检索效果结论。

## 1. 已固定的数据和监督

沿用 [`dna_homology_split_v1`](../data/processed/dna_homology_split_v1/summary.md)，不重新划分、不按配对行随机拆分。

| 集合 | BGC 数 | 同源组数 | 用途 |
| --- | ---: | ---: | --- |
| train | 908 | 855 | 唯一训练锚点及候选来源，也是检索图库 |
| validation | 114 | 107 | 每轮检索评估、早停及最优 checkpoint 选择 |
| test | 113 | 106 | 确定模型方案之后单独评估 |

监督分数仍是作者提供的 RDKit 指纹 Tanimoto，多产物 BGC 间取产物配对的最大相似度。设 \(S_{ij}\) 为化学分数，\(c_{ij}\) 为归一化 BGC 向量的余弦相似度，则每个三元组的目标是：

\[
L(i,j,k)=\left[(c_{ij}-c_{ik})-(S_{ij}-S_{ik})\right]^2.
\]

实现使用严格的 **MSE**，目标差值缩放固定为 1；没有改成 Huber，也没有将分数转换成 0/1 正负标签。分差为零时仍保留该关系。多产物最大分数和产物记录不完整可能带来监督偏差，训练损失下降本身不能证明真实功能或 GCF 结构改善。

每个 epoch 对全部 908 个训练 BGC 打乱顺序，每个 BGC 恰好作为一次锚点，共 908 个三元组。候选都来自训练集，且与锚点及另一候选不同：

- 一半锚点：从该锚点化学相似度最高的 **10 个邻居**中抽取一个，再从其余候选中均匀抽取另一个。
- 另一半锚点：按该锚点的候选相似度排序，划成四个数量尽量相等的区间，随机选择两个不同区间，各取一个候选。
- 候选的前后顺序随机交换，连续目标可为正值、负值或零。同分候选每轮先作由 seed/epoch 决定的随机排列，再按分数稳定排序，避免固定 BGC ID 获得近邻名额。

这里的 top-10 和四分位只决定抽样覆盖，不是化学正负阈值。908 个锚点中包括原来没有 `S > 0.65` 邻居的 475 个 BGC。当前队列大小为偶数，两类采样各占 454 个锚点；奇数队列时近邻策略多一个。小型测试队列的 top-10 自动缩为 `min(10, n−1)`。

数据加载检查划分清单、三个 ID 列表、输入文件表及同源组表的哈希，检查 BGC/组不跨集合及外部关联组隔离；矩阵必须匹配划分时记录的源文件哈希。读入每条 DNA 时检查原 GenBank 文件、单记录身份、完整大写序列及长度。当前作者矩阵是完整有限值矩阵；缺失或非有限训练分数会明确报错，不会填成零。

## 2. 三个实验模式

| `--mode` | 编码器及投影 | 训练内容 |
| --- | --- | --- |
| `frozen` | 原始 encoder，内容 token 均值后 L2 归一化；原维度，当前配置为 768 | 不更新参数，给出冻结基线 |
| `head` | 冻结 encoder，归一化原向量 → 无 bias 的线性投影至 256 维 → L2 归一化 | 仅训练投影头 |
| `lora` | encoder 加 LoRA，使用相同 256 维投影头 | 训练 LoRA 及投影头 |

内容池化保留 UNK，排除 CLS、SEP、PAD。原 tokenizer 对整个 DNA 一次 BPE 分词；offset 必须连续覆盖完整输入。保留原方向，不自动反向互补、切窗或裁剪。配置中 `max_position_embeddings` 不作为截断长度。

`head` 与 `lora` 在相同 seed 下使用相同投影初始化、三元组采样规则和候选曝光量。LoRA 保持原始模型参数冻结，默认 `r=8`、`alpha=16`、`dropout=0.05`，目标后缀为 `attn.Wqkv` 和 `attn.Wo`；加载时核对它们是否匹配真实 Linear 模块，没有匹配即报错。

| 参数 | 默认值 |
| --- | --- |
| seed | 42 |
| 精度／attention | BF16／FlashAttention 2 |
| 投影头学习率 | `1e-4` |
| LoRA 学习率 | `3e-5` |
| 优化器／weight decay | AdamW／`0.01` |
| 每次 optimizer update 的三元组数 | 8，末组按实际数量归一化 |
| 学习率计划 | 5% warmup，随后线性衰减 |
| 梯度范数上限 | 1.0 |
| 最大 epoch 数 | 20 |
| 早停 | 验证主指标连续 4 轮没有超过历史最佳值 `1e-4` |
| 最大 token 数 | `null`，完整输入，无额外长度过滤 |

这些参数在 JSON 配置中集中管理。首轮不自动搜索超参数，也不自动运行多种 seed。协议确认后可用新的 `--seed` 和独立 `--output-dir` 重复实验，比较结果稳定性。

## 3. 服务器准备

已确认硬件型号为 **L40S 48 GB**，驱动 `580.82.07`；本脚本按单进程、单 GPU 工作，没有 DDP 训练入口。若服务器有多张 GPU，可在启动前用 `CUDA_VISIBLE_DEVICES` 选择一张，恢复运行时保持相同可见 GPU 配置。

将项目代码及以下已有数据目录复制到服务器，保持项目内相对路径：

```text
contrastive_learning/
├── configs/dna_margin_mse.json
├── dna_training/
├── scripts/
├── requirements.txt
└── data/
    ├── raw/BGC-clustering-benchmark/
    │   ├── benchmark_bgc/
    │   └── tanimoto_results/NPAtlas_bm_v1.tsv
    └── processed/dna_homology_split_v1/
```

建议复制原始 benchmark 目录和完整划分快照，不重新下载一个可能变化的版本。`data/` 被 Git 忽略，只有克隆代码不会得到这些输入。训练不需要在服务器重跑 BLAST 同源搜索，也不依赖本机旧绝对路径下的比对临时文件。以下命令均在服务器项目根目录运行；从其他目录启动时也可使用脚本绝对路径，并通过 `--project-root` 指定项目根。

服务器实际环境是 **Python 3.11.11、PyTorch 2.6.0.dev20241112／CUDA 12.4、FlashAttention 2.8.3.post2**。保留这组 GPU 二进制，使用统一的 [`requirements.txt`](../requirements.txt) 补齐依赖。完整版本表、环境备份／克隆方法、安装计划核对、导入检查及 BF16 内核验证见 **[服务器环境指南](server-environment.md)**。

在已有环境或其 Conda 克隆中执行：

```bash
# 若使用克隆环境，改为 conda activate dna_training。
conda activate natural_product
python -m pip install --dry-run -r requirements.txt
python -m pip install -r requirements.txt
python -m pip check
```

缺失的 LoRA 库固定为 `peft==0.18.1`；现有 Transformers 5.1.0、Tokenizers 0.22.2、Accelerate 1.5.2、NumPy 2.2.4 和 Biopython 1.85 保留。此文件面向已有服务器环境，不是从空环境重建旧 nightly 的安装器；若 pip 准备替换／编译 PyTorch 或 FlashAttention，先核对解释器。不要使用旧指南中的 PyTorch 2.10 / Python 3.12 本地测试组合。包导入和小型 GPU 内核检查通过后再进行完整 DNA 探测。

正式运行前，修改配置中的三个模型路径：

| 配置项 | 含义 |
| --- | --- |
| `checkpoint` | 包含真实预训练 MLM 权重的本地 checkpoint 目录 |
| `tokenizer_path` | 默认为 `null`，读取权重目录内的原始 `tokenizer.json` 和 `tokenizer_config.json`；分词器另存时才指定其目录 |
| `config_path` | 默认为 `null`，读取 `checkpoint/config.json`；配置另存时才指定原始配置文件或所在目录 |

示例配置的权重目录是 `/home/s252613/BGC_encoder/BGC_annotation/BGC_transformer/checkpoint-13920`，模型配置和 tokenizer 默认都从该权重目录读取。服务器文件清单已确认原始 `tokenizer.json`（286487 字节）和 `tokenizer_config.json`（270 字节）在该目录内。此前把 `config_path`、`tokenizer_path` 指向父目录，分别造成配置识别和 tokenizer backend 初始化失败，均未进入 DNA 前向／反向；示例已移除这两个父目录假设。其他 checkpoint 若将配置或分词器另存，可分别用 `--config-path` 和 `--tokenizer-path` 指定原始文件位置。

若报 `Original model config not found` 或缺少 `model_type='modernbert'`，先检查报错指向的原配置位置，不要给一个错误文件补字段来猜模型结构。加载器会在加载模型张量之前终止；这种配置错误不是 CUDA 显存溢出。重新探测应使用新的 `--report-dir`，保留前一次失败记录。

若报 `Original fast DNA tokenizer not found`，检查 `tokenizer_path` 是否选中了包含原始 `tokenizer.json` 的目录。Transformers 的通用 backend 初始化错误可能建议安装 `sentencepiece` 或 `tiktoken`，但这不能补回缺失的原始 DNA BPE 词表及合并规则。不要重新训练或替换为通用 ModernBERT 分词器。加载器会保留原始文件，只在内存中加载并核对词表与特殊 token。

模型仅从本地加载，不会因缺权重而自动下载或随机初始化。完整 MLM 权重加载不匹配、缺分片、词表/PAD 不一致或 LoRA 目标不存在都会终止。CLS/SEP 等过时的配置元数据会根据已核验的原 tokenizer 修正，并记录修正内容；不改变词表或扩展 embedding。

## 4. 推荐运行顺序

### 4.1 先做数据审计

```bash
python scripts/train_dna_encoder.py audit \
  --report runs/dna_training_audit.json
```

`audit` 不加载模型、不要求 checkpoint 或 GPU；单独安装 `numpy==2.2.4 biopython==1.85` 也可执行，版本与统一 [`requirements.txt`](../requirements.txt) 对齐。报告包括划分计数、训练 DNA 验证、完整 DNA 长度统计、采样协议及第 0 轮分差摘要。它只核验训练 DNA 的实际文件内容；验证/测试 DNA 会在各自需要编码时核验。审计通过不能替代模型资源验证。

### 4.2 在真实 GPU 上做 LoRA 资源探测

核实并修改示例配置中的模型路径后执行：

```bash
python scripts/train_dna_encoder.py probe \
  --config configs/dna_margin_mse.json \
  --mode lora \
  --report-dir runs/dna_probe_lora_s42
```

该步骤先完整分词全部训练 BGC，再按**实际输入 token 长度**选中位数、P90 和最长样本。各样本与另外两个训练 BGC 组成探测三元组，并额外将**三条最长 DNA 一起**组成 `longest_triplet` 压力样本，执行前向、反向、optimizer update，最后验证保存重载。自适应重算配置还探测预算内总 tokens 最大的第 0 轮实际三元组 `activation_budget_boundary`。报告记录各成员 tokens、合计长度、实际重算开关、用时、CUDA allocated/reserved 峰值及保存重载误差；探测更新后的权重不进入正式实验。

必须检查 `probe.json` 的 `status`，尤其是最长样本是否成功。当前默认 `gradient_replay=true`：每个三元组先无梯度编码三个 BGC，计算表示层梯度，再恢复各次 RNG 状态，逐个 BGC 重算和反传。每条 DNA 经过两遍编码，LoRA 模式同时启用层级 gradient checkpointing，因此会增加计算量；反传时仅保留一个完整 BGC 的计算图。

这种重算不会将 DNA 切成窗口，也不会把历史 LoRA 向量作为持久训练缓存。它无法解决“单条完整 DNA 本身也装不下”的情况。若 OOM、输入超过显式 `max_tokens` 或任意样本失败，程序写出失败记录并终止，不静默裁剪、切窗、跳过或剔除长序列。需要调整资源协议时，应另建运行目录并记录变化。

### 4.2.1 L40S 性能配置：保留激活，减少重算

新配置为 [`configs/dna_margin_mse_lora_fast.json`](../configs/dna_margin_mse_lora_fast.json)。原配置继续保留；两个配置的损失、数据划分、三元组顺序、学习率、梯度累积数、LoRA 目标和 dropout 均相同，只调整计算与存储策略。

| 项目 | 原 LoRA 配置 | 新性能配置 |
| --- | --- | --- |
| `gradient_replay` | `true`，三元组调用 encoder 6 次 | `false`，调用 3 次，保留三个计算图至反传结束 |
| `gradient_checkpointing` | `true`，所有三元组都重算层 | `true`，由下面的预算决定何时重算 |
| `checkpoint_token_budget` | `0`，保持静态重算设置 | `30000`，三条输入合计 ≤30,000 tokens 时保留层激活，超过时重算 |
| 固定输入 | 每次创建 token/mask 张量 | 一次验证并缓存 GPU 张量，全程复用 |
| 最佳模型验证导出 | 训练结束重新编码 train/validation | 复用所选 checkpoint 当轮的验证向量 |

tokens 预算包括每个 BGC 的 CLS/SEP，只控制是否保存反传需要的中间激活，**不是截断长度**。第 0 轮固定采样中，624/908 个三元组落在 30,000 预算内，合计约占该轮输入 token 工作量的 50%；这不是 GPU 提速比例。30,000 是待实测的资源预算，不能从旧配置的 3.66 GiB 推断完全关闭重算也一定适合 48 GB。层级 checkpoint 本身就是以重算换显存，详情见 [PyTorch 2.6 文档](https://docs.pytorch.org/docs/2.6/checkpoint.html)。

另一个已修正的开销来自梯度检查：原先每个三元组和更新前分别对约 89 个可训练张量进行 Python 布尔判断，逐个等待 GPU。现在在设备端合并有限性结果，正常路径每个设备只同步一次；发现非有限值仍在 optimizer 更新前终止，并报告具体参数名。[PyTorch 同步开销说明](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization)。

可以保存的内容及边界：

- **输入 token/mask** 与模型参数无关，908 条训练 DNA 共约 122 MiB；加入 validation 后稍大。每个 BGC 只创建一次设备张量，后续不再逐 token 检查或重复 CPU→GPU 搬运。重载模型时仍核对 tokenizer、形状、device 和未修改状态。
- **本三元组的计算图／激活** 可保留到本次反传结束，使三个分支都贡献梯度。预算内直接保存，预算外启用层重算；仍逐三元组反传，不会同时保留 8 个三元组的所有图。
- **最佳 checkpoint 的验证向量** 约 1 MiB，和它的 epoch、权重、数据及协议身份一起存入 `last.pt`/`best.pt`。最终只复用同一最佳模型的结果；显式 `evaluate` 仍独立重算，每轮 LoRA 验证也仍用当时的参数编码全图库。
- **跨 optimizer update 的 LoRA 训练向量不能复用**，参数更新会让旧向量失效。实际检查发现，每 8 个三元组内部按 BGC ID 去重只节省约 0.9% 编码，且共享一次 dropout 会改变当前训练语义，因此没有采用这种缓存。

在服务器更新代码并激活 `dna_training` 后，先执行：

```bash
python scripts/train_dna_encoder.py probe \
  --config configs/dna_margin_mse_lora_fast.json \
  --report-dir runs/dna_probe_lora_fast_s42
```

检查整体 `status=passed`，预算边界案例的 `active_gradient_checkpointing=false`、三最长案例为 `true`，以及两者的峰值和耗时。三最长输入合计为 **138,497 tokens**；这比旧 probe 的“一个最长＋两个最短”更接近最坏的训练组合。每项失败仍会终止，报告中的 `active_probe` 标出出错组合；不会在 OOM 后静默改变方案。

如预算边界 OOM，可用较小预算重新 probe，例如 `--checkpoint-token-budget 20000`，并在正式训练使用同一设置。设为 `0` 则所有三元组保留层级重算，但仍关闭第一层 replay。`--no-gradient-checkpointing` 是另一个需单独 probe 的可选实验，不建议仅根据旧显存结果直接用于完整数据。

新 probe 通过后，先运行一个完整 epoch，保持原来的 20 epoch 学习率计划：

```bash
python scripts/train_dna_encoder.py train \
  --config configs/dna_margin_mse_lora_fast.json \
  --output-dir runs/dna_lora_fast_s42 \
  --stop-after-epochs 1
```

日志现在分开显示初始验证、训练更新及每轮验证；第一个更新、每 10 个更新和最后一个更新打印实际耗时。`history.json` 新增 `training_seconds`、`validation_seconds`、`mean_update_seconds`、`checkpointed_triplets` 和 reserved 显存峰值。用这些实测项评估提速，不把包含初始验证的总运行时直接除以三元组数。

**旧运行与切换：** 保留原 `runs/dna_lora_s42`，性能配置使用新的输出目录。旧程序只在完整 epoch 边界保存，当前第 1 轮 160/908 的中途进度不能精确续跑。更换本轮代码／资源配置后，旧运行的 `--resume` 会因身份检查拒绝，不能手改保存的配置绕过。已完成的 frozen/head 结果保留，不要求重跑。新运行完成首轮后，用下方正常 `--resume` 继续；探测和训练都不覆盖师兄的原 checkpoint。

本地 Python 3.11 / PyTorch 2.6 CPU 环境下 **116 项测试通过**，涵盖有 dropout 时直接／重算路径的梯度与 RNG 一致性、输入缓存更新边界、自适应重算、最佳验证缓存身份、断点恢复及 probe 失败记录。L40S 的实际速度与显存需要上面的新报告确认。

### 4.3 运行三个对照

```bash
python scripts/train_dna_encoder.py train \
  --config configs/dna_margin_mse.json \
  --mode frozen \
  --output-dir runs/dna_frozen_s42

python scripts/train_dna_encoder.py train \
  --config configs/dna_margin_mse.json \
  --mode head \
  --output-dir runs/dna_head_s42

python scripts/train_dna_encoder.py train \
  --config configs/dna_margin_mse_lora_fast.json \
  --mode lora \
  --output-dir runs/dna_lora_fast_s42
```

以上命令按顺序独立运行；已有 frozen/head 结果可保留，LoRA 性能配置须先通过 4.2.1 的新 probe。若已用 `--stop-after-epochs 1` 启动该 LoRA 目录，下一步使用 `--resume`，不要重新执行首次启动命令。`frozen` 命令只导出和评估冻结表示，不进行 optimizer update。`head` 和 `lora` 在第 0 轮评估之后训练，每轮使用验证集选择最佳状态；如果训练始终没有改善，最佳 checkpoint 可以保留在第 0 轮。

`frozen` 和 `head` 可以缓存带来源哈希的归一化 encoder 原向量；`head` 每次仍使用当前投影头计算表示。`lora` 禁止使用这种脱离梯度的冻结缓存，验证时重新编码当前模型下的训练图库和验证查询。

新实验目录必须为空，程序不会覆盖已有实验。每个运行目录使用进程锁，防止两个进程同时写入。

### 4.4 中断与恢复

```bash
python scripts/train_dna_encoder.py train \
  --output-dir runs/dna_lora_fast_s42 \
  --resume
```

不再传 `--config` 时，恢复命令从该目录保存的 `config.json` 读取原协议。`last.pt` 保存最近完成的 epoch、模型可训练状态、optimizer、学习率进度、随机状态、早停状态和历史。若在 epoch 中间失败或中断，恢复时从最近完整 epoch 的下一轮重新开始，不声称恢复到中断的某一个三元组。

恢复会比对设置、数据/原始权重/tokenizer 身份、源代码哈希及依赖版本；更换数据、修改训练代码、精度、attention 后端或训练协议不会被当作同一实验静默继续。项目和 checkpoint 可搬迁到新路径，但内容身份必须相同，且恢复时应提供新的 `--project-root`、`--checkpoint`、`--tokenizer-path`、`--config-path`。CUDA RNG 恢复还要求相同可见设备数量。CUDA 内核的逐位确定性仍受后端影响，不能用 CPU 重复性测试替代这一保证。

如需有计划地先运行少量完整 epoch，可在首次命令添加 `--stop-after-epochs 1`；它保留配置中完整 `max_epochs` 对应的学习率计划，之后使用 `--resume`。不要通过修改 `max_epochs` 来模拟同一计划的暂停。

### 4.5 确定模型方案后显式评估 test

训练过程只做验证评估。选择 head 或 LoRA、固定参数和重复实验方案之后，才对所选运行执行：

```bash
python scripts/train_dna_encoder.py evaluate \
  --run-dir runs/dna_lora_fast_s42 \
  --split test
```

这里的 LoRA 目录只是命令示例，实际应使用最终选定的运行。独立重算验证结果可将 `--split` 改为 `validation`。评估始终加载 `best.pt`，不按 test 表现再次选择 epoch 或调整参数；完整检索向量和分数写入 `evaluation_test/` 或 `evaluation_validation/`。

## 5. 指标与选择规则

默认验证为 **114 个 validation 查询 → 908 个 train 图库**，最终测试为 **113 个 test 查询 → 同一个 train 图库**。检索分数为向量余弦，预测同分按图库 BGC ID 排序，与化学标签无关。

- **主指标：nDCG@10，连续 gain = S**。按查询计算后宏平均，用于最佳 checkpoint 与早停。没有有效正增益的查询记为未定义，并单独报告数量；不会把缺失关系当成零相关。
- **Spearman**：每个查询对全部有效非自身图库关系计算，采用平均秩处理同分，再对有定义的查询宏平均。常量分数或有效候选不足时记为未定义。
- **Recall@10，辅助诊断**：仅对图库中至少有一个 `S > 0.65` 邻居的查询计算，同时报告适用/不适用查询数量。现有覆盖为 validation 的 50/114、test 的 47/113；0.65 不进入训练损失。

指标 JSON 中主字段为 `aggregate.ndcg_at_k`，同时记录 `k=10`、gain 定义、查询/图库计数以及各项有效分母。不要将只覆盖部分查询的 Recall 与所有有效查询上的连续 nDCG 混为同一评价范围。

本轮已执行 BGC/同源组隔离，但训练脚本尚未实现按同源组、GC、物种、长度的分层评估，也未实现 GCF 聚类评估、化学骨架隔离或远缘蛋白同源审查。当前最优 checkpoint 的含义限于该固定划分上的化学检索主指标，不能直接解释为完整功能表示或 GCF 泛化已经改善。

## 6. 输出文件和复核

| 路径 | 内容 |
| --- | --- |
| `config.json` | 此次运行实际设置，供恢复使用 |
| `run.json` | 数据/模型/源码/依赖身份，采样、损失、验证选择规则和参数数目 |
| `token_lengths.tsv` | 本次训练及验证输入的完整 token 数、DNA 哈希、UNK/模糊碱基统计；不含测试输入 |
| `history.json` | 第 0 轮和各训练轮的验证指标、训练误差、optimizer 步数、分差覆盖；分开的训练／验证耗时、更新耗时、实际重算三元组数及显存 |
| `best.pt` | 验证选择的最佳模型状态，用于评估；恢复训练使用 `last.pt` |
| `last.pt` | 最近完整 epoch 的恢复状态，与 `best.pt` 的用途不同 |
| `summary.json` | 是否完成/停在 epoch 边界、最佳轮次和指标、已运行轮次及步数；`test_evaluated=false` |
| `failure.json` | 训练失败/中断时的异常及 BGC 上下文；成功完成后移除旧失败标记 |
| `base_features.npz`、`base_features.json` | 仅 frozen/head 的原向量缓存及身份/内容哈希；NPZ 字段为 `ids`、`embeddings` |
| `validation/metrics.json`、`validation/per_query.tsv` | 最佳模型的验证总体及逐查询结果 |
| `validation/embeddings.npz` | 与明确 ID 顺序对应的查询和图库向量 |
| `evaluation_test/` 或 `evaluation_validation/` | 显式评估产生的同类结果，另含 `provenance.json` |
| 探测目录 `probe.json`、`train_token_lengths.tsv` | 资源探测状态及完整训练 token 长度表 |

检索向量 NPZ 使用四个字段：`query_ids`、`gallery_ids`、`query_embeddings`、`gallery_embeddings`；每一行与对应 ID 数组对齐。可使用 `numpy.load(path, allow_pickle=False)` 读取。逐查询表中的 `top_ids` 是 JSON 列表。

`best.pt`/`last.pt` 保存投影头和 LoRA adapter 等可训练状态，并引用原 checkpoint 的身份，不复制完整冻结 backbone。`last.pt` 还保留最佳可训练状态和该模型对应的 validation 向量，可在两次文件写入之间中断后恢复 `best.pt`，并避免结束时重复编码。验证向量不是训练缓存，不会用于后续 LoRA 更新。重新部署或评估仍需保留原预训练 checkpoint、原 tokenizer/config 和本项目代码；不要只复制一个 `.pt` 文件就删除原权重。

本地自测可运行：

```bash
python -m unittest discover -s tests -v
```

测试包含数据身份和泄漏边界、采样复现、手算检索指标、短合成 ModernBERT 的梯度/LoRA/保存重载及训练恢复行为。真实 checkpoint 和最长完整 DNA 的 CUDA 能力以服务器 `probe` 的结果为准。

## 7. 方法来源

- 连续分差监督参考 [Hofstätter et al., Margin-MSE](https://arxiv.org/abs/2010.02666)。本项目把检索教师分数替换为固定化学相似度，相关效果尚需 BGC 实验验证。
- 参数适配使用 [Hu et al., LoRA](https://arxiv.org/abs/2106.09685) 的低秩适配思路，通过 PEFT 实现。
- 表示梯度与逐输入重算的分离参考 [Gao et al., GradCache](https://aclanthology.org/2021.repl4nlp-1.31/)。本项目实现的是三元组 Margin-MSE 的逐序列重算，不依赖 GradCache 包，也没有构造大批量 InfoNCE 分母或跨步陈旧向量队列。
