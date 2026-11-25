# contrastive_learning

基于产物结构关系的 BGC 监督式对比学习项目。

创建日期：2026-11-20（Europe/Paris）。

**当前状态：已完成 DNA 近同源分组、按 BGC／组划分及连续差值监督训练脚本；尚未在真实服务器 checkpoint 上训练。** 最新入口：[训练与服务器运行指南](docs/dna-training.md)、[划分结果](data/processed/dna_homology_split_v1/summary.md)、[分组规则与复现](docs/homology-split.md)。原始数据、[初始清单快照](data/processed/benchmark_manifest_v1/summary.md)及全部产物关系保留。

服务器依赖已统一为 [requirements.txt](requirements.txt)，对应已确认的 Python 3.11 / PyTorch 2.6 nightly / CUDA 12.4 / L40S 环境，补入 PEFT 0.18.1。安装、版本依据与 GPU 检查见[环境指南](docs/server-environment.md)。

2026-11-25 完成的固定划分使用 `seed=42`、目标比例 `80/10/10`。1343 个有 DNA 的候选与 421 个 B/E/G/H 记录联合建图；在明确的 DNA 一致性／覆盖规则下隔离 208 个候选（比此前精确重合审计增加 24 个），其余 1135 个 BGC、1068 个组进入划分：

| 集合 | BGC | 同源组 | 集合内 S > 0.65 正对 | 有集合内正邻居的 BGC |
| --- | ---: | ---: | ---: | ---: |
| train | 908 | 855 | 888 | 433 |
| validation | 114 | 107 | 25 | 32 |
| test | 113 | 106 | 21 | 28 |

分组标准为通过一致性 ≥80%、长度 ≥100、E-value ≤1e-10 的局部比对，在较短 DNA 上的去重覆盖 ≥80%，另保留正反向完整相等／包含关系；连通组不可跨集合。全量搜索使用 BLASTN 的 `megablast` 任务、16 碱基种子，补充此前中止搜索中已有的有效证据。阈值是本次操作性隔离定义，不代表全部远缘蛋白同源或化学骨架已隔离；详细来源与限制见上述协议。化学分数仅在划分固定后用于诊断，未用于挑选划分。

## 1. 项目目标

学习能够独立表示单个生物合成基因簇（BGC）的向量，使产生相似天然产物的 BGC 在向量空间中更接近，用于 BGC 相似性检索，并通过聚类检验其家族组织能力。

当前第一步研究问题：用产物结构关系监督微调已有的 ModernBERT DNA encoder，能否比冻结 DNA 表示得到更适合产物检索的 BGC 向量？蛋白聚合器训练、与冻结 ESM2 的比较及独立 curated GCF 评估保留为研究方向。

需要始终区分三个目标：

- **产物结构关系**：本项目首轮候选监督的依据。
- **生物合成机制与模块组成**：解释、核验和可能的辅助监督依据。
- **GCF 归属**：独立的家族评估目标，不与产物结构相似性自动等同。

共享 NRPS、PKS 等粗类别不能直接证明两个 BGC 应组成正例；相似产物也不一定来自相同机制或同一家族。

## 2. 与现有项目的关系

| 项目／资料 | 作用 |
| --- | --- |
| [BGC_retrieve](../BGC_retrieve/README.md) | 已有 B/E/G/H 数据准备、ESM2／BGC-MAC／DNA 特征、average-linkage 与 HDBSCAN 聚类比较 |
| [BGC_annotation](../BGC_annotation/README.md) | 现有蛋白质／DNA 编码器与特征导出实现，后续优先复用已经验证的输入策略 |
| [本地 MIBiG 4.0 GBK](../BGC/data/raw/mibig_gbk_4.0/) | 可供序列对应与重合审查的现有参考数据 |
| [MIBiG 与 B/E/G/H 重合审计](../BGC_retrieve/data/processed/mibig_training_audit_v1/summary.md) | 已有泄漏证据与监督信息审查；不是本项目已完成的训练划分 |
| [已有聚类比较](../BGC_retrieve/data/processed/hdbscan_comparison_v1/summary.md) | 冻结表示的性能背景和比较依据 |

BGC-MAC 在部分现有比较中的表现，构成研究可训练聚合器的动机；当前结果并未单独证明 attention 的因果贡献，也不能消除训练暴露和输入差异的影响。

本项目独立于上述目录。现有原始数据、准备快照和实验结果应保留；未来产生的新数据和结果在本项目内追溯。实际模型训练与大规模特征提取计划在服务器上进行，路径和硬件配置届时明确。

## 3. 首选数据来源

首选审阅论文：

> Oyedele & Walker, *Benchmarking methods for measuring biosynthetic gene cluster similarity and determination of gene cluster families*. Bioinformatics, 2025, btaf636.

- [论文](https://academic.oup.com/bioinformatics/article/41/12/btaf636/8339750)
- [作者仓库](https://github.com/aswalker-lab/BGC-clustering-benchmark)
- [作者数据发布页](https://huggingface.co/datasets/allie-walker/BGC-clustering-benchmark)

该工作关联 MIBiG BGC 与 NPAtlas 产物结构，对部分错误关联进行了修正并补充遗漏产物。论文最终集合及公开化学分组文件包含 **1438 个 BGC ID**。这只是发布数据规模，不是去重、排除测试重合和划分后的训练样本数。

### 3.1 数据文件的含义

以下路径相对于已克隆的 `data/raw/BGC-clustering-benchmark/`。本次清单固定的来源提交为 `bb8500d60f90cb43397cc41de5aed396725aa800`；文件哈希见清单的 `input_files.tsv`。

| 路径 | 含义与计划用途 |
| --- | --- |
| `source_data/NPAtlas_bm_v1.tsv` | 修订后的产物结构和 BGC—产物关联源表 |
| `tanimoto_results/NPAtlas_bm_v1.tsv` | BGC × BGC 的产物相似度矩阵，首选监督来源；虽然扩展名为 `.tsv`，作者脚本按逗号分隔文件读取 |
| `benchmark_bgc/` | 当前有 1343 个可解析 GBK，矩阵中的另外 95 个 BGC 缺文件；尚未做模型 tokenization／训练资源验证 |
| `product_clusters/butina_clusters_pt2.csv`、`pt3.csv` | 化学距离的 Butina 分组，可作为可选对照，不直接视为人工 GCF 真值 |
| `bgc_similarities/` | BiG-SCAPE 等被比较方法的 BGC 相似度输出，用于基线比较，不作为本项目的化学监督 |

### 3.2 产物相似度定义

作者代码使用 `GetRDKitFPGenerator(fpSize=8192)` 生成 RDKit 指纹，再计算 Tanimoto 相似度。对两个多产物 BGC，取所有产物配对中的最大值：

\[
S_{ij}=\max_{a\in P_i,\,b\in P_j}
\operatorname{Tanimoto}(\operatorname{FP}(a),\operatorname{FP}(b))
\]

其中 `P_i` 为 BGC `i` 已记录的产物集合。高分表示至少一对已知产物相似，不代表整个产物集合相同；低分也可能受到产物记录不完整的影响。

监督分值来自已知产物结构的化学计算，不来自 BiG-SCAPE 的预测家族。该分值也不是实验测得的功能相似度。

## 4. 已确定首轮协议：连续相似度差值回归

每个训练单元为三个不同的训练 BGC：锚点 `a` 和候选 `b, c`。两个候选分别与锚点计算化学分数和向量余弦，优化固定尺度为 1 的 **MarginMSE**：

\[
L=\left[(\cos(z_a,z_b)-\cos(z_a,z_c))-(S_{ab}-S_{ac})\right]^2
\]

该目标同时保留顺序和分差，不要求绝对余弦值等于化学分数。使用原始连续矩阵；既有 `positive_pairs.tsv` 和 **0.65** 阈值仍用于覆盖诊断及辅助 Recall，**不参与主损失、训练配对资格或模型选择**。多产物最大相似度和不完整产物记录仍是标签解释的限制。

每轮随机排列全部 **908 个训练锚点**，每个恰好出现一次，包括 475 个没有阈值正邻居的 BGC。约一半训练单元从锚点的前 10 个化学近邻取一个候选，再取另一个随机候选；另一半从不同相似度排名四分位取候选。同分候选按 seed／epoch 确定性随机排序，候选方向随机化；这些区间只控制采样，标签保持连续，相同分数关系也保留。

采样只访问训练集内部关系；独立生物样本规模仍是 908 个 BGC／855 个同源组。当前源矩阵已通过完整、有限值校验；若输入出现未知或非有限分数，训练加载会报错，不会填零或自动转成负例。

配对关系不要求传递，不把化学正例图的连通分量转换成互斥家族标签。

### 首轮三项比较

- **frozen**：原冻结 DNA 表示，内容均值后归一化；不训练。
- **head**：冻结 encoder，仅训练无偏置的线性投影头，默认 768→256。
- **lora**：LoRA 微调 encoder，加上完全相同的投影头和差值损失。

head／lora 使用相同投影初始化 seed 和三元组序列。以验证集连续检索指标判断更新 encoder 是否值得保留；二值阈值、软标签 KL、Huber 等不是本轮默认训练目标，也尚未实现为额外实验臂。

## 5. 模型与输入计划

用户已明确先做基于产物关系的 DNA encoder 监督式对比微调。当前建议的结构为：

```text
完整 BGC DNA → 原 DNA tokenizer → ModernBERT DNA encoder（微调）
            → 内容 token 池化／投影 → 归一化 BGC 向量
```

完整 DNA、内容 token 均值池化、线性投影、LoRA 和差值回归已实现于 `dna_training/`，入口为 `scripts/train_dna_encoder.py`。CLS／SEP 不参与池化，UNK 保留；池化向量和投影后向量分别归一化。代码严格加载本地 MLM checkpoint，不以随机权重替代缺失权重，不截断或自动切窗。此前的蛋白聚合方案保留为后续比较：

```text
BGC → 现有蛋白／域输入解析 → 冻结 ESM2 token 表示
    → 可训练 BGC 聚合器／投影 → 归一化 BGC 向量
```

两个 BGC 通过共享编码器分别产生独立向量，以余弦或距离参与对比学习。产物结构只用于构造训练监督；推理和检索时只需要 BGC 输入。

- 后续蛋白路线考虑训练聚合器，保留冻结 ESM2 简单聚合为对照。attention 的具体层数、池化方式、输出维度尚未决定。
- 现有蛋白流程以 `aSDomain` 切分；没有相应注释的 CDS 使用完整翻译。Pfam 坐标切分仍是此前暂缓的方向，不在本项目创建时改变。
- 保持已有长蛋白处理策略并记录其配置。复用缓存前检查输入序列、分段、池化、模型和源码的对应关系，不能只凭 BGC ID 对齐。
- 学习目标在 BGC 层面计算，不把 BGC 的产物关系直接复制为每条蛋白的功能标签。
- ESM2 LoRA／部分解冻可作为后续比较，当前未确定实施；固定 token 缓存不能支持 ESM2 反向传播。
- DNA–蛋白融合属于后续可选方向。DNA 输入保持全长策略，不重新引入历史 DNA 长度过滤；实际训练可运行长度取决于硬件和反向传播资源，资源失败需单独记录。

## 6. 数据审计与划分原则

原始下载内容存入 `data/raw/`，初始清单存入 `data/processed/benchmark_manifest_v1/`，不覆盖原始文件。初始快照保留 1438 个 BGC、2351 个关联产物和 2498 条 BGC—产物关系，原有 `not_assessed`／`unassigned` 状态保持不变。实际训练资格与划分见新的 [bgc_split_manifest.tsv](data/processed/dna_homology_split_v1/bgc_split_manifest.tsv)，其中保留 95 个缺 DNA 和 208 个整组隔离记录及原因。

数据审计与划分遵循以下原则；已完成的本轮 DNA 范围见最新协议，未实施的额外目标保持单独标记：

1. 固定作者数据版本／提交、下载来源和文件哈希；核对源表、矩阵、产物列表、GBK 的身份对应及覆盖。
2. 检查矩阵行列 ID、对称性、对角线、数值范围、缺失值和重复项；复核修订版产物关系。
3. 检查 DNA 正反互补重复、边界包含、蛋白重复及近同源；保留重复和冲突证据。
4. 隔离与所有预留评估集合的重合，再按 BGC／同源组划分训练、验证和测试。
5. **划分完成后，分别在各集合内部生成配对。禁止随机拆分配对行来代替 BGC 划分。** 同一 BGC 的所有产物、蛋白、域和派生视图随其所属组移动。
6. 如果评估未见化学骨架的迁移，需要另外定义化学分组，并考虑每个 BGC 的全部已知产物；该协议尚未确定。

历史 MIBiG 审计发现，本地参考中有 **301 个唯一条目**与 B/E/G/H 存在确证的编号或 DNA 整体／完整包含重合，另有 `BGC0002037` 待复核。这些对应旧输入，不是本项目的排除数量。本项目重新核验后，精确重合为 184 个候选；本轮 DNA 近同源联合图隔离共 208 个，包含 `BGC0002037` 所在组。蛋白远缘同源和化学骨架隔离未实施。

A/C/D/F/I 五个 curated 集合可作为后续监督候选，但尚未决定纳入本项目训练。它们存在来源集中、标签冲突、跨集重复和家族粒度差异，不能直接拼接成统一训练真值。

## 7. 训练前需要量化的监督规模

首先统计各候选阈值下：

- 正对、负对、未知关系的数量和比例。
- 每个 BGC 的正邻居数量，以及没有正例的 BGC 比例。
- 正例是否集中于少数家族、近重复序列或具有多个已知产物的 BGC。
- 阈值附近配对的数量，按生物合成类别、来源和序列相似度分层的分布。
- 去重、评估隔离和划分后，实际可用的 BGC、正例关系及生物覆盖。

初始清单的划分前覆盖为：完整 1438 个 BGC 在 `S > 0.65` 下有 2194 个正对；1343 个有 DNA 的子集有 1917 个正对、899236 个负对，563 个 BGC 没有正邻居。详见[初始统计](data/processed/benchmark_manifest_v1/pair_statistics.tsv)。这些数字不能作为训练规模。

实际划分后的统计见 [pair_statistics.tsv](data/processed/dna_homology_split_v1/pair_statistics.tsv)。训练集有 888 个正对、433 个正例锚点，另外 475 个无集合内正邻居的 BGC 仍保留。验证和测试 query 对训练 gallery 分别有 50/114、47/113 个 query 存在 `S > 0.65` 正例，见 [retrieval_coverage.tsv](data/processed/dna_homology_split_v1/retrieval_coverage.tsv)；这是覆盖统计，不是检索性能。

当前训练按第 4 节进行连续三元组采样，所有训练 BGC 均可作为锚点；不因没有 0.65 阈值正例而删除样本。阈值表作为历史诊断保留。

## 8. 评估计划

同时评估产物检索和家族组织能力：

| 评估目标 | 候选指标／方式 |
| --- | --- |
| 产物结构相似性检索 | 已实现：validation query→train gallery，nDCG@10，线性增益 `gain=S`，按查询平均；仅此主指标选模型 |
| 相似度排序 | 已实现逐查询 Spearman 及有效查询计数；辅助 Recall@10 只在有 `S>0.65` 邻居的查询上平均并报告分母 |
| GCF 聚类 | 后续独立评估：延续原始 cosine＋average-linkage／HDBSCAN，报告 V-measure、ARI、簇数、覆盖和噪声比例 |
| 泛化与偏差 | 后续独立分层：同源隔离、未见家族／来源、类别及注释完整性等情形 |

B/E/G/H 已经参与前期方法比较与设计决策，可以保留为固定比较集合，但不能称为从未接触的盲测集。希望获得严格外部泛化结论时，需要另外预留独立数据。

保留冻结 ESM2、已有 BGC-MAC 和适用的 BiG-SCAPE 结果作为参照，核实同一输入与有效样本范围。既有 average-linkage 按真实家族数选择切割属于标签辅助诊断；用于部署的参数应在开发数据上确定。HDBSCAN 噪声样本不能从评分中静默去掉。

评价模型时，不能只看训练损失、配对准确率或 UMAP 图形。产物关系监督的提升也不保证 GCF V-measure 同步提高。

## 9. 下一步工作顺序与当前边界

1. **已完成清单步骤**：固定来源与哈希，保留所有 BGC／产物关系、95 个 DNA 缺失、产物字段冲突及历史重合 ID 提醒。结构化学验证和缺失补齐未完成。
2. **已完成 DNA 近同源分组与划分**：固定序列规则下与 B/E/G/H 联合建图，整组隔离并生成训练／验证／测试名单；不声称完成所有远缘同源审查。
3. **已完成划分后覆盖统计**：分别记录各集合内部及 query→train gallery 的监督覆盖；0.65 作为辅助覆盖诊断保留，主训练已采用连续差值监督。
4. **已实现服务器训练流程**：固定连续差值 MSE、三种实验臂、训练集采样、验证选模、epoch 边界恢复和显式测试命令。服务器启动前运行 `audit` 和 `probe`，核实真实 checkpoint 及全长反向传播资源。
5. 运行冻结／仅投影头／LoRA 三项比较；固定方案后重复种子并进行测试集和独立聚类评估，再决定是否扩展损失或模型。

**当前已完成清单、同源划分和训练代码；本地验证使用合成的小型 ModernBERT checkpoint。** 排除记录仍保留，原始文件未删除。真实服务器 checkpoint 的全长资源验证、正式训练及生物学检索／聚类结果尚未完成；结构修复和蛋白远缘同源搜索也未实施。
