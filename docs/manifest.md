# 候选数据清单

清单对应用户于 2026-11-23 授权的第一步数据准备。它记录 BGC、DNA 文件和全部已知产物关系，保留缺失与冲突；不执行去重、剔除、训练划分、数据补齐或模型训练。

当前快照：[benchmark_manifest_v1/summary.md](../data/processed/benchmark_manifest_v1/summary.md)。原始来源：`data/raw/BGC-clustering-benchmark/`。既有 `.gitignore` 的 `data/*` 规则继续适用，生成数据不自动纳入 Git。

## 生成与验证

依赖为 NumPy 和 Biopython，无需 PyTorch、模型权重、RDKit 或联网。项目依赖现已合并到 [`requirements.txt`](../requirements.txt)；仅做清单处理可单独安装 `numpy==2.2.4 biopython==1.85`。历史快照的实际生成版本仍记录在其 `provenance.json`，不能用当前服务器依赖覆盖这份来源记录。

在项目根目录运行；当前本地可用解释器为 `/home/qiju/miniforge3/envs/BGC/bin/python`：

```bash
python -B scripts/build_manifest.py \
  --source data/raw/BGC-clustering-benchmark \
  --output data/processed/benchmark_manifest_v1 \
  --historical-audit-root ../BGC_retrieve/data/processed/mibig_training_audit_v1

python -B scripts/build_manifest.py \
  --validate data/processed/benchmark_manifest_v1

python -B -m unittest discover -s tests -v
```

生成命令拒绝覆盖已有目录，重跑时请选新的输出目录。校验命令检查输出哈希、输入哈希、关系完整性、缺失数量及正邻居数与配对统计的一致性。

`--historical-audit-root` 可选，读取该目录的 `confirmed_overlap_mibig_ids.txt` 和 `additional_review_mibig_ids.txt`。省略时历史重合状态为 `not_checked`，不会默认没有重合。使用该参数时，复制的是 ID 参考信息到清单，不复制或修改原始 GBK。

迁移快照后若暂时没有原始输入，可运行：

```bash
python -B scripts/build_manifest.py \
  --validate data/processed/benchmark_manifest_v1 --skip-input-hashes
```

这只验证快照输出和内部关联，返回 `input_hashes_checked=0`，不代表验证过迁移后的原始数据。输入源码也记录哈希，因此修改脚本后，旧快照的完整来源校验会报告差异；旧结果仍可做输出校验，需要新结果则生成新快照。

## 输出文件

| 文件 | 一行代表什么 | 主要内容 |
| --- | --- | --- |
| `bgc_manifest.tsv` | 一个 BGC ID | 矩阵位置、DNA 可用性、全部产物 ID、骨架计数、历史审计标记、待核验状态 |
| `products.tsv` | 一个有关联 BGC 的产物 | 原始结构字段、名称、ID、URL、源行号、分子式一致性标记 |
| `bgc_products.tsv` | 一条 BGC—产物关联 | `bgc_id`、`product_id`、产物源行号 |
| `dna_files.tsv` | 一个实际 GBK 文件 | 文件和 DNA 哈希、record 信息、长度、GC、模糊碱基、来源分类及解析状态 |
| `bgc_scaffolds.tsv` | 原始骨架文件的一行 | BGC ID、骨架字符串、源行号、空值标记；重复行保留 |
| `issues.tsv` | 一条审查证据 | 实体、问题代码、严重度、JSON 证据 |
| `pair_statistics.tsv` | 一个队列的一种阈值 | 非自身无序正负对数、正邻居覆盖 |
| `historical_overlap_ids.tsv` | 一个历史审计 ID 标记 | 原清单状态、路径和文件哈希，包含未匹配当前队列的参考 ID |
| `input_files.tsv` | 一个来源文件 | SHA256、字节数、角色、路径；覆盖 GBK、源表、矩阵、骨架、定义脚本、许可及清单脚本 |
| `provenance.json` | 一个快照 | 来源 Git 提交、生成命令、环境版本、统计、限制与其余输出哈希 |
| `summary.md` | 一个快照 | 中文结果说明和文件入口 |

TSV 均有表头，使用 UTF-8 和标准 CSV quoting。以 `csv.DictReader(..., delimiter="\t")` 读取；不要用 `str.splitlines()` 处理原始产物表，产物名称中的换页字符会被错误拆成额外行。

## BGC 主表的关键字段

- `matrix_index`：矩阵列顺序对应的零起始位置；若源矩阵行顺序不同，先按 ID 对齐并在元数据记录。没有矩阵记录的 ID 仍保留，位置为空。
- `dna_status`：`available` 仅表示存在唯一可解析、非空、单 record 且字符合法的 GBK；缺文件为 `missing_gbk`。其他解析状态和文件身份冲突分别保留。
- `gbk_file_count`、`gbk_paths_json`：保留同 ID 的全部文件路径。多文件时不擅自选择代表，单文件摘要列为空。
- `gbk_sha256`：原始文件字节哈希。`dna_sha256`：完整 DNA 转大写后的序列哈希。`dna_rc_canonical_sha256`：正向与反向互补序列中字典序较小者的哈希，支持后续准确识别整序列重复。
- `gc_fraction`：`(G+C)/(A+C+G+T)`，不是百分数。N 及其他模糊字符不进入分母，但保留在 DNA 和长度中；分母为零则为空。
- `product_ids_json`：全部关联产物 ID，保留 `BM...` 补充记录；不根据 InChIKey、URL 或 scaffold 合并产物。
- `product_review_ids_json`：有关联字段审查标记的产物 ID，须查看产物表及证据。
- `positive_neighbors_065_full_matrix`：在完整矩阵中满足严格 `S > 0.65` 的其他 BGC 数量。
- `positive_neighbors_065_available_dna`：同一规则在有可用 DNA 的矩阵子集中统计。空值表示该 BGC 不属于此子集；0 表示属于子集但没有正邻居。
- `historical_overlap_status`：历史清单的 ID 匹配结果。`no_id_match_in_reference` 只说明该旧清单无相同 ID，不能解释为无泄漏。
- `current_evaluation_overlap_status`、`homology_status`：本轮均为 `not_checked`。
- `split_status`、`training_eligibility`：本轮均为 `unassigned`、`not_assessed`，即使 DNA 可用也不代表进入训练。

主表覆盖矩阵、产物关联、骨架和 DNA 文件的 ID 并集；有缺失或冲突的 BGC 不会从清单消失。原始文件可读但解析失败时保留失败行；如果连原始文件哈希也无法读取，生成过程会明确停止，避免发布无法验证来源的完整快照。

## 结构字段核对的边界

`formula_consistency_status` 比较源表 `compound_molecular_formula` 与 InChI 的公式层，忽略元素顺序和显式计数 1。只处理简单、无电荷的元素计数；含电荷、质子化、同位素、复杂组分或不支持语法的情况标为 `not_checked`。

`match` 不表示 SMILES 或 InChI 已经过化学验证；`mismatch` 也不会触发自动修复。冲突证据记录同 InChIKey 的源表记录，便于追溯作者补充记录可能复用的元数据。

本次检测到 7 条公式字段不一致：`BM000001–BM000006`，以及 `NPA020256`（Kolossin A，源表第 20074 行，关联 `BGC0001641`）。后者的分子式字段为 `C80H136N15O20`，InChI 公式层为 `C80H137N15O20`；目前仅作待核验标记。前 6 条的共享字段证据可追溯到 `NPA033565`。所有原值与产物关联均保留。

骨架空值、同 BGC 的多个骨架以及重复骨架行属于辅助信息，不自动当作坏样本或互斥分类标签。骨架表没有产物 ID，因此清单不根据行序推断其产物身份。

## 当前结果与下一步

当前快照保留 1438 个 BGC、1343 个 GBK、2351 个关联产物及 2498 条关联；95 个矩阵 ID 缺 GBK。可用 DNA 子集在 `S > 0.65` 下有 1917 个正对、899236 个负对、563 个无正邻居 BGC。

这不是训练集规模承诺。下一步先处理缺失与冲突、核验当前评估集重合和同源分组，再确定划分并分别统计集合内部的监督覆盖。本步骤没有生成训练配对文件，也没有删除重复或待核验样本。

## 后续正例覆盖审计（2026-11-23）

原始清单快照保持不变。后续按实际 DNA 内容排除 B/E/G/H 的确证重合后，分析队列为 1159 个 BGC，ID 见 [remaining_after_exact_screen.txt](../data/processed/begh_content_overlap_v1/remaining_after_exact_screen.txt)。这一步尚未完成近同源隔离或训练划分。

[阈值统计](../data/processed/product_thresholds_v1/threshold_statistics.tsv)采用无序、非自身配对和严格 `S > threshold`。其中 `S > 0.65` 有 1415 个正对，涉及 602 个 BGC，557 个没有正邻居。暂时隔离待复核的 `BGC0002037` 不改变这三个数中的正对数和有正邻居数，只将无正邻居数减为 556。

复现家族／类别覆盖分析（输出目录须不存在）：

```bash
python -B scripts/analyze_positive_coverage.py \
  --threshold-root data/processed/product_thresholds_v1 \
  --threshold 0.65 \
  --output data/processed/positive_coverage_065_v1
```

分析沿用作者已经发布的分组，不在剩余队列上重新聚类。原始标签与成员关系保留；类别重复记录取标签并集，冲突另作标记。原始 `alkalod` 拼写、thiopeptide 等标签不擅自更正或合并。Hybrid 可贡献到多个原子类别，类别行不可直接相加。类别文件存在缺失、冲突及需要生物学核验的标注，因此类别差异不能直接解释为机制差异。

预测 GCF 主统计只使用单一非噪声归属。缺失和多重归属均保留为未确定，不补成 singleton、不选择最后一行覆盖；另列“共享至少一个标签”的多重归属敏感性统计。Butina 是化学分组，预测 GCF 不是人工真值，正例图连通分量也不是家族标签。组中产物名称只表示成员的已知关联，不为整个组命名，也不声称是产生最大相似度的那对产物。

输出在 [positive_coverage_065_v1](../data/processed/positive_coverage_065_v1/)：

| 文件 | 用途 |
| --- | --- |
| `class_statistics.tsv` | 原子类别的总 BGC、有正邻居 BGC、同类别正对；Hybrid 行可重叠 |
| `class_signature_statistics.tsv` | 完整类别组合的互斥统计，含缺失类别组 |
| `group_statistics.tsv` | 每个方法每个组的成员 ID、产物示例、组内正对和正例端点权重 |
| `method_coverage.tsv` | 各方法可归属、缺失、多标签的数量，以及组内／跨组／未确定正对 |
| `group_concentration.tsv` | 按端点权重排序的前 K 组集中度，分母为全部 2830 次端点出现 |
| `positive_group_pairs.tsv` | 两个已明确归属组之间的正对数，含同组情形 |
| `bgc_annotations.tsv` | 每个 BGC 的正邻居数、全部产物名、原始类别和各方法归属状态 |
| `positive_pairs.tsv` | 1415 个审计用正对及类别／分组关系；不是已划分的训练配对 |
| `positive_components.tsv` | 正例关系图的连通分量，按正对数排序；仅描述关系集中度 |
| `source_classes.tsv`、`source_cluster_memberships.tsv` | 保留源行号的原始类别／分组记录 |
| `summary.json`、`provenance.json` | 统计口径、缺失／冲突、来源和输出哈希 |

`internal_positive_pairs` 只计组内正对；`incident_positive_pairs` 计至少一端在该组的正对；`positive_degree_sum` 计该组成员在所有正对中出现的次数，即组内边计 2、跨组边计 1。后者在覆盖完整且互斥的分组下合计为 `2 × 1415 = 2830`。所有覆盖率均为划分前统计，训练／验证／测试分开后须重新计算。

## 后续 DNA 分组与划分（2026-11-25）

原清单及上述审计快照保持不变。最新 [DNA 分组与划分结果](../data/processed/dna_homology_split_v1/summary.md)已按固定 DNA 近同源规则隔离 208 个有 DNA 的候选，剩余 1135 个 BGC 分为 train 908、validation 114、test 113；其集合内部 `S > 0.65` 正对分别为 888、25、21。

实际使用新的 `bgc_split_manifest.tsv` 和三份 `*_ids.txt`，不要把本页前述 1159 队列或初始 `unassigned` 状态当成当前训练划分。完整定义、搜索来源、单独保留的局部相似性及复现命令见 [homology-split.md](homology-split.md)。
