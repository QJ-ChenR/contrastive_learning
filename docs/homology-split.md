# DNA 近同源分组与 BGC 划分

本流程为 DNA encoder 微调生成可复现的训练、验证和测试 BGC 名单。先只用 DNA 序列构建不可跨集合的分组，再统计产物监督覆盖。分组是固定规则下的泄漏隔离单元，不是 GCF、产物类别或完整的进化同源关系。

两个入口分别为 [`run_dna_homology.py`](../scripts/run_dna_homology.py) 和 [`build_homology_split.py`](../scripts/build_homology_split.py)。搜索与划分写入不同的新目录，不覆盖原始数据、初始清单或历史审计。实际完成状态和数量以对应快照的 `provenance.json`、`summary.json` 为准；只有来源与输出校验通过、schema 为 `dna_homology_search_v1` 的搜索快照才能作为完整主搜索。记录了中止状态的 partial 快照即使有 `provenance.json`，也不是完整搜索结果。

## 固定的序列关系规则

联合输入包含全部 **1343 个有 DNA 的候选 BGC**，以及 **421 个 B/E/G/H 外部评估记录**。已经发现精确重合的候选也保留为图节点，以保留经过这些节点的传递关系。外部记录用 `EVAL_` 前缀与候选 BGC 区分。每个 GBK 使用完整、原方向、大写 DNA；不截断、不按长度剔除，模糊碱基保留。

全序列同时作为查询和数据库运行 BLASTN，双链搜索。完整主搜索固定使用 **`-task megablast -word_size 16`**，开启 DUST 和 soft masking；最大目标数等于全部序列数，不设置 `max_hsps`。`max_target_seqs` 控制保留的目标数，未设置 `max_hsps` 时保留满足期望值条件的 HSP。[NCBI 参数说明](https://www.ncbi.nlm.nih.gov/books/NBK279684/)。

初始尝试为 `-task blastn -word_size 11`。由于全队列搜索耗时，在尚未生成分组或读取化学监督用于划分时进行了可运行性探测：同一条 59,320 bp 查询对同一数据库，blastn11 耗时 74.79 秒，megablast16 为 7.22 秒。测量时存在其他搜索的资源竞争，这不是完整队列的稳定加速比或完成时间保证。最终主搜索采用 megablast16，后处理的 80% 一致性、80% 短侧覆盖、100 比对列和 E-value 门槛不变。

更换 task 和种子会改变启发式召回及默认打分／延伸行为。NCBI 将 megablast 定位于较相近的序列，blastn 的默认种子更短；这里明确设置 16，而不是 megablast 默认的 28。[任务说明](https://www.ncbi.nlm.nih.gov/books/NBK569839/)、[具体参数默认值](https://www.ncbi.nlm.nih.gov/sites/books/NBK279684/table/appendices.T.blastn_application_options/)。因此本协议表示 **megablast16 检出证据下的操作性近同源隔离**，不声称与 blastn11 或穷举比对具有相同召回，也不把下述阈值视为已验证的生物学界限。

中止的原搜索保留于 `data/processed/dna_homology_search_word11_partial/`，schema 标为 incomplete。其字面相等／包含扫描已经完成，可以单独复用；复用必须确认完整 FASTA 与节点清单哈希一致，并重新验证每条精确边的坐标。原目录中的部分 BLAST HSP 不代表完成了全部查询，不能据此作全队列阴性判定。

在查看分组或化学统计前，另外固定纳入该 word11 搜索中已产生的**全部完整、可严格解析的 HSP**，避免舍弃已检出的近同源证据。通过 `--supplement-search` 明确指定这一补充来源，先校验来源哈希，再与完整主搜索的 HSP 合并；所有来源使用同一后处理门槛，最后统一计算两侧区间并集，而不是先分别成边再取并集。仅末尾没有换行结束的记录被隔离，记录该尾部字节数和哈希；内部坏行会导致失败，不静默跳过。

word11 补充覆盖的查询范围不均匀，也不是第二次完整搜索。实际分组因此基于完整 megablast16 加既有 word11 补充证据，不能假设每条 BGC 都接受了相同程度的 word11 搜索，或用该部分结果估计全队列漏检率。主搜索的 query 覆盖单独统计，不能用补充来源的 ID 填补主搜索覆盖检查。

每条局部比对 HSP 必须同时满足以下条件，边界均包含等号：

| 条件 | 定义 |
| --- | --- |
| DNA 一致性 | `nident / length >= 0.80` |
| 最短 HSP | `length >= 100`；这里是 BLAST 比对列数，包含 gap 列 |
| 显著性 | `E-value <= 1e-10` |

同一无序序列对汇总通过上述条件的全部 HSP，包括两个查询方向和明确登记的补充来源。使用 BTOP 还原配对碱基区间：匹配与错配碱基均计入比对覆盖，任一侧出现 gap 的列在两侧都不计覆盖。BLAST 的 1-based inclusive 坐标转换为原序列的 0-based half-open 区间，反链也映射回原坐标。每侧分别计算区间并集，重叠 HSP、反向查询、跨来源重复命中不重复累加。BTOP 及各字段一致性也会验证；输出字段来自 NCBI 支持的自定义格式。[输出格式说明](https://www.ncbi.nlm.nih.gov/books/NBK569862/)。

两侧覆盖率均以各自**完整 DNA 长度**为分母。较短序列的覆盖率 `>= 0.80` 时建立近同源边；长度相同则保守采用两侧覆盖的较小值。保留两侧覆盖率、HSP 数、一致性范围和期望值，不将多 HSP 的覆盖率称为整段全局一致性。区间并集不要求所有 HSP 构成一条共线比对链。

另对所有无序序列对进行正反向的字面全长相等／完整包含检查，发现的关系直接作为边加入，不依赖 BLAST 的启发式搜索、DUST 或 HSP 门槛。`N` 在该检查中只是字面字符，不是任意碱基通配符。

## 联合分组、隔离和划分

对近同源边与字面相等／包含边的并集求连通分量。A–B、B–C 即使没有 A–C 直接边，也属于同一隔离组。组 ID 由排序成员 ID 的哈希确定；整个分量不可拆开。

- 包含任何 B/E/G/H 节点的分量，其全部候选 BGC 隔离，不进入内部训练、验证或测试。
- 历史仍待复核的候选作为隔离种子，其整个分量同样隔离。当前来源审计中的 `BGC0002037` 属于这一类。待复核隔离与已证实 DNA 关系的原因分别保留，不把待复核改写为确证。
- 缺 DNA 的候选保留在输出总清单中，标记 `excluded_missing_dna`，不参与图或分配。
- 其余完整分量参与内部划分；同一 BGC 的全部产物关系和后续派生视图继承相同划分。

预先固定 `seed=42`，以 BGC 数为目标比例 `train:validation:test = 80:10:10`。算法按组的 BGC 数从大到小处理，同大小组先按固定种子打乱；每一步选择使 `sum(actual_BGC_count**2 / target_BGC_count)` 最小的集合，相同目标值按 train、validation、test 顺序确定。若至少有三个可用组，最后会为尚为空的集合保留整组。实际比例受不可拆分组的大小约束，并不保证精确 80:10:10 或全局最优。

分配前不读取产物相似度，不依据正例数、类别覆盖或后续成绩挑选 seed、同源阈值或划分。化学正例图的连通分量不作为此处的同源组。

## 划分后的产物监督统计

固定分配后才读取作者化学相似度矩阵，在各集合内分别计算 `S > 0.60/0.65/0.70/0.75` 的无序、非自身配对统计。主正例表采用严格 `S > 0.65`，等于阈值不算正例。只有 `split=train` 的配对可用于模型训练；验证／测试配对和跨集合化学关系不得回流为训练监督。

无正邻居的 BGC 仍保留。训练阶段需另外明确它们的负例池和损失角色；不能为提高覆盖而静默移除。`retrieval_coverage.tsv` 另外报告各集合 query 对训练 gallery 的正邻居覆盖，排除自身检索。这是监督可用性诊断，不是检索性能。

## 输出文件

搜索快照默认位于 `data/processed/dna_homology_search_v1/`：

| 文件 | 用途 |
| --- | --- |
| `sequence_manifest.tsv`、`sequences.fasta` | 全部图节点、角色、完整 DNA、GBK／DNA 哈希 |
| `exact_edges.tsv` | 字面相等或完整包含的边及原序列坐标 |
| `blast_hsps.tsv.gz` | 完整 BLAST HSP 输出，保留 BTOP，供重算与复核 |
| `commands.json`、`makeblastdb.log`、`blastn.log` | 实际命令和工具日志 |
| `protocol.json`、`input_files.tsv`、`provenance.json` | 固定协议、来源校验、工具版本／二进制及产物哈希 |

划分快照默认位于 `data/processed/dna_homology_split_v1/`：

| 文件 | 用途 |
| --- | --- |
| `bgc_split_manifest.tsv` | 全部原始候选、全部产物 ID、分组、划分及排除原因 |
| `train_ids.txt`、`validation_ids.txt`、`test_ids.txt` | 下游读取的 BGC 名单 |
| `excluded_ids.txt`、`excluded_missing_dna_ids.txt` | 分别记录整组隔离与缺 DNA 的候选 |
| `homology_groups.tsv`、`homology_members.tsv` | 组成员、候选／外部数量、隔离种子、集合归属 |
| `homology_edges.tsv`、`exclusion_paths.tsv` | 直接边以及候选连接到外部／待复核种子的可追溯路径 |
| `homology_pair_diagnostics.tsv` | 短侧覆盖至少 50% 的 HSP 配对证据，包含是否达到成边标准 |
| `cross_split_partial_hits.tsv` | 跨集合短侧覆盖为 50% 至不足 80% 的局部命中；至少一端属于内部划分，也可能涉及外部或已隔离组 |
| `alignment_statistics.json` | HSP 过滤统计、覆盖定义、报告命中和自匹配的 query ID，以及未出现的 ID |
| `bgc_products_split.tsv` | 每条 BGC—产物关系对应的 BGC 分组与划分 |
| `positive_pairs.tsv`、`pair_statistics.tsv`、`positive_neighbors.tsv` | 各集合内部正例、候选阈值统计和每个 BGC 的正邻居数 |
| `positive_pair_locations.tsv`、`retrieval_coverage.tsv` | 正例落在哪些集合，以及 query 对训练 gallery 的覆盖 |
| `summary.md`、`summary.json` | 实际队列与划分后监督规模 |
| `protocol.json`、`input_files.tsv`、`provenance.json` | 协议、输入／代码／输出哈希和版本 |

`alignment_statistics.json` 中的 query ID 来自原始 HSP 行，在额外的 HSP 过滤前统计。outfmt 6 不列出零命中查询，因此未出现的 ID 只表示没有报告 HSP，不能据此证明没有执行查询或没有任何同源关系。运行日志、完整 FASTA、命令成功退出和 ID 统计共同用于检查搜索覆盖。

## 复现与验证

需要 Python、NumPy、Biopython，以及同一 BLAST+ 安装目录下的 `blastn` 和 `makeblastdb`。Python 依赖统一见 [`requirements.txt`](../requirements.txt)；仅做数据处理可单独安装 `numpy==2.2.4 biopython==1.85`，无需 GPU 训练依赖。BLAST+ 实际版本与二进制 SHA256 保存在搜索快照。以下本地路径可替换为服务器路径，输出目录必须尚不存在：

```bash
/home/qiju/miniforge3/envs/BGC/bin/python -B scripts/run_dna_homology.py \
  --manifest data/processed/benchmark_manifest_v1 \
  --overlap data/processed/begh_content_overlap_v1 \
  --blast-bin /tmp/bgc-homology-tools/bin \
  --threads 8 \
  --task megablast \
  --word-size 16 \
  --output data/processed/dna_homology_search_v1

/home/qiju/miniforge3/envs/BGC/bin/python -B scripts/build_homology_split.py \
  --search data/processed/dna_homology_search_v1 \
  --supplement-search data/processed/dna_homology_search_word11_partial \
  --output data/processed/dna_homology_split_v1

/home/qiju/miniforge3/envs/BGC/bin/python -B scripts/build_homology_split.py \
  --validate data/processed/dna_homology_split_v1

/home/qiju/miniforge3/envs/BGC/bin/python -B -m unittest discover -s tests -v
```

搜索入口核验初始清单、旧审计和实际 GBK 哈希，并要求重新找到旧审计的全部精确重合关系。划分入口核验搜索快照后才建组；最终校验检查 BGC 身份与产物保留、组不跨集合、合格边不跨组、外部／待复核组未进入内部划分、ID 名单与主表一致、正对没有跨集合。

若使用已经完成的同一输入字面扫描，搜索命令可额外指定 `--reuse-exact data/processed/dna_homology_search_word11_partial`。这只复用经过哈希及逐边坐标验证的完整精确扫描，不将中止的 BLAST 搜索标记为完成；完整主搜索仍会执行全部查询。

主搜索本身不依赖旧 partial：省略 `--reuse-exact` 时重新执行全部字面扫描，随后运行完整 megablast16。划分入口的 `--supplement-search` 也是可选项；若只使用完整主搜索，可省略该参数并选新输出目录，例如 `data/processed/dna_homology_split_primary_only_v1`。这是未加入既有 word11 证据的独立结果，不能当作本次补充方案的精确复现；要重现本次分组，须保留同一 partial 快照及其校验信息。

快照迁移后暂时没有原始输入时，可加 `--skip-input-hashes` 仅校验输出及内部关系；此时 `input_hashes_checked=0`，不能声称核验过迁移后的来源。重复运行应选新目录。命令、时间、路径等元数据可能不同；固定输入、算法和种子用于稳定产生同一分组和 BGC 归属。

## 解释边界

该协议隔离固定 DNA 规则能够检出的相近／包含序列，并没有完成蛋白远缘同源、化学骨架、物种或 GC 背景的独立隔离。BLAST 是启发式局部搜索；task、DUST、种子和 HSP 门槛都会影响检出。megablast16 可能漏检 blastn11 能找到的短或阈值附近的 HSP；后处理阈值相同不代表搜索召回相同，不能称为所有同源已清除。

短侧覆盖规则允许较短 BGC 被较长 BGC 包含，有助于隔离边界不同的近重复；也可能因很短片段、重复区域、区间并集或连通链而合并过多候选。两侧覆盖、长度与传递路径因此完整保留。跨集合的 50%–80% 局部关系仍可能存在，单独列出，不因看到诊断结果而改动当前划分。

本协议不是未见化学骨架或未见物种泛化协议。B/E/G/H 已参与前期方法比较，继续作为固定外部比较集合，不称为全新盲测。数据划分完成也不等于训练已完成或模型资源已验证。
