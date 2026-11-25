# DNA 训练服务器环境

本项目统一使用根目录的 [`requirements.txt`](../requirements.txt)，原来的两个 requirements 文件已删除。该文件用于补齐服务器已有的 `natural_product` 环境，或其 Conda 克隆；它固定已确认的核心版本，同时列出辅助依赖的兼容范围。它不是在空环境中重建历史 CUDA 二进制的完整锁文件。

## 1. 已确认的配置与本次问题

以下来自 2026-11-25 用户在**发生报错的同一环境**执行命令的输出，不再采用本地 CPU 测试环境推断服务器版本：

| 项目 | 实际值 | 处理方式 |
| --- | --- | --- |
| Python | 3.11.11，Conda 环境 `natural_product` | 保留 |
| 解释器 | `/home/s252613/miniforge3/envs/natural_product/bin/python` | 用此环境的 `python -m pip` 安装 |
| GPU | NVIDIA L40S，46068 MiB，标称 48 GB | 单 GPU、BF16 |
| 驱动 | 580.82.07 | 本次无需更换 |
| PyTorch | 2.6.0.dev20241112 | 保留已有开发版二进制 |
| PyTorch 构建 CUDA | 12.4 | 保留 |
| PyTorch CXX11 ABI | `False` | 与现有 FlashAttention 配套 |
| 本机 CUDA toolkit | 12.4，nvcc 12.4.131 | 保留；不同于驱动版本 |
| FlashAttention | 2.8.3.post2 | 保留，不重新编译 |
| Transformers | 5.1.0 | 固定 |
| Tokenizers | 0.22.2 | 固定；原始 DNA BPE 文件不变 |
| Accelerate | 1.5.2 | 固定 |
| NumPy / Biopython | 2.2.4 / 1.85 | 固定 |
| PEFT | **未安装** | 补入 **0.18.1** |

`ModuleNotFoundError: No module named 'peft'` 的直接原因是这次运行使用的解释器没有 PEFT。这个报错本身不是显存不足，也不证明 PyTorch 或 tokenizer 不兼容。原 requirements 虽然列出了 PEFT，但没有在该解释器中完成安装；其 `torch==2.10.0`、`numpy==2.5.3` 来自本地测试环境，后者要求 Python ≥3.12，确实不适合原样用于本服务器。

新增的 PEFT 0.18.1 包含 Transformers v5 兼容修复，官方声明要求 Python ≥3.10、Accelerate ≥0.21。经核对，项目所用 PEFT 导入的 Accelerate API 在 1.5.2 中已存在，无需因此升级 Accelerate 或 PyTorch。版本依据：[PEFT 发布说明](https://github.com/huggingface/peft/releases/tag/v0.18.1)、[PEFT 依赖声明](https://github.com/huggingface/peft/blob/v0.18.1/setup.py)、[Accelerate 1.5.2 API](https://github.com/huggingface/accelerate/blob/v1.5.2/src/accelerate/utils/__init__.py)。

Transformers 5.1.0 要求 `huggingface-hub>=1.3.0,<2.0`、`tokenizers>=0.22.0,<=0.23.0`，运行时要求 PyTorch ≥2.4。旧 `BGC_annotation/environment.yml` 中的 Hub 0.30.1 只是历史记录，不能直接作为当前版本。未报告的辅助包使用上游兼容范围，安装时由 pip 补齐；安装后的完整版本另存快照。[Transformers 依赖声明](https://github.com/huggingface/transformers/blob/v5.1.0/setup.py)、[运行时检查](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/utils/import_utils.py)。

## 2. 在现有环境安装

以下命令在服务器**仓库根目录**执行，该目录应包含 `requirements.txt`、`configs/`、`dna_training/` 和 `scripts/`。先更新仓库代码，再激活环境：

```bash
conda activate natural_product
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip --version
```

如果希望原 `natural_product` 的软件包也完全保留，可先克隆环境，之后在克隆中执行本页全部步骤：

```bash
conda create --name dna_training --clone natural_product
conda activate dna_training
```

安装前保存快照；`runs/` 已被 Git 忽略。下面的变量仅用于本次环境记录，在同一 shell 中继续执行后续命令：

```bash
env_snapshot="runs/environment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$env_snapshot"
python -m pip freeze > "$env_snapshot/pip-before.txt"
conda list --explicit > "$env_snapshot/conda-before.txt"
python -m pip check > "$env_snapshot/pip-check-before.txt"
```

先查看安装计划，再安装：

```bash
python -m pip install --dry-run -r requirements.txt
python -m pip install -r requirements.txt
python -m pip check
```

PyTorch、FlashAttention 和已匹配的核心包应显示 `Requirement already satisfied`。主要新增项是 `peft==0.18.1`；未满足范围的辅助包也可能更新。若计划替换或编译 PyTorch / FlashAttention，先检查是否激活了正确的环境。不要加 `--upgrade`、`--force-reinstall`、`--ignore-installed` 或 `--no-deps`：本次要补齐完整依赖，同时保留现有 GPU 二进制。

如果提示找不到 `torch==2.6.0.dev20241112`，通常说明当前环境没有这份历史 Conda 安装。该 nightly 不保证仍能从当前 PyPI 下载；应回到已有环境，或克隆它。不要删掉版本约束后让 pip 自动升级 PyTorch。`pip check` 检查的是整个环境；若已有其他项目的冲突，保留具体输出后处理，不要通过批量升级隐藏它。

## 3. 验证解释器、包版本和 GPU

下面先检查每条 requirements 约束，再实际导入模型依赖和 CUDA 扩展。它不读取或修改原 checkpoint。

```bash
python - <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
from packaging.requirements import Requirement

assert sys.version_info[:2] == (3, 11), sys.version
print("Python:", sys.version, sys.executable)
for line in Path("requirements.txt").read_text().splitlines():
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    requirement = Requirement(line)
    version = metadata.version(requirement.name)
    assert requirement.specifier.contains(version, prereleases=True), (line, version)
    print(f"{requirement.name}: {version}")

import torch
import transformers
import peft
import accelerate
import flash_attn
from transformers import ModernBertForMaskedLM
from peft import LoraConfig, get_peft_model

assert torch.version.cuda == "12.4", torch.version.cuda
assert not torch._C._GLIBCXX_USE_CXX11_ABI
assert torch.cuda.is_available(), "CUDA unavailable in this process"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable on selected device"
print("CUDA:", torch.version.cuda, "CXX11_ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("GPU:", torch.cuda.get_device_name(0))
print("Imports and version checks passed")
PY

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version
```

仅列出已安装版本不能证明 CUDA 扩展能运行。继续执行一个小型 BF16 变长 attention 前向／反向，检查 FlashAttention 的实际内核和梯度：

```bash
python - <<'PY'
import torch
from flash_attn import flash_attn_varlen_qkvpacked_func

qkv = torch.randn(8, 3, 12, 64, device="cuda", dtype=torch.bfloat16,
                  requires_grad=True)
cu_seqlens = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
output = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, 5, dropout_p=0.0)
output.float().square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(output).all()
assert qkv.grad is not None and torch.isfinite(qkv.grad).all()
assert qkv.grad.abs().sum() > 0
print("FlashAttention BF16 forward/backward passed:", tuple(output.shape))
PY
```

完成后保存实际解析的版本及验证结果：

```bash
python -m pip freeze > "$env_snapshot/pip-after.txt"
conda list --explicit > "$env_snapshot/conda-after.txt"
python -m pip check > "$env_snapshot/pip-check-after.txt"
python -m unittest discover -s tests -v > "$env_snapshot/tests.log" 2>&1
```

`pip freeze` 记录 Python 包，`conda list --explicit` 记录 Conda 构建来源；它们都不能单独保存 GPU 驱动或已不可下载的旧 wheel。本页的核心版本约束、服务器输出和这两份快照共同用于追溯。小型内核检查与合成模型测试通过后，仍需按[训练指南](dna-training.md#42-在真实-gpu-上做-lora-资源探测)运行真实 checkpoint 的全长 probe，确认中位数、P90 和最长输入的能力。使用新的报告目录，例如 `runs/dna_probe_lora_s42_envfix`。

## 4. FlashAttention 二进制与重建边界

FA2 2.8.3.post2 支持 Ada GPU 和 BF16，要求 PyTorch ≥2.2、CUDA ≥12.0。其预编译 wheel 匹配同时涉及 Python、PyTorch 主次版本、CUDA 主版本、Linux 平台和 CXX11 ABI；只看 `flash-attn==2.8.3.post2` 不足以保证二进制兼容。本次保留已有组合，实际可用性以上面的导入／内核检查为准。[官方安装要求](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/README.md#installation-and-features)、[wheel 匹配代码](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/setup.py#L406)。

若未来需要在空环境重建，应另行确定可获取的 PyTorch 构建并重新验证，不能声称这份历史 nightly 已被完全复现。如果必须源码编译 FA2，需要已安装的目标 PyTorch、CUDA 12.4 开发工具链、兼容 C++ 编译器，以及 `packaging`、`psutil`、`ninja`、`setuptools`、`wheel`、`einops`；上游安装使用 `--no-build-isolation`，并可用 `MAX_JOBS` 限制编译并行度。缺少 PEFT 时**无需执行这一步**。

训练使用原始 DNA tokenizer，不需要 `sentencepiece` 或 `tiktoken`；既有矩阵和同源划分快照也不要求安装 RDKit、BLAST+、MMseqs2。只有重新运行同源搜索才需要单独的 BLAST+ 工具。单独运行清单／数据审计可只安装 `numpy==2.2.4 biopython==1.85`，不必安装 GPU 依赖；历史快照实际生成环境仍以各自 `provenance.json` 为准。

LoRA 加载器现在会在读取 checkpoint 张量前检查 PEFT。缺包时提示当前解释器对应的安装命令；如果是 PEFT 内部依赖不兼容，保留原始错误，不把所有导入失败都解释成缺包。安装环境和这些检查不会写入师兄的原始 checkpoint；训练输出仍只写入指定的新 `runs/` 目录。

## 5. 本次验证范围

在独立临时环境中使用 **Python 3.11.11、官方稳定版 PyTorch 2.6.0+cpu**，搭配服务器相同的 Transformers 5.1.0、Tokenizers 0.22.2、Accelerate 1.5.2、NumPy 2.2.4、Biopython 1.85，以及新增的 PEFT 0.18.1，`pip check` 无冲突，完整 **104 项测试通过**。辅助包由 pip 解析，其中 Hub 为 1.33.0，Safetensors 为 0.8.0；这些实测值没有被强制锁给服务器。

测试涵盖合成 ModernBERT 的 LoRA 梯度、保存／恢复、完整训练流程，以及缺失或损坏 PEFT 时的提前报错。此环境使用稳定 CPU 版作为兼容性检查，**没有复现服务器的旧 nightly、CUDA、FlashAttention 二进制或真实 checkpoint**；服务器最终结论必须来自本页检查和完整 DNA probe。
