# 动态鬼成像与运动图重建

本项目研究运动引导的时空图聚合能否改善动态计算鬼成像的重建质量。当前主线是在冻结的交替时空 Transformer 粗重建之后，利用逐帧 GI 测量残差和估计运动进行图细化，并与 CNN 细化、固定坐标图细化比较。

仓库保存笔记本阶段的代码、研究笔记和已有实验结果。版本 `v0.1.0-notebook-baseline` 是后续服务器实验的起点。当前代码保留已有实验协议和本机路径，服务器适配将在后续版本完成。

## 实验范围

数据为单数字全局平移的合成 Moving MNIST，图像大小 64×64，每条序列 8 帧。当前主协议使用 800 条训练序列、120 条验证序列，测试按保留身份块组织；散斑数为 128 或 256，分别对应 3.125% 和 6.25% 采样率。

每帧独立获得 bucket 测量，细化器使用粗视频、测量残差回投影和估计运动。当前实现借鉴 MadyGraph 的方法思路，属于逐帧 GI 适配原型；与原论文的视频快照压缩成像观测模型不同。

早期“图模型直接替换 Transformer”的实验也保存在仓库中，供研究路线追溯。两条路线采用不同的比较对象，应分别解读结果。

## 代码入口

| 入口 | 用途 |
|---|---|
| `train_madygraph_gi_plugin.py` | 当前 CNN、固定图、运动图细化器训练与评估 |
| `train_graph_protocol_v2.py` | 受控数据协议与 Transformer、直接图模型训练 |
| `train_spatial_motion_diagnostic.py` | 当前细化器使用的运动估计器训练 |
| `evaluate_graph_protocol_v2.py` | 直接重建路线的受控评估 |
| `benchmark_madygraph_gi_plugin_efficiency.py` | 已有细化方案效率测量脚本，运行前需核对残差接口 |
| `evaluate_madygraph_measurement_noise.py` | 测量噪声实验 |
| `mini_repro_dynghost.py`、`train_spatiotemporal_transformer.py` | GI 仿真和早期交替时空 Transformer 基础代码 |

完整实验交接见 [动态 GI 研究交接文档](动态GI动态图研究_完整交接文档_2026-09-29.md)。该文档记录截至 2026-09-29 的研究状态。当前仓库与版本说明以本 README 为准，实验结果仍以对应输出文件为准。

## 已有结果

以下数据来自 `madygraph_gi_plugin_v2_outputs/benchmark_summary.csv` 的 `speed_bin=all` 行：测试块 1、120 条序列、细化器种子偏移 0。它是一次已有实验，不能替代多个种子和其他测试块的统计。

| 散斑数 | 方法 | PSNR / dB | SSIM |
|---:|---|---:|---:|
| 128 | Transformer 粗重建 | 18.19 | 0.699 |
| 128 | CNN 细化 | 18.53 | 0.734 |
| 128 | 固定图细化 | 18.26 | 0.732 |
| 128 | 运动图细化 | 19.38 | 0.744 |
| 256 | Transformer 粗重建 | 20.36 | 0.825 |
| 256 | CNN 细化 | 20.81 | 0.844 |
| 256 | 固定图细化 | 20.66 | 0.841 |
| 256 | 运动图细化 | 21.52 | 0.853 |

| 输出位置 | 内容 |
|---|---|
| `madygraph_gi_plugin_v2_outputs/` | 当前向量化细化实验、逐序列结果、配对增益、数据清单与效率记录 |
| `madygraph_gi_plugin_seed1_outputs/`、`madygraph_gi_plugin_seed2_outputs/` | 其他细化器随机种子的已有结果 |
| `madygraph_gi_motion_block2_outputs/` | 运动图在另一个保留测试身份块上的结果 |
| `graph_protocol_v2_trained/`、`graph_protocol_v2_outputs/` | 直接重建路线的受控比较 |
| `transformer_vs_joint_repeat_summary.json`、`joint_motion_tgnn_repeat_summary.json` | 早期多种子模型对比汇总 |
| `madygraph_gi_plugin_noise_outputs/` | 测量噪声条件下的结果 |
| `notebook_verification_outputs/` | 已有算子、闭环和过拟合检查记录 |

`*_smoke/`、`*_train_smoke/` 保存的是小规模运行检查，应与正式训练结果区分。`madygraph_gi_plugin_outputs/test_visuals.png` 是早期测试的图像示例，与 v2 或测试块 2 的结果不能直接混用。

现有结论适用于上述合成数据和模型规模。自然视频、真实光学 GI 测量和更大模型尚待验证。三个细化器种子的实验共享粗重建器与运动估计器，随机性只覆盖细化器训练。

## 环境与运行

笔记本环境的实际版本记录在 [environment_notebook.json](environment_notebook.json)。`requirements.txt` 给出直接运行依赖，属于依赖清单；安装后需确认 PyTorch 与 torchvision 匹配，CUDA 安装方式应按实际 GPU 环境选择。

```powershell
python -m pip install -r requirements.txt
python train_madygraph_gi_plugin.py --help
```

现有笔记本 Python 位于 `D:\Anaconda\envs\pytorch_env\python.exe`。在当前电脑、完整 MNIST 数据与既有权重就位后，可使用以下命令进行新的小规模细化训练。输出使用新目录，避免覆盖已归档结果。

```powershell
python train_madygraph_gi_plugin.py --epochs 5 --variants cnn,fixed,motion --seed-offset 0 --test-block 1 --output-dir runs/local_refiner_seed0
```

上面的命令会进行训练和测试，文档示例未在此次 Git 整理过程中执行。服务器正式实验计划使用新的保留测试块，并在适配时拆分训练和最终测试流程。

## 数据、权重与路径

数据、权重和论文 PDF 保存在本地，未进入 Git。克隆仓库后需要另行准备这些资源。

| 资源 | 当前本机位置 |
|---|---|
| 完整 MNIST 原始数据 | `D:\download\DynGhost-mini-repro\data\MNIST\raw` |
| Transformer 粗重建权重 | `graph_protocol_v2_trained/best_transformer.pt` |
| 运动估计器权重 | `spatial_motion_outputs/best_spatial_motion_net.pt` |
| 其他已训练模型 | 相应实验输出目录中的 `.pt` 文件 |
| 参考论文 | `动态CGI/`、`原文献/` 中的 PDF |

当前仓库根目录中的 `data/MNIST/raw` 只有训练图像及其压缩文件，不能视为完整 MNIST 数据。依赖模块已经补入本仓库根目录，但多个研究脚本仍将 `PROJECT` 或 `DEFAULT_PROJECT` 指向 D 盘旧项目。部分旧脚本还读取旧项目的输出；服务器迁移时需要统一数据、模型和输出路径。

效率脚本与部分可视化入口保留了历史实现，运行前应核对 checkpoint、测试块及 `physical_residual(bucket, coarse, patterns)` 的参数约定。本次归档保存已有文件，没有重新生成指标或图表。

## 源码来源与版本

本仓库根目录固定为 `C:\Users\lingye\Desktop\文献\文献及笔记`，分支为 `main`。当前研究文件先形成独立快照，再补齐基础模块与说明。

补入的 10 个基础模块来自 `D:\download\DynGhost-mini-repro` 的实际工作目录。该项目当时的 HEAD 为 `130c64e3e35482dbcdc1516f29afe1c25d0d9656`。其中 7 个模块受旧 Git 仓库管理，另 3 个模块当时尚未提交；逐文件 SHA256、旧仓库跟踪状态和来源记录见 [source_provenance.json](source_provenance.json)。

`v0.1.0-notebook-baseline` 标记整理完成的笔记本阶段版本。此次提交使用实际日期，不将此前实验记录日期写成 Git 提交日期。

Git 保存实验代码、自写 Markdown/Word 笔记、CSV/JSON 指标及结果 PNG；忽略数据、模型权重、论文 PDF、缓存和临时文件。完整实验恢复仍需同时保存本地模型与数据。
