# 动态鬼成像与运动图重建

本项目研究运动引导的图聚合能否改善动态计算鬼成像的重建质量。当前实验以同一个冻结的交替时空 Transformer 作为粗重建器，比较 CNN、固定坐标图和运动图三种细化方案。早期直接用图模型替代 Transformer 的实验仍保存在仓库中。

## 工程目录

本机唯一项目根目录为 `C:\Users\lingye\Desktop\文献\文献及笔记`。代码、完整 MNIST、必需权重和实验结果均已集中到这里。运行程序根据自身位置确定根目录，无需配置旧 D 盘项目路径。整体移动文件夹后，目录内部的相对位置保持不变即可。

```text
项目根目录/
├── run_server_experiment.py        # 统一运行入口
├── project_paths.py                # 根目录与资源定位
├── configs/mnist_block3.json       # 已完成的 pilot 配置
├── configs/mnist_scaleup.json      # 已完成的扩大实验配置
├── scripts/                        # Linux 启动与本地检查
├── data/MNIST/raw/                 # 完整的四个 MNIST 原始文件
├── checkpoints/                    # 冻结的 Transformer 与运动估计器
├── runs/                           # 新训练、验证、最终测试和计时输出
├── dist/gi_project_scaleup_server.zip # 扩大实验的完整工程包
├── server_results/                 # 服务器结果、核查记录与图表
└── *_outputs/                      # 已归档的研究结果
```

GitHub 保存代码、笔记、CSV、JSON 和结果图；数据、权重、原始运行目录与压缩包由 `.gitignore` 排除，`server_results/` 中整理后的指标与图表归档入库。仅克隆 GitHub 仓库不能恢复完整运行资源。扩大实验使用 `dist/gi_project_scaleup_server.zip`，其中只有一个顶层目录 `gi_project/`，包括所需数据和权重。资源大小及 SHA256 记录在 `asset_manifest.json`；上传包内另有 `PACKAGE_MANIFEST.json` 和 `CODE_VERSION.json`。

## 当前实验进展

2026-10-14 的扩大实验已完整完成。单卡 RTX 3090 使用 4,000 个训练身份、600 个验证身份、600 个保留测试身份，三个细化方案各训练 20 轮、5 个种子，共 15 次训练。08:44:52 启动，10:07:02 完成结果上传，约 82 分 10 秒；随后自动关机。结果包校验、训练预算、身份划分、批次配对及最终权重签名均已核对。

**扩大实验的整体主判据未通过。** 运动图相对粗重建和固定图仍有改善，但整体平均 PSNR、SSIM 略低于 CNN，完整推理时间约为 CNN 的两倍。

| 散斑数 | 运动图−CNN PSNR（dB） | 配对 95% 区间 | 相对 CNN 的 MSE 增加 | 高速组 PSNR 差值（dB） |
|---:|---:|---|---:|---:|
| 128 | −0.258 | [−0.320, −0.196] | 7.12% | +0.264 |
| 256 | −0.105 | [−0.154, −0.054] | 3.38% | +0.253 |

高速组为 4–6 像素/帧，每个测量率 200 个身份，MSE 相对 CNN 降低约 5%。这一分组结果保留为后续研究线索，不替代整体主判据。学习曲线显示运动图早期改善较快、CNN 后期继续改善；本轮也改变了训练样本量与测试身份，尚不能将与旧实验的差异只归因于训练轮数。研究范围仍为 64×64、8 帧的 MNIST 单数字合成运动序列，尚无自然视频或真实光学测量下的确认结果。

- [扩大实验完整分析](server_results/scaleup_v1/扩大实验结果_20261014.md)
- [最终指标与置信区间](server_results/scaleup_v1/artifacts/gi_results/final/results.json)
- [速度分组图](server_results/scaleup_v1/motion_vs_cnn_by_speed.png)、[验证集学习曲线](server_results/scaleup_v1/validation_learning_curves.png)
- [结果核查记录](server_results/scaleup_v1/结果核查_20261014.json)
- [扩大实验运行说明](SCALEUP_RUN.md)

### 已完成的 pilot

2026-10-12 的 MNIST block 3 服务器 pilot 结果位于 `server_results/block3/gi_results/`：运动图相对 CNN 在 128/256 散斑下平均 PSNR 增益为 0.827/0.815 dB，MSE 降低 16.57%/16.25%，均满足该轮预设判据。以下为 pilot 设置，扩大实验的设置与结果以上述记录为准。

| 项目 | 固定设置 |
|---|---|
| 数据 | MNIST 数字生成的单数字全局平移序列，64×64，8 帧 |
| 训练 / 验证 / 最终测试 | 800 / 120 / 120 个数字身份 |
| 散斑数 | 128、256，对应 3.125%、6.25% 采样率 |
| 比较方法 | Transformer 粗重建、CNN 细化、固定图细化、运动图细化 |
| 训练预算 | 三种细化器分别运行三个种子，每次 5 轮，共 9 次训练 |
| 共享模型 | 粗重建器和运动估计器保持冻结 |
| 最终测试 | 保留的 MNIST 身份块 3，在训练全部完成后统一读取 |

两个采样率混合训练，每个细化器同时支持两者。三个种子只覆盖细化器训练，不代表三个独立训练的完整重建系统。同一随机种子内，三种细化器使用相同训练数据与小批次顺序，模型执行顺序不影响各自的随机初始化和数据顺序。

训练入口默认只使用训练集与验证集，以验证误差选择权重。最终测试由独立入口执行，并记录配置和权重指纹；相同实验再次执行时读取已完成结果，配置或权重变化时会拒绝沿用该次最终测试。

主要判据预先设为：两个采样率下，运动图相对 CNN 的平均配对 PSNR 增益均达到 0.5 dB，95% 置信区间下限大于 0，平均 SSIM 不下降。同时报告相对固定图和粗重建的结果。达标只能支持当前合成数据和模型规模下的结论，自然视频与真实光学 GI 仍需后续实验。

## 指标与计时

质量指标包括 PSNR、SSIM、前景 PSNR、时间差分误差和测量一致性。百分比使用实际 MSE：

`MSE 降低率 = 100 × (基准平均 MSE − 细化后平均 MSE) / 基准平均 MSE`

PSNR 增益以 dB 表示，不转换成 PSNR 百分比。先对每个测试身份的细化器种子指标取平均（pilot 三个、扩大实验五个），再按身份进行配对 bootstrap；种子间增益波动另行输出，避免把重复身份当成独立测试样本。重建图在指标计算前统一截断到 `[0, 1]`。

效率报告包含完整重建时间、相对粗重建增加的时间、细化模块自身时间、运动估计时间和训练时间。额外时间为“该方案完整时间 − 粗重建完整时间”。CNN 与固定图的完整流程不计入无须调用的运动估计器；运动图包含运动估计。计时从已取得并驻留设备的 bucket 测量开始，不包含光学采集、网络传输与磁盘读写。显存同时记录常驻量和额外峰值；基准脚本一次加载多个模型，绝对峰值不能解释为单独部署某一个模型所需的显存。

## 运行方法

扩大实验的完整步骤见 [扩大实验运行说明](SCALEUP_RUN.md)，服务器环境为 Python 3.11、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128。完整工程包解压后，从项目根目录执行：

```bash
bash scripts/setup_hengyuan.sh
bash scripts/run_hengyuan_scaleup.sh --upload --shutdown-after-upload
```

该入口完成五个种子的训练、最终评估、计时与上传。当前保留测试结果已固定保存；后续新方案先在开发和验证数据上选择，另设确认测试。

### pilot 复现入口

pilot 步骤见 [恒源云运行教程](SERVER_RUN.md)。CUDA 镜像应提供兼容的 `torch` 与 `torchvision`；安装其余依赖时使用 `requirements-runtime.txt`，保留镜像中的 CUDA 配套版本。`requirements.txt` 列出全部直接依赖，笔记本原环境记录在 `environment_notebook.json`。

```bash
python -m pip install -r requirements-runtime.txt
python run_server_experiment.py --stage check --require-cuda
bash scripts/run_hengyuan_mnist.sh
```

启动脚本依次完成资源检查、小规模训练检查、三个种子的正式训练、统一最终测试和效率测量。需要逐步执行时可使用：

```bash
python run_server_experiment.py --stage smoke --require-cuda
python run_server_experiment.py --stage train --require-cuda
python run_server_experiment.py --stage evaluate --require-cuda
python run_server_experiment.py --stage benchmark --require-cuda
python package_gi_project.py --results
```

已完成的正式训练种子可以跳过；新版本对未完成的方案逐轮保存并恢复训练。pilot 输出位于 `runs/mnist_block3/`，包括训练历史、九份细化器权重、逐序列指标、配对统计、结果图、效率表及运行环境。pilot 结果包为 `dist/gi_results_block3.zip`；扩大实验使用独立目录与压缩包。

本地检查使用 `python run_server_experiment.py --stage smoke`；重新制作工程上传包使用 `python package_gi_project.py`。准备检查只读验证身份，不读最终测试块。实际检查记录见 [PREPARATION_CHECKS.json](PREPARATION_CHECKS.json)。服务器 Linux/CUDA 镜像仍需在启动实例后通过同一检查入口确认。

## 已有研究结果

以下为早期 `madygraph_gi_plugin_v2_outputs/benchmark_summary.csv` 中测试块 1、120 条序列、细化器种子偏移 0 的结果。

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
| `server_results/scaleup_v1/` | 已完成扩大实验的结果、核查证据和分析图表 |
| `server_results/block3/gi_results/` | 已完成的 RTX 3090 pilot 结果 |
| `madygraph_gi_plugin_v2_outputs/` | 当前细化路线的已有指标与图表 |
| `madygraph_gi_plugin_seed1_outputs/`、`madygraph_gi_plugin_seed2_outputs/` | 其他细化器种子的已有结果 |
| `madygraph_gi_motion_block2_outputs/` | 另一个保留身份块上的已有结果 |
| `graph_protocol_v2_trained/`、`graph_protocol_v2_outputs/` | 直接重建路线的受控比较 |
| `transformer_vs_joint_repeat_summary.json` | 早期直接替代 Transformer 的多种子比较 |
| `madygraph_gi_plugin_noise_outputs/` | 已有测量噪声实验 |
| `notebook_verification_outputs/` | 算子、闭环与过拟合检查记录 |

历史交接见 [动态 GI 研究交接文档](动态GI动态图研究_完整交接文档_2026-09-29.md)。历史文档中的本机路径和旧运行方法用于追溯，当前运行方法以本 README 为准。当前实现借鉴 MadyGraph 的运动图思路，是逐帧 GI 适配原型，观测模型与原论文的视频快照压缩成像不同。

## 版本与源码来源

GitHub 私有仓库为 `icy768/dynamic-gi-motion-graph`，默认分支为 `main`。`v0.1.0-notebook-baseline` 保存整理前的笔记本基线；`v0.1.1-server-ready` 保存本次目录集成与服务器准备。

`v0.2.0-scaleup-ready` 保存扩大实验的代码、固定数据划分、本地准备检查，以及已完成的 RTX 3090 pilot 指标和图表。MNIST 数据、模型权重及完整上传包仍通过独立工程 ZIP 保存。

`v0.2.1-scaleup-ready` 使用适合恒源云 OSS 的校验压缩包，完整运行入口将上传结果 ZIP 和校验 ZIP，再执行可选关机。

扩大实验采用提交 `391b6c611067a5f9e5e2bbbf408e99289cbf427a` 的代码完成。本次进度归档补入实验结果和分析，不更改用于本轮训练的代码、配置与保留测试划分。

10 个基础模块最初从旧项目工作目录补入，来源 HEAD 为 `130c64e3e35482dbcdc1516f29afe1c25d0d9656`。原始文件哈希与旧仓库跟踪状态见 `source_provenance.json`，该文件记录导入时的来源，不是当前修改后文件的哈希。旧工作目录现已不作为运行依赖。
