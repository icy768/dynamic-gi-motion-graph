# 恒源云运行教程

本次使用已有 MNIST 合成运动序列，检验运动图相对 CNN、固定图及粗重建的增益是否稳定，并测量增加的推理时间。数据、冻结权重和代码已装入同一个工程包，服务器不需要重新下载数据集，也不需要连接旧 D 盘项目。

## 1. 上传工程包

本地文件为项目根目录下的 `dist/gi_project_server.zip`。可以先上传到恒源云“个人数据”，再启动 GPU 实例，上传方法见平台[快速开始](https://gpushare.com/docs/getting-started/quickstart/)。使用平台提供的 Windows OSS 工具时，在压缩包所在目录执行：

```powershell
oss login
oss cp gi_project_server.zip oss://
```

账号和密码在登录工具的提示中输入。也可以使用平台支持的客户端上传。此次不必在服务器上克隆 GitHub；完整工程包已经包含代码版本标识和运行资源。

## 2. 创建实例与进入项目

这次规模采用单卡 24 GB 显存即可，优先从可用的 RTX 3090 实例开始。4090 也能使用，但实际速度需以本项目的计时结果为准。选择带 NVIDIA 驱动、CUDA 和兼容 PyTorch/torchvision 的 Linux 镜像，Python 建议 3.11；不需要多卡。

创建后，从平台提供的 SSH 登录指令或 JupyterLab 打开终端。把工程包下载到实例存储并解压：

```bash
cd /hy-tmp
oss login
oss cp oss://gi_project_server.zip .
python -m zipfile -e gi_project_server.zip .
cd /hy-tmp/gi_project
```

后续命令都在 `/hy-tmp/gi_project` 中执行。若已解压过，继续使用原目录；不要在训练进行中重新解压覆盖代码或权重。

## 3. 检查环境

```bash
python --version
nvidia-smi
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pip install -r requirements-runtime.txt
python run_server_experiment.py --stage check --require-cuda
```

最后应看到 `preflight passed`。该步骤核对四个 MNIST 文件和两份冻结权重的 SHA256、检查 Python 语法、运行三个细化器的前向计算，并验证 bucket 测量到 DGI 的接口。

如果 CUDA 不可用或 torch/torchvision 导入失败，先确认镜像和当前 Python 环境。选取兼容 CUDA 的 PyTorch 镜像后重新运行检查。`requirements-runtime.txt` 不会主动重装镜像里的 torch/torchvision。

## 4. 启动实验

推荐使用 tmux，使 SSH 断开后训练继续运行。操作方法见平台 [Tmux 文档](https://gpushare.com/docs/best_practices/tmux/)。

```bash
tmux new -s gi
bash scripts/run_hengyuan_mnist.sh
```

按 `Ctrl+B`，松开后再按 `D` 可离开会话；重新连接使用：

```bash
tmux attach -t gi
```

启动脚本按以下顺序执行：

1. 资源、环境和前向接口检查。
2. 16 条训练序列、8 条验证序列的小规模训练与指标检查。
3. 正式训练 CNN、固定图、运动图，每种运行三个种子、每次 5 轮。
4. 在九份细化器权重全部确定后，统一评估保留的最终测试块 3。
5. 测量完整重建、各阶段时间和显存，输出 CSV 与结果图。

日志保存在 `runs/server_日期_时间.log`，正式结果保存在 `runs/mnist_block3/`。已完成的种子再次启动时会跳过；中断的种子从该种子训练开始。最终评估完成后再次启动会复用其结果；配置、代码或权重改变时，检查会拒绝混用不同实验版本。

如需分步执行，依次运行：

```bash
python run_server_experiment.py --stage smoke --require-cuda
python run_server_experiment.py --stage train --require-cuda
python run_server_experiment.py --stage evaluate --require-cuda
python run_server_experiment.py --stage benchmark --require-cuda
```

执行 `evaluate` 前应完成所有正式训练，不根据最终测试结果重新选参数或挑权重。

## 5. 查看结果

| 文件 | 内容 |
|---|---|
| `runs/mnist_block3/seed0/`、`seed1/`、`seed2/` | 各细化器训练历史、权重和身份清单 |
| `final/benchmark_summary.csv` | 两个采样率和不同运动速度的质量指标 |
| `final/paired_gains.csv` | 配对 PSNR 增益、95% 区间、MSE 降低百分比等 |
| `final/seed_gains.csv` | 三个细化器种子的增益波动 |
| `final/per_sequence_results.csv` | 每条序列、每个方法、每个种子的原始指标 |
| `final/quality_comparison.png` | PSNR 与 SSIM 对比 |
| `final/paired_gains.png` | 运动图增益及配对置信区间 |
| `final/measurement_consistency.png` | 测量一致性 |
| `final/example_reconstructions.png` | 按固定规则选取的重建图像 |
| `final/results.json` | 实验配置、版本、环境和预设判据结果 |
| `efficiency.csv`、`efficiency.png`、`efficiency.json` | 完整时间、额外时间、阶段时间与显存 |

表中 `final/...` 均位于 `runs/mnist_block3/` 内。比较细化开销时使用完整时间减去粗重建完整时间，也保留模块本身的计时；运动图完整流程包括运动估计。

两个采样率下，运动图相对 CNN 的平均 PSNR 增益达到 0.5 dB、95% 区间下限大于 0、平均 SSIM 不下降，才记为满足本次预设判据。三种细化器共享冻结的粗重建器与运动估计器，因此结果首先回答当前合成数据下细化模块的价值。

## 6. 保存结果与关机

```bash
python package_gi_project.py --results
oss cp dist/gi_results_block3.zip oss://
oss ls -s -d oss://
```

确认个人数据中出现结果压缩包，再下载到本机保存，并在控制台关闭实例。平台说明 `/hy-tmp` 不随实例迁移，关机超过 24 小时可能删除其中的数据，因此需先保存实验结果，详见[数据存储文档](https://gpushare.com/docs/data/storage/)。

结果包包括九份细化器权重、训练历史、最终统计、图表、计时记录和实验配置。冻结模型与数据已在最初工程包中归档。

## 时间预估

现有笔记本同规模单个种子、三种细化器各 5 轮的历史运行约 126.5 秒。服务器正式计算先预留 10–30 分钟，第一次租用另预留环境检查、上传和结果回收的时间，总体可按 1–2 小时安排。这个范围用于安排工作，尚无恒源云实测；启动日志会给出实际耗时。
