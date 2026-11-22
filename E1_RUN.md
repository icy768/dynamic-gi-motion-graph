# 实验 E1：在 Transformer 之上凑测量还能提升多少

E1 回答一个问题：Transformer 粗重建本来就会参考相邻帧，在它之上再用运动把 8 帧的测量凑在一起，还能多出多少 dB？背景和路线见 [2026-11-19 交接文档](动态GI研究_交接文档_2026-11-19.md) 第 4 节，可行性实验见 [motion_pooling/](motion_pooling/运动补偿凑测量_可行性实验_2026-11-18.md)。

## 和扩大实验相比改了什么

| 项目 | 扩大实验 scaleup_v1 | E1 |
|---|---|---|
| 散斑 | 8 帧共用一套 | **每帧一套**（真实 DMD 的做法） |
| 粗 Transformer | 冻结的旧权重（800 条、10 轮训练） | 在每帧散斑下**重训**：4,000 条、20 轮 |
| 运动估计 MotionNet | 冻结的旧权重 | 旧权重与重训权重在验证集上比，误差小的留下 |
| 生成数据的插值 | 双线性 | **双三次**，避免和物理层用同一个双线性模型 |
| 验证 / 测试身份 | 600 / 600 | 全新的 600 / 600，排除了以前所有用过的 2,552 个身份 |

## 五组对照

同一套观测、同一批测试身份：

1. **coarse**：粗 Transformer
2. **cnn**：粗 Transformer + CNN 插件（扩大实验同款，5 个种子）
3. **phys_est**：粗 Transformer + 物理层，运动靠估计
4. **phys_true**：粗 Transformer + 物理层，用真实运动（上限）
5. **phys_cnn**：粗 Transformer + 物理层 + CNN 插件（5 个种子）

另有一个诊断组 **phys_own**：物理层只用本帧自己的测量，用来区分"凑多帧"和"只是对齐本帧测量"各贡献多少。

物理层：从粗图出发，对 `½Σ_k‖A_k S(p_k−p_t) x − y_k‖² + μ/2‖x − x_co‖² (+ TV)` 做带 [0,1] 约束的 FISTA，梯度由 autograd 给出精确伴随；0/1 散斑的整体亮度分量拆开并降权（同 `mcgi.centred_system`）。运动：MotionNet 给初值 → 在测量域里对每对相邻帧做网格搜索配准 → 与物理层交替，按参考画布逐帧网格搜索细化（同 `mcgi.refine_against_canvas`，GPU 向量化）。μ 和 TV 权重只在 120 条验证序列上、按测量数分别选。

## 事先写定的判据（`configs/e1_motion_pooling.json`）

- **通过**：128 和 256 两种测量数下，第 3 组相对第 2 组的配对 PSNR 增益 ≥ 1 dB，95% bootstrap 区间下限 > 0，平均 SSIM 不下降，测量残差下降。
- **止损**：两种测量数下，第 3 组相对第 1 组的增益都 < 0.5 dB，**并且**最终运动误差 < 0.2 px，说明 Transformer 已用掉大部分多帧信息，C 路线停止。
- 同时报告速度分档、每个种子的增益、每轮运动误差和耗时。

## 上传与运行

本地打包（只含代码、数据和两份旧权重）：

```powershell
D:\Anaconda\envs\pytorch_env\python.exe package_gi_project.py --runtime-only --output dist/gi_project_e1_server.zip
& "D:\download\oss.exe" cp "C:\Users\lingye\Desktop\文献\文献及笔记\dist\gi_project_e1_server.zip" oss://
```

服务器（单卡 RTX 3090，与扩大实验相同的镜像）：

```bash
cd /hy-tmp
oss login
oss cp oss://gi_project_e1_server.zip .
python -m zipfile -e gi_project_e1_server.zip .     # 若已有旧的 gi_project，先改名保存
cd /hy-tmp/gi_project
bash scripts/setup_hengyuan_e1.sh                    # 环境 + scipy + 单元检查
tmux new -s gi_e1
bash scripts/run_hengyuan_e1.sh --upload             # 或加 --shutdown-after-upload
```

脚本依次执行：冒烟检查（开发身份，几十秒）→ 全部阶段 → 结果打包 → 上传。每个阶段都能断点续跑：中断后在同一目录重跑同一命令即可。日志和结果在 `runs/e1_motion_pooling_v1/`。

各阶段也可单独运行：`python e1_pipeline.py --stage <check|data|coarse|motion|features|cnn|tune|physcnn|evaluate|package>`。

## 纪律

- 最终测试序列只在 `evaluate` 阶段生成；`results.json` 写出后不会重跑。
- 改代码、数据或参数需要新的实验目录和新的测试身份块，不能再用这次的测试结果挑参数。
- 本地排练用 `--dev`（`runs/e1_dev/`），只用开发身份（官方训练集里从未参与训练的 600 个），不碰验证 / 测试块。

## 结果

`dist/gi_results_e1_motion_pooling_v1.zip`（不含可再生的序列和特征张量）与校验包 `..._checksums.zip`。核心是 `evaluate/results.json`：`verdict`（判据）、`summary`、`paired`（各组配对增益、区间、按速度分档、每个种子）、`motion_epe_by_round_px`、`timing_seconds`。
