# 网卡电磁信号攻击检测

统一入口为 `experiment.py`，依赖当前 `vul-detect` Conda 环境中的 numpy、pandas、scipy、scikit-learn 和 PyTorch，无需安装额外包。原始 CSV 只读。

## 数据检查

```bash
conda run --no-capture-output -n vul-detect python -u experiment.py inspect
```

输入默认是 `/home/PublicData/qc-data/SCA/Data_new`。检查真实 Pico 表头 `时间,通道 B`、单位 `(s),(V)` 和空行；逐文件扫描统计数据行数，检查首尾各 10,000 点的时间间隔。采样率由首尾时间与精确样本数计算，兼容导出时间的小数舍入。完整数据准备阶段还会逐点检查时间轴及样本总数；时间戳缺失、间断或格式错误直接报错。电压中的 NaN/Inf 会导致所在完整窗口被排除，不补值、不删除单点拼接信号。

结果保存在 `outputs/inventory.csv` 和 `inventory.json`，包括每个 session 的路径、类别、字节数、样本数、起止时间、采样率和时长。原始时间可以从负数开始。

实测共 25 个 CSV、116,014,020,437 字节、4,656,149,931 个数据点。采样率约为 500 kHz、796.178 kHz 或 1 MHz；总采集时长约 8,427.168 秒。

| 类别 | CSV 数 | 单 session 时长范围（秒） |
|---|---:|---:|
| Hping3 | 3 | 227.361–500 |
| dirsearch | 2 | 500 |
| gobuster | 3 | 500 |
| nmap port | 4 | 100–500 |
| nmap version | 5 | 15.399–200 |
| normal | 2 | 500 |
| sql | 3 | 58.720–500 |
| xssser | 3 | 139.148–500 |

## 划分与窗口

每个原始 CSV 是一个 session。先按 seed=42 打乱各类 session 顺序，建立 25 轮严格 leave-one-session-out 划分，再提取窗口：

- 各类交替取出 session，形成固定测试顺序；每轮恰好一个完整 CSV 作为测试，25 轮遍历全部文件。
- 每类剩余至少两个 session 时，确定性选一个完整 session 验证，其余训练；只剩一个时全部留给训练。
- 25 个 session 各测试一次。每轮训练包含全部 8 类。由于 normal 和 dirsearch 各只有两次采集，当这两类中的一个 session 被测试时，验证集缺少该 session 所属类别。
- 每轮显式断言 train、val、test 的 session 集合两两不交，且覆盖全部 session；再次检查测试覆盖恰好一次。训练时重复检查同样的断言。

默认窗口长 0.01 秒，不重叠；不足一个窗口的尾部丢弃。按各文件采样率换算窗口点数，取最近整数。对每个窗口使用 `scipy.signal.resample_poly` 抗混叠重采样，统一到 100 kHz、每窗 1,000 点。默认共同带宽约为 0–50 kHz；这是基线预处理选择，并不覆盖原始记录全部频带。可通过 `--target-rate` 调整，所有方法必须使用同一个准备结果。记录原始窗口点数和舍弃点数，可以回溯每窗原始行范围：窗口 i 对应零起始数据行 `[i*N, (i+1)*N)`。

逐 session 保存 `.npy`，准备后的缓存以 mmap 打开，训练启动时将全部有效窗口一次性加载到所选设备。缓存保留候选窗口的原始位置，被排除的槽位标记为 NaN；RF 和全部 CNN 共用有效窗口索引，只读取有效窗口。CNN 在设备上通过各 split 的索引批量取数，训练时不再逐窗口执行 Python 读取或逐批搬运数据；加载到同一设备不改变 session 划分，训练仅使用 train 索引。`candidate_windows` 是检查的候选窗口数，`windows` 是有效窗口数，`invalid_signal_samples` 统计读取范围内的非有限电压点（含尾部），`excluded_windows` 和 `excluded_window_indices` 记录排除的完整窗口数及其原始编号。预测文件继续使用原始窗口编号，可回溯 CSV 数据行。无有效窗口的 session 直接报错。窗口只计算一次，各轮通过 session 清单引用，因此共享缓存不会混合训练和测试数据。`dataset.json` 保存种子、参数、类别映射与所有划分；`session_splits.csv` 逐轮列出 train/val/test 来源、样本数、可用窗口数和实际使用窗口数。原始电压特征处理不拟合全数据统计量。

## 方法、训练与评价

每个窗口长度运行同一组 25 折，两项任务、五种方法，共 250 次模型拟合。每个模型统一评估 1、3、5、10 窗概率聚合，共 1,000 条逐折指标。

| 方法 | 实现 |
|---|---|
| rf | RMS、平方和 energy、std、peak-to-peak、mean、mean absolute、max absolute；100 棵树 |
| rf_frequency | 上述 7 维统计 + 16 维频域特征；相同 RandomForest |
| raw_cnn | 重采样原始时序；3 层 Conv1d + 时间均值池化 + 线性分类 |
| zscore_cnn | 相同 1D-CNN；每窗减均值并除总体标准差 |
| stft_cnn | Hann 窗，FFT=128、hop=32、center=False；log1p 功率谱 + 两层 Conv2d |

逐窗抗混叠 FIR 滤波器按 session 构建一次并复用，系数与 scipy 原默认设置一致，避免长窗口反复构建超长滤波器。

频域特征使用 Welch PSD，Hann 窗、nperseg=min(4096, 窗长点数)、默认半窗重叠、逐段去均值。在 100 kHz 下频段为 0–1、1–5、5–10、10–20、20–35、35–50 kHz；其他采样率按 Nyquist 同比例调整。各频段计算能量（PSD 积分乘窗口秒数）和能量占比，另加 AC 总功率、主频、频谱质心、归一化频谱熵。DC 信息由原有均值等时域特征保留；零 AC 功率的频域特征为零。不用训练外数据拟合变换。

默认 `--sampling balanced`：

- 二分类每 epoch 总抽样数向上取整到 14 的倍数，normal 占 7 份，七类 attack 各 1 份，严格满足 1:1。
- 八分类总数取整到 8 的倍数，各类等额。
- 每类配额再均分给该折训练 session，差值最多为 1；余数由带种子的随机顺序分配。每个 session 内均匀有放回抽样。均衡是分层均衡，不要求不同类别的所有 session 拥有相同权重。
- 每 epoch 抽样量约为原训练有效窗口总数。三个 CNN 共享每 epoch 的索引；两个 RF 共用第一个 epoch 的索引。均衡采样时不再使用逆频率类权重，避免重复校正。
- `training_sampling.csv` 保存每折、任务、epoch、训练 session、原窗口数和抽样配额。RF 只使用 epoch 1；CNN 使用全部 epoch。验证和测试不重采样。
- `--sampling window` 保留按原窗口量训练和原有类权重，供消融比较。

默认 `--augmentation signal` 仅作用于 CNN 训练输入：每窗随机乘 0.8–1.2，加入 ±0.1×该窗标准差的 DC 偏移及标准差为 0.01×该窗标准差的高斯噪声。增强发生在 z-score/STFT 之前；不修改缓存，不增强验证和测试。`--augmentation none` 用于对照。采样和增强使用独立随机数生成器，打开增强不改变训练窗口抽样顺序。

CNN 使用 Adam（lr=0.001），默认 20 epoch；以验证集未加权平均交叉熵选择 checkpoint。默认 `--batch-size 1024 --max-batch-points 1024000`，实际 batch 为两者和窗口点数共同决定的上限。在 100 kHz 的 0.5/1/2/5 秒窗口下实际 batch 分别为 20/10/5/2。保持完整输入长度，不为节省显存再截短信号。配置记录实际 batch。不同窗口长度的窗口数和优化器步数不同，应结合训练历史解释结果。

测试聚合采用 stride=1 的连续完整窗口概率均值，然后 argmax；二分类等价于 attack 概率超过 0.5 判为 attack（相等时 normal）。只在同一测试 session 内聚合，原窗口编号必须连续，遇到被剔除的窗口即断开。不补齐开头，也不跨 session 或折拼接。预测 CSV 保存起止原窗口编号、真实/预测类别和每类概率。

四种聚合分别报告 Accuracy、Macro Precision、Macro Recall、Macro F1；八分类保存每折及 pooled confusion matrix。二分类额外报告 AUROC、AUPRC（average precision，attack 为正类）、Balanced Accuracy、normal recall、attack recall。每折只有一个测试 session，因此只有一个真实类别：该折 AUROC/AUPRC/Balanced Accuracy 和缺失类 recall 记为 JSON null；在全部留出预测 pooled 后计算这些指标。固定 2/8 类 macro 指标继续使用原口径，缺失类置零。

`summary.json` 保存逐折等权均值、标准差及每项指标有效折数，并保存所有留出预测的 pooled 指标。pooled 使用不同折模型的预测，按输出窗口计权；长 session 仍对 pooled 指标贡献较多。指标中另存原测试窗口数、聚合后输出数和覆盖率。部分短 session 或缺失值造成的短连续片段无法形成 5/10 窗聚合，此时保留该折、输出空预测表和 null 指标，并标明原因；八分类混淆矩阵为全零计数。各聚合的覆盖范围不同，不能把指标差异全归因于聚合本身。

## 窗口与结果目录

准备与运行均通过 `--window-seconds` 选择窗口。0.5/1/2/5 秒缓存各自保存到 `outputs/window_0.5s/cache/` 等目录；共同复用 `outputs/inventory.json`。每次先从相同 seed=42 的 session 清单生成划分，再切窗。运行时校验保存的划分等于重新生成的划分，并逐折检查集合不相交。已有 0.01 秒缓存保留在 `outputs/cache/`。

结果写入各窗口目录的 `results_balanced_none/`、`results_balanced_signal/` 或对应的 `window` 采样对照目录；原 `outputs/results/` 不覆盖。每个目录保存 config、抽样配额、指标、预测、checkpoint、训练历史及混淆矩阵。重复相同配置会覆盖该配置目录，不自动续训。

NaN/Inf 仍导致整个候选窗口被剔除，不插值、不拼接；长窗口可能损失更多可用时长，以各自 `session_splits.csv` 为准。每个 session 必须至少有一个有效窗口，否则准备直接报错。

## 最小检查

```bash
conda run --no-capture-output -n vul-detect python -m unittest -v test_experiment.py
conda run --no-capture-output -n vul-detect python -u experiment.py prepare --window-seconds 0.5 --smoke --smoke-windows 12
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n vul-detect python -u experiment.py run --window-seconds 0.5 --smoke --device cuda
```

smoke 只读取指定数量的原始前缀窗口，RF 10 棵树、CNN 1 epoch，但保留全部 25 折及五种方法。缓存与结果分别使用 `smoke_cache/` 和 `smoke_results_*`。如果较长窗口的前缀全被缺失值污染，需要显式增大 `--smoke-windows`；全量准备不会受这个参数限制。smoke 指标不能作为性能结论。

本次验证：8 项自动测试通过，包含四种窗口长度的合成数据准备检查。真实数据使用 0.5 秒、每 session 前 12 个候选窗口，共 300 个候选、287 个有效窗口；CUDA 上完成 25 折 × 2 任务 × 5 方法的 250 次 smoke 拟合和 1,000 条聚合评价。已回查全部预测概率平均、原窗口连续性、混淆矩阵计数和 50 组训练抽样配额，划分与原 `outputs/cache/dataset.json` 完全一致。1/3/5/10 窗聚合分别输出 287/215/156/45 个窗口，覆盖 25/25/24/15 折。结果保存在 `outputs/window_0.5s/smoke_results_balanced_signal/`；未执行正式全量训练。

## 正式全量命令

已有 `outputs/inventory.json`，无需重新 inspect。以下执行四种窗口长度，并分别训练无增强与有增强对照；均保持类别/session 均衡和同样的 25 折。GPU 编号可按空闲情况调整。

```bash
set -e
for seconds in 0.5 1 2 5; do
  conda run --no-capture-output -n vul-detect python -u experiment.py prepare \
    --window-seconds "$seconds" --target-rate 100000 --seed 42
  for augmentation in none signal; do
    CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n vul-detect python -u experiment.py run \
      --window-seconds "$seconds" --device cuda --epochs 20 \
      --sampling balanced --augmentation "$augmentation"
  done
done
```

每种窗口长度只准备一次，两个增强设置共用其缓存；RF 不受增强设置影响，两个目录中的 RF 是同一对照的重复拟合。每次 prepare 会顺序扫描原始约 109 GiB CSV，因此四种窗口共扫描四次。CNN 使用 CUDA；数据准备、特征计算与 sklearn RF 使用 CPU。全量 GPU 信号缓存大小随有效总时长变化，加载和特征处理按采样点数限制分块大小。

若需单独测量均衡采样的作用，可对已准备的相同窗口再运行 `run --window-seconds 0.5 --sampling window --augmentation none --device cuda`，并与同窗口的 `balanced_none` 比较；其他长度同理。
