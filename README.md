# 网卡电磁信号攻击检测

统一入口为 `experiment.py`。数据默认位于 `/home/PublicData/qc-data/SCA/Data_new`，包含 `Hping3`、`dirsearch`、`gobuster`、`nmap port`、`nmap version`、`normal`、`sql`、`xssser` 八类 Pico CSV。

## 数据与划分

每个 CSV 视为一个独立 session。实验先按完整 session 建立 25 轮 leave-one-session-out 划分，再在各 session 内切不重叠窗口。任何原始 CSV 都不会跨 train/val/test；每个 session 恰好完整作为测试集一次。

攻击采集过程中，攻击在 Pico 开始采集前已经启动，并持续到该 CSV 采集结束，因此每个攻击 CSV 的完整有效区间使用同一攻击标签。

当前正式窗口只比较：`0.5 s`、`1 s`、`2 s`、`5 s`。旧的 10 ms 窗口不再作为正式主实验。所有窗口统一重采样到 100 kHz；对异常 Pico 采样率，polyphase 重采样比率限制为有界有理近似，避免长窗口产生异常巨大的 FIR 滤波器。

## 训练与评价

默认训练使用类别与 session 均衡采样：二分类中 normal 与 attack 为 1:1，七种攻击等额；八分类各类等额，同类训练 session 尽量等权。CNN 可加入随机幅值缩放、直流偏移和少量高斯噪声。

模型包括：

- `rf`：基础时域统计特征 + RandomForest；
- `rf_frequency`：时域统计 + 频段能量/比例、主频、频谱质心、频谱熵等 + RandomForest；
- `raw_cnn`：原始时序 1D-CNN；
- `zscore_cnn`：逐窗 z-score 后的 1D-CNN；
- `stft_cnn`：STFT 时频表示 + CNN。

CNN 固定训练指定 epoch 数；validation 只作为诊断，不再根据 validation 选择最佳 epoch，因此 validation 中缺少 normal/dirsearch 类别时不会影响 checkpoint 选择。

第一阶段只评估单窗口（`--aggregations 1`），比较 0.5/1/2/5 秒窗口，确定最佳窗口长度。第二阶段只在选定窗口长度上增加连续窗口概率平均，例如 `--aggregations 1,3,5,10`。结果同时记录真实观察时长 `window_seconds × aggregation`。

每折测试集只有一个完整 session，因此逐折 Macro F1 只作诊断。正式主结果使用 `summary.json` 中的 `pooled`：把 25 个严格留出的测试 session 的预测汇总后统一计算 Accuracy、Macro Precision、Macro Recall、Macro F1；二分类额外计算 AUROC、AUPRC、Balanced Accuracy、normal recall、attack recall。`session_accuracy` 给出每个测试 session 等权的准确率均值与标准差。

## 检查

```bash
conda run --no-capture-output -n vul-detect python -m unittest -v test_experiment.py
```

可先对 1 秒窗口做 smoke test：

```bash
conda run --no-capture-output -n vul-detect python -u experiment.py prepare \
  --window-seconds 1 --smoke --smoke-windows 12

CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n vul-detect python -u experiment.py run \
  --window-seconds 1 --aggregations 1 --smoke --device cuda
```

本次 RQ 扩展验证：11 项自动测试通过；使用现有 0.5 秒、287 个有效前缀窗口完成 CUDA smoke，行为任务为 25 折 × 5 模型（125 次拟合），unseen 为 14 折 × 3 模型（42 次拟合），均仅 1 epoch / 10 棵树。离线聚合分别完成 20/12 组结果，回查了实际保存的划分、抽样配额及共同结束位置窗口数。与原版采样器比较的 100 个既有 LOSO 案例索引完全一致。测试中同时覆盖了异常概率拒绝、历史浮点舍入、空聚合、消融设置不匹配拒绝。尚未运行新增 RQ 的正式全量训练。

## 第一阶段：窗口长度比较

先准备四种窗口：

```bash
for w in 0.5 1 2 5; do
  conda run --no-capture-output -n vul-detect python -u experiment.py prepare --window-seconds $w
done
```

然后分别运行单窗口正式实验：

```bash
for w in 0.5 1 2 5; do
  CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n vul-detect python -u experiment.py run \
    --window-seconds $w \
    --aggregations 1 \
    --sampling balanced \
    --augmentation signal \
    --epochs 20 \
    --device cuda
done
```

比较四个窗口目录中 `summary.json -> pooled` 的结果，选定最佳窗口长度。

## RQ2：行为粒度与工具粒度

`--tasks` 可选 `binary,multiclass,behavior`，默认仍为 binary,multiclass。工具级任务保持原 8 类；行为级任务按已确认的映射独立训练 5 类模型：

| 行为标签 | 原始目录 |
|---|---|
| normal | normal |
| Hping3 | Hping3 |
| directory_enumeration | dirsearch、gobuster |
| network_reconnaissance | nmap port、nmap version |
| web_injection | sql、xssser |

这是依据采集工具定义的粗粒度行为标签，不是新增的逐流量行为标注。行为任务继续使用原 25 折，不将八分类预测简单合并作为独立训练结果。采样先平衡 5 个行为组，再平衡组内工具和工具内 session。

现有 2 秒缓存可直接使用，无需重新 prepare：

```bash
export CUDA_VISIBLE_DEVICES=1
conda run --no-capture-output -n vul-detect python -u experiment.py run \
  --window-seconds 2 --tasks behavior --models all --aggregations 1 \
  --sampling balanced --augmentation signal --epochs 20 --device cuda
```

新增任务目录后缀为 `_p-loso_t-behavior`，不会覆盖原二分类/八分类结果。配置保存行为映射、实际 evaluation_folds；结果中的 `session_splits.csv` 和 `training_sampling.csv` 可核查来源及配额。

## RQ3：表示、观察时长与增强消融

已有四种时长、五种表示、增强开启的正式结果可直接保留。以下只补三个 CNN 的无增强对照，两个 RF 不受信号增强影响，不必重复训练：

```bash
for w in 0.5 1 2 5; do
  CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n vul-detect python -u experiment.py run \
    --window-seconds "$w" --tasks binary,multiclass \
    --models raw_cnn,zscore_cnn,stft_cnn --aggregations 1 \
    --sampling balanced --augmentation none --epochs 20 --device cuda
done

conda run --no-capture-output -n vul-detect python experiment.py compare --results \
  outputs/window_*s/results_balanced_signal_m-rf-rf_frequency-raw_cnn-zscore_cnn-stft_cnn_a-1 \
  outputs/window_*s/results_balanced_none_m-raw_cnn-zscore_cnn-stft_cnn_a-1 \
  --destination outputs/rq_comparison
```

`comparison.csv` 汇总各任务/窗口/表示的 pooled 指标、有效 batch 和测试窗口数。`augmentation_effects.json` 只对同窗长、任务、模型和聚合长度的 none/signal 配对，检查划分、有效窗口、epoch、学习率、采样方式和环境等设置一致，输出 signal−none 的指标差值（0.01 表示 1 个百分点）。不把不同窗长或 batch 的差异当成增强收益。跨窗长因 NaN 整窗剔除造成覆盖差异，需结合缓存中的有效窗口数解释。

## RQ2 / RQ4：混淆分析与时间聚合，无需重训

`analyze` 直接读取已保存的单窗概率。下面同时生成已有八分类的细粒度分析和二分类/八分类的时间聚合对照：

```bash
known_results=outputs/window_2s/results_balanced_signal_m-rf-rf_frequency-raw_cnn-zscore_cnn-stft_cnn_a-1
conda run --no-capture-output -n vul-detect python experiment.py analyze \
  --results "$known_results" --aggregations 1,3,5,10

behavior_results=outputs/window_2s/results_balanced_signal_m-rf-rf_frequency-raw_cnn-zscore_cnn-stft_cnn_a-1_p-loso_t-behavior
conda run --no-capture-output -n vul-detect python experiment.py analyze \
  --results "$behavior_results" --aggregations 1,3,5,10
```

分析入口拒绝缺失单窗预测或未完成的源运行，逐折校验 session、真实标签、概率以及原有效窗口编号完整一致。它不会修改 checkpoint 或源结果，在源目录下写入 `analysis_a-1-3-5-10/`：

- `summary.json`：全可用窗口指标、相同结束窗口位置上的 `common_endpoint_metrics`、覆盖窗口数和有效折数；2 秒窗口的观察时长为 2/6/10/20 秒。
- `folds.json`、`sessions.csv`：逐折、逐 session 的指标与覆盖率，支持严格跨 session 的稳定性分析。
- `*_confusion.csv`、`*_confusion_normalized.csv`、`*_per_class.csv`：计数矩阵、按真实类别归一化矩阵及逐类 Precision/Recall/F1/support。
- `*_dirsearch_gobuster.json`：两个工具的正确数、互相混淆数、误判到其他类的数量和完整去向；保留八分类中的其他错误，不通过只保留两个预测类别人为提高成绩。
- `*_predictions.csv`：聚合起止原窗口、是否属于共同结束位置、标签及概率。

聚合采用 stride=1 的完整连续窗口概率均值，严格禁止跨 session、跨折或跨缺失窗口。`common_endpoint_metrics` 对所有聚合档位使用同一组结束位置，避免仅因排除难窗口而声称聚合有效；同时保留各档全部可用位置的结果。重叠聚合的输出不是独立 session。连续窗口不足时输出空预测和 null 指标，混淆矩阵为零计数，不补齐或拼接。没有增加测试集阈值选择，二分类仍使用固定 0.5。

## RQ4：未见攻击工具的二分类检测

`--protocol unseen --tasks binary` 是单独的 leave-one-attack-type-out 协议，不替换原 25 折 LOSO：

1. 每次将某一种攻击工具的所有 CSV 留给测试，从 train 和 val 同时完全排除。
2. 对两个 normal session 分别留出一个作测试负类，另一个用于训练。
3. 其余 6 种已知攻击按完整 session 划分 train/val，训练中 normal:attack=1:1、六种攻击等额、类内 session 均衡。
4. 共 7 种工具 × 2 个 normal 留出轮次 = 14 折，各模型使用完全相同的划分。每折显式检查三集合互斥、全数据覆盖和未见攻击不进入 train/val。

由于 normal 仅两个 session，此协议的验证集没有 normal；延续固定 20 epoch 的规则，不根据验证或测试结果选择 checkpoint、阈值。

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n vul-detect python -u experiment.py run \
  --window-seconds 2 --protocol unseen --tasks binary --models all \
  --aggregations 1 --sampling balanced --augmentation signal --epochs 20 --device cuda

unseen_results=outputs/window_2s/results_balanced_signal_m-rf-rf_frequency-raw_cnn-zscore_cnn-stft_cnn_a-1_p-unseen_t-binary
conda run --no-capture-output -n vul-detect python experiment.py analyze \
  --results "$unseen_results" --aggregations 1,3,5,10
```

每折包含 normal 和未见攻击，能在同一模型下计算 AUROC/AUPRC、Balanced Accuracy、normal recall 和 unseen attack recall（字段名仍为 `attack_recall`）。主结果为 `summary.json -> primary_metrics` 的等折均值和 `held_out_attack_metrics` 的逐工具两轮均值。分析输出对应 `fold_metrics` 与 `held_out_attack_metrics`；`folds.json` 保留每一轮。

同一未知攻击 session 会在两种 normal 留出配置下测试，normal 也会在不同攻击留出实验中重复出现，因此不把 14 折预测拼接成独立样本总体计算 pooled 分数。配置与逐行划分表保存 held_out_attack、normal_rotation 和测试 normal 来源。

这里的 unseen 指该轮模型训练/验证未见该攻击工具；不是新漏洞识别或从未被研究者观察过的数据。当前 2 秒配置来自已完成的探索性 LOSO 比较，新增实验固定这个配置，不再用未知攻击测试结果调参。
