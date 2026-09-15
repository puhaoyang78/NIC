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

第一阶段只评估单窗口（`--aggregations 1`），比较 0.5/1/2/5 秒窗口，确定最佳窗口长度。第二阶段只在选定窗口长度上增加连续窗口概率平均，例如 `--aggregations 1,2,5,10`。结果同时记录真实观察时长 `window_seconds × aggregation`。

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

## 第二阶段：连续窗口聚合

假设第一阶段最佳窗口为 1 秒，则只需要再运行：

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n vul-detect python -u experiment.py run \
  --window-seconds 1 \
  --aggregations 1,2,5,10 \
  --sampling balanced \
  --augmentation signal \
  --epochs 20 \
  --device cuda
```

如果最佳窗口不是 1 秒，将 `--window-seconds` 替换成实际最佳值。不要在四种窗口上同时穷举所有聚合长度。
