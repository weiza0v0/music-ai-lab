# 数据卡：FMA-small（主实验数据集）

> 本文所有数字来自本机实际运行，不含估计值。GTZAN 的 MVP 结果见
> [`data_card.md`](data_card.md)；两者差异见第 6 节。
> 生成脚本：`src/data/prepare_fma.py`、`scripts/extract_features.py`。
> 本次数据准备未改动这两个脚本（`prepare_fma.py` 的 artist-disjoint 划分逻辑
> 由团队在本轮任务中更新，非本数据卡作者修改）。

---

## 1. 来源与获取

- 数据集：**FMA-small**（Free Music Archive 的 8,000 首子集），官方仓库 <https://github.com/mdeff/fma>，
  音频镜像 `https://os.unil.cloud.switch.ch/fma/`。
- 下载脚本：`scripts/download_fma.sh`（`curl -L --retry 5 -C -`，可断点续传）。
- 两个压缩包（保存在 `data/fma/raw/`，不提交仓库）：

| 文件 | 字节数 | 说明 |
|---|---|---|
| `fma_metadata.zip` | 358,412,441（342 MiB） | 106,574 首曲目元数据、流派、artist |
| `fma_small.zip` | 7,323 MiB（7.2 GiB） | 8,000 首 MP3，解压后 7.43 GiB |

实测下载耗时：音频包 **24 min 41 s**，平均约 **5.0 MB/s**（日志 `/tmp/fma_download.log`，
结尾 `DOWNLOAD_DONE`，全程无重试）。

### 1.1 压缩包完整性校验（先校验再解压）

```bash
PY=$CONDA_PREFIX/bin/python
$PY -c "import zipfile; z=zipfile.ZipFile('data/fma/raw/fma_small.zip'); print(len(z.namelist()), z.testzip())"
```

| 压缩包 | 条目数 | `testzip()` | 耗时 |
|---|---|---|---|
| `fma_metadata.zip` | 12 | `None`（全通过） | 36.6 s |
| `fma_small.zip` | 8,002 | `None`（全通过） | 325.4 s |

8002 个条目 = 8,000 个 MP3 + 目录项；CRC 全部通过，说明**下载与落盘没有损坏**。

解压后再做一次字节级复核：用归档自带的 `data/fma/fma_small/checksums`
（上游发布的 8,000 条 SHA1 清单）逐个比对——

```bash
cd data/fma/fma_small && sha1sum -c checksums | grep -v ': OK$'   # 无输出 = 全部通过
```

结果：**8000/8000 `OK`，0 个不匹配**（耗时 1 min 47 s）。即解压出的音频与上游字节完全一致，
后续发现的所有损坏都能确定为**上游数据本身的问题**。

### 1.2 一个必须记录的坑：元数据压缩包的顶层前缀

`fma_metadata.zip` 内的路径带 `fma_metadata/` 前缀，而 `prepare_fma.py` 读取的是
`data/fma/metadata/tracks.csv`。直接 `extractall` 会解到 `data/fma/fma_metadata/`，
随后 `load_tracks()` 会因找不到 `data/fma/metadata/tracks.csv` 而失败。

处理方式（**没有修改 `src/`**）：按去前缀的方式把所需文件解到脚本期望的位置。

```bash
$PY - <<'EOF'
import zipfile, pathlib
dst = pathlib.Path("data/fma/metadata"); dst.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile("data/fma/raw/fma_metadata.zip") as z:
    for i in z.infolist():
        rel = i.filename[len("fma_metadata/"):]
        if rel in ("tracks.csv", "genres.csv"):
            (dst / rel).write_bytes(z.read(i))
EOF
# tracks.csv 260.4 MB, genres.csv 3.9 KB
```

之后 `--extract` 检测到 `metadata/tracks.csv` 已存在，只解压音频，行为正确。
（若换机器复现，要么先跑上面这段，要么把脚本里的 `META` 改成 `FMA/"fma_metadata"`。）

### 1.3 解压与审计耗时（实测）

| 步骤 | 命令 | 耗时 |
|---|---|---|
| 解压 8,000 个 MP3（7.43 GiB → `data/fma/fma_small/`） | `--extract` | 约 7 min（14:19:06 → 14:26:31，由目录 mtime 推算，脚本无时间戳） |
| 审计 + 划分 | 同上（`--audit --split`） | 2 min 25 s（14:26:31 → 14:28:56） |

首次运行曾因 pandas 3.0 下 `itertuples()` 的属性命名差异报
`AttributeError: 'Pandas' object has no attribute 'path'`（日志 `/tmp/fma_prep.log`），
修复后重跑成功；**以 `data/splits/fma/*` 与 `stats.json` 的当前内容为准**。

---

## 2. 规模与清洗

- 元数据共 **106,574** 首；其中 `set/subset == "small"` 的 **8,000** 首才是本次有音频的部分，
  8 个顶层流派各 **1,000** 首（`audit()` 已按该 subset 过滤）。
- 审计结果：**7,994 首可解码（99.93%）**，8 个流派。

| 流派 | 元数据 | 可解码 |
|---|---|---|
| Electronic | 1000 | 999 |
| Experimental | 1000 | 999 |
| Folk | 1000 | 1000 |
| Hip-Hop | 1000 | 997 |
| Instrumental | 1000 | 1000 |
| International | 1000 | 1000 |
| Pop | 1000 | 1000 |
| Rock | 1000 | 999 |
| **合计** | **8000** | **7994** |

### 2.1 6 个失败曲目（全部为上游归档自带损坏）

| track_id | 文件大小 | 现象 | 判定 |
|---|---|---|---|
| `098565` | 300,753 B | `sf.info` 报时长 1.634 s；解码报 `dequantization failed` | 截断 MP3 |
| `098567` | 141,545 B | 报 0.537 s；解码报 `Giving up resync` | 截断 MP3 |
| `098569` | 236,749 B | 报 1.555 s；解码失败 | 截断 MP3 |
| `099134` | 1,298 B | `sf.info` 无法打开（"not a regular file"） | 空壳文件 |
| `108925` | 1,371 B | 无法打开 | 空壳文件 |
| `133297` | 1,752 B | 无法打开 | 空壳文件 |

关键证据：这 6 个条目在 **zip 里的 `file_size` 就是 1,298–300,753 B**，且 `testzip()` 的
CRC 全部通过；解压后的 SHA1 又与上游 `checksums` 清单**逐一相符**。即压缩包内保存的字节
与官方完全一致，**损坏来自 FMA 上游数据，不是下载或解压造成**。
无法通过重新下载修复，按不可解码剔除。

### 2.2 重复检查（与 GTZAN 的关键差异）

- 先按文件大小分组，对 4,300 个尺寸中同尺寸的 5,044 个候选文件逐个算 MD5：
  **0 组字节级完全重复**（GTZAN 当时有 28 首、剔除 14 首）。
- 元数据层面有 **61 组**「同 artist + 同 title」的多轨（涉及 135 轨），例如同名现场版 /
  多个 `Title Unknown`，MD5 并不相同，属于正常的不同录音，**保留**。

---

## 3. 采样率、声道与时长（对全部 7,994 个文件独立复核）

| 属性 | 实测 |
|---|---|
| 采样率 | 44,100 Hz × 7,569；48,000 Hz × 411；22,050 Hz × 14（与 `splits.csv` 的 `sr` **零不一致**） |
| 声道 | 立体声 7,909；单声道 85 |
| 时长 | min 29.965 s / max 30.003 s / mean 29.9882 s / std 0.0126 s |

下游统一重采样到 **22,050 Hz 单声道**并取**中心 5 秒**（`src/data/audio_features.py`），
与 GTZAN 协议一致，因此 44.1 kHz/48 kHz 混装不影响模型输入。

---

## 4. 划分协议：为什么必须按 artist 划分

### 4.1 泄漏诊断

FMA 的目录里同一 artist 往往有多首曲目：**2,304 个 artist**，每 artist 曲目中位数 1、
均值 3.47、**最大 208 首**（Blue Dot Sessions）；51.8% 的 artist 只有 1 首，
而 **37.6% 的曲目来自 ≥10 首的 artist**。

在这种情况下按曲目随机划分，同一 artist 的作品会同时落进训练集和测试集，模型可以
靠「认出这位歌手/这支乐队」而不是「听懂流派」拿分。实测：

> 曲目级 70/15/15 划分下，**971 / 1,199（81.0%）** 的测试曲目与训练集共享至少一个 artist。

这与「先按曲目划分再切片」的协议并不矛盾——协议要求的是不泄漏**同一首曲目**，
但 artist 泄漏是 FMA 特有的更强捷径，必须单独处理。

### 4.2 主划分（`data/splits/fma/splits.csv`，artist-disjoint）

- 算法：`make_artist_splits()`，seed = 42。把每个 artist 当作不可分割的整体，
  随机打乱后按曲目数从大到小，逐一组贪心分配到「相对缺口最大」的划分
  （`(target - current) / target`），保证总体比例接近 70/15/15。
- 结果：**训练 5,596 / 验证 1,199 / 测试 1,199 = 70.00% / 15.01% / 14.99%**。
- **artist 完全不相交**：train∩val = 0，train∩test = 0，val∩test = 0；
  测试集 344 个 artist，**0 首**测试曲目与训练共享 artist（`stats.json` 中
  `test_tracks_with_train_artist = 0`）。
- 可复现：用同一份输入重跑 `make_artist_splits(seed=42)`，与已落盘的 `split` 列
  **逐行完全一致**。

**每流派 × 划分（曲目数）**：

| 流派 | train | val | test | 合计 |
|---|---|---|---|---|
| Electronic | 726 | 134 | 139 | 999 |
| Experimental | 733 | 119 | 147 | 999 |
| Folk | 614 | **249** | 137 | 1000 |
| Hip-Hop | 694 | 168 | 135 | 997 |
| Instrumental | 792 | **82** | 126 | 1000 |
| International | 650 | 194 | 156 | 1000 |
| Pop | 689 | 120 | **191** | 1000 |
| Rock | 698 | 133 | 168 | 999 |
| **合计** | **5596** | **1199** | **1199** | **7994** |

### 4.3 为什么每类 val/test 数量不均（如实记录，不强行抹平）

按 artist 分组后，一整个曲库只能整体移动，无法再细分。实测归因：

- Folk 验证集 249 首，其中 **Big Blood（91 首）** 整个落到 val，是主要来源；
- Instrumental 验证集只有 82 首，因为它的几个大曲库都落在别处：
  Blue Dot Sessions（208 首）→ train，The Impossebulls（68 首）→ test，Cory Gray（36 首）→ test。

因此 val 每类 82–249、test 每类 126–191 属**预期结果**。训练集每类仍有 614–792 首，
类别信息量充足；模型选择/评估时应同时报告 **Macro-F1** 与混淆矩阵，不要只看 Accuracy，
并注意 val 上 Folk/Instrumental 的样本数差 3 倍，小类指标方差更大。

### 4.4 对比划分（`data/splits/fma/splits_tracklevel.csv`，仅作泄漏对照）

- 曲目级分层 70/15/15：**5,595 / 1,200 / 1,199**，每流派每划分恰为 150/150/150（8×150=1200）。
- **814 个 artist 跨划分出现**；测试集 **971/1,199（81.0%）** 曲目与训练共享 artist。
- 该文件**不用于训练**，只用于「有 artist 泄漏 vs 无泄漏」的对照实验，用来量化
  「模型是在听流派还是在记歌手」。

---

## 5. 特征缓存

命令（耗时 **535.1 s ≈ 8.9 min**，12 workers，输出 176 MB）：

```bash
$PY scripts/extract_features.py \
    --splits data/splits/fma/splits.csv \
    --out data/features/fma/features.npz \
    --workers 12
```

`data/features/fma/features.npz` 内容：

| 键 | 形状 / 类型 | 说明 |
|---|---|---|
| `track_ids` | (7994,) int64 | 与 `splits.csv` **同序**一一对应 |
| `genres` | (7994,) object | 8 个流派字符串 |
| `splits` | (7994,) object | train / val / test |
| `X_feat` | (7994, 209) float32 | MFCC+Δ 均值/方差、chroma、spectral contrast、质心/带宽/rolloff/ZCR/RMS/节拍 → SVM、MLP |
| `X_mel` | (7994, 64, 216) float16 | 64 维 Log-Mel，5 秒 @ 22.05 kHz，hop 512 → CNN/CRNN/Transformer |

`N = 7994` 与可解码曲目数完全一致（**解码失败 0 例**）。逐项校验结果：

- `track_ids` / `splits` / `genres` 与 `splits.csv` 逐行一致，无重复、无缺失；
  划分计数 5,596 / 1,199 / 1,199、每流派计数与第 2 节相同。
- `X_feat` 与 `X_mel` **无 NaN、无 Inf**；`X_feat` 取值范围 −1131.4 ~ 9548.4（MFCC 直流分量可到 −1000 量级，正常）；
  `X_mel` 取值 **−80.0 ~ 0.0 dB**（`power_to_db(ref=np.max)`，每行最大值恰为 0 dB）。
- **注意：`genres`/`splits` 是 object 数组**，读取必须
  `np.load(..., allow_pickle=True)`，否则 numpy 会抛 `ValueError: Object arrays cannot be loaded`。
- **注意：`track_ids` 是 int64**。`splits.csv` 落盘时是 6 位补零文本（`000002`），
  pandas 读入后变成整数 2；回查路径需 `f"{tid:06d}"`。

### 5.1 需要下游知道的样本质量

1. **4 首「中心 5 秒恰好是数字静音」**：`107535`（train, Instrumental）、`110736`（train, Hip-Hop）、
   `119979`（train, Hip-Hop）、`119257`（**test**, Rock）。这些曲目整体有声
   （整轨 max|x| 0.25–0.98），但中间 5 秒是全零，导致 209 维里只有 4 维非零常数、
   Log-Mel 全为 0 dB 的退化输入。占比 0.05%，未剔除（保持「N=可解码数」口径一致）；
   若要严格处理，建议在训练/评估时按 `track_ids` 过滤这 4 首——**其中 1 首在测试集**，
   会在极小程度上影响测试指标。
2. 中心片段很安静的曲目还有：`rms < 1e-3` 共 10 首、`< 1e-2` 共 58 首、`< 5e-2` 共 523 首
   （安静段落，属真实音频内容，保留）。
3. **5 首 MP3 含单个损坏帧**：`011298`、`021657`、`029245`、`054568`、`054576`，
   解码时 libmpg123 打印 `dequantization failed` / `part2_3_length too large`，
   但**均成功返回完整 30 秒音频**，抽查其中心片段 `rms` 0.11–0.31、Log-Mel 均值正常，
   特征未受影响。提取日志里只看到 4 行，是因为 12 个 worker 并发写同一 stderr 时
   行被交错截断；逐文件重解码复核后确认是 5 首。

---

## 6. 与 GTZAN（MVP）的差异

| 维度 | GTZAN（MVP） | FMA-small（主实验） |
|---|---|---|
| 可用曲目 | 985（原始 1000，剔 1 坏 + 14 重复） | **7,994**（8,000 中剔除 6 个上游损坏） |
| 流派 | 10 类，每类约 100 | **8 类**，每类约 1,000 |
| 格式 | 22,050 Hz 单声道 WAV | MP3：44.1k×7569 / 48k×411 / 22.05k×14；立体声 7909 / 单声道 85 |
| 时长 | 29.93–30.65 s（均值 30.02） | 29.965–30.003 s（均值 29.988） |
| artist 元数据 | 无 | 有，2,304 个 artist，可用于去泄漏 |
| 划分 | 曲目级 70/15/15 = 689/149/147 | **artist-disjoint** 70/15/15 = 5,596/1,199/1,199 |
| 测试集 artist 泄漏 | 无法度量（无 artist 信息） | **0 首**（对照：曲目级划分 81.0%） |
| 完全重复音频 | 28 首（MD5 相同） | **0 组** |
| 每类 val 数量 | 均匀（约 15） | 82–249（artist 分组所致，见 4.3） |
| 特征缓存 | `data/features/gtzan/features.npz`，985 条 | `data/features/fma/features.npz`，7,994 条 |

对实验设计的三点影响：

1. 数据量约 **8 倍**，测试集 1,199 首（GTZAN 147 首），accuracy 的置信区间显著收窄，
   少量样本翻车不再主导结论。
2. 因为做了 artist-disjoint，FMA 上的绝对准确率**可能低于**曲目级划分的同类结果，
   这是更诚实的数字，**不应**与「有 artist 泄漏」的结果直接比较排名。
3. 8 个流派全部是 FMA 的顶层流派，语义与 GTZAN 的 10 类不完全对齐（没有 Jazz/Blues/Classical
   这些 GTZAN 主力类），两个数据集的逐类 F1 不可直接横向比较。

---

## 7. 复现命令（完整顺序）

```bash
cd ~/MUSIC_AI
PY=$CONDA_PREFIX/bin/python

# 0) 下载（已下载可跳过，脚本自带 -C - 续传与 skip 已存在文件）
bash scripts/download_fma.sh          # 末尾打印 DOWNLOAD_DONE

# 1) 校验两个 zip
$PY -c "import zipfile; print(zipfile.ZipFile('data/fma/raw/fma_metadata.zip').testzip())"
$PY -c "import zipfile; print(zipfile.ZipFile('data/fma/raw/fma_small.zip').testzip())"   # 约 5 min

# 2) 元数据去前缀（见 1.2，只需一次）
# 3) 解压 + 审计 + 双划分
$PY -m src.data.prepare_fma --extract --audit --split

# 3.5) 解压后与上游 SHA1 清单比对（可选，约 2 min）
cd data/fma/fma_small && sha1sum -c checksums | grep -v ': OK$'; cd ../../..

# 4) 特征缓存（只对主划分 = artist-disjoint）
$PY scripts/extract_features.py \
    --splits data/splits/fma/splits.csv \
    --out data/features/fma/features.npz \
    --workers 12
```

产出与磁盘占用：

| 路径 | 内容 | 大小 |
|---|---|---|
| `data/fma/raw/` | 两个 zip 原件 | 7.5 GiB |
| `data/fma/fma_small/` | 8,000 个 MP3（`000/`…`155/`）+ 归档自带的 `README.txt`、`checksums` | 7.5 GiB |
| `data/fma/metadata/` | `tracks.csv`、`genres.csv` | 0.26 GiB |
| `data/features/fma/features.npz` | 特征缓存 | 176 MB |
| `data/splits/fma/splits.csv` | **主划分**（artist-disjoint） | 863 KB |
| `data/splits/fma/splits_tracklevel.csv` | 泄漏对照划分 | 863 KB |
| `data/splits/fma/stats.json` | 规模/采样率/失败清单/泄漏统计 | 1.6 KB |

`data/fma` 合计约 **17 GB**（`.gitignore` 已排除 `data/`，音频与特征均不入库）。

---

## 8. 已知问题与限制

1. **6 首上游损坏**：`098565`、`098567`、`098569`、`099134`、`108925`、`133297`，
   已剔除，不可通过重新下载修复（CRC 与 zip 内条目大小证明）。
2. **4 首中心 5 秒为静音**（见 5.1），特征退化，其中 `119257` 在测试集。
3. **每类 val/test 数量不均**（82–249 / 126–191），是 artist-disjoint 的固有代价，
   已在 4.3 归因，评估时请报告 Macro-F1。
4. **artist 名称未做归一化**：FMA 元数据中同一乐队可能因大小写/写法不同被当成两个 artist，
   本流程按元数据原样分组；这只会让「去泄漏」略偏保守（可能仍有个别同一实体跨划分）。
5. **元数据 top-level 前缀与脚本期望路径不一致**（1.2），本机用去前缀解压绕过；
   跨机器复现时注意。
6. `genres`/`splits` 为 object 数组、`track_ids` 为 int64（5 节），读取时必须
   `allow_pickle=True` 并注意补零。

---

## 9. 引用

```bibtex
@inproceedings{defferrard2017fma,
  title={FMA: A Dataset for Music Analysis},
  author={Defferrard, Micha{\"e}l and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle={18th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2017}
}
```

- 代码与说明：<https://github.com/mdeff/fma>
- 音频镜像：<https://os.unil.cloud.switch.ch/fma/>
- 数据划分与统计：`data/splits/fma/splits.csv`、`splits_tracklevel.csv`、`stats.json`
