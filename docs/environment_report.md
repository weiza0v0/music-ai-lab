# 环境审计报告（阶段 A）

- 生成时间：2026-10-19T12:10:16+08:00
- 操作系统：Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
- Python：3.12.13（/home/weizai0v0/.apps/anaconda3/envs/DataAnalysis/bin/python）
- CPU：x86_64 / 16 逻辑核
- 内存：Mem:           7.8Gi       3.4Gi       1.1Gi       4.0Mi       3.2Gi       4.1Gi
- 磁盘：
```
Filesystem      Size  Used Avail Use% Mounted on
D:\             1.6T  206G  1.4T  14% /mnt/d
```

## GPU

- `nvidia-smi` 正常：
```
Mon Oct 19 12:10:18 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.54                 Driver Version: 595.79         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   50C    P0             15W /   80W |       0MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|======================================================================================
```
- torch 可见 1 块 GPU：NVIDIA GeForce RTX 4060 Laptop GPU，算力 (8, 9)。

## PyTorch

- torch 2.12.1+cu130，CUDA 可用：True，设备数：1

## 关键依赖

| 包 | 版本 |
|---|---|
| torch | 2.12.1 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| librosa | 1.0.0 |
| soundfile | 0.14.0 |
| matplotlib | 3.11.0 |
| streamlit | 1.66.0 |
| pytest | 9.1.1 |

## 音频工具链

- ffmpeg: ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
- ffprobe: /usr/bin/ffprobe
