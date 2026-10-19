"""Music AI Lab — 现场演示界面（Streamlit）。

Run:  .venv/bin/streamlit run app/demo.py
All predictions come from checkpoints produced by src/train_*.py.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import available, predict_audio  # noqa: E402

st.set_page_config(page_title="Music AI Lab · 音乐 AI 鉴定所", page_icon="🎧", layout="wide")

MODELS = [m for m in ["svm", "mlp", "cnn", "crnn", "transformer"] if m in available()]
LABEL = {"svm": "SVM（人工特征）", "mlp": "MLP（人工特征）", "cnn": "CNN（Log-Mel）",
         "crnn": "CRNN（CNN+GRU）", "transformer": "Transformer（轻量）"}


@st.cache_data(show_spinner=False)
def _load_bytes(name: str, data: bytes):
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050
    return y, sr


def show_audio(y: np.ndarray, sr: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.6))
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#c2703d")
    axes[0].set_title("波形", fontsize=10)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
    img = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr,
                                   hop_length=512, x_axis="time", y_axis="mel", ax=axes[1],
                                   cmap="magma")
    axes[1].set_title("Log-Mel 频谱", fontsize=10)
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.title("🎧 Music AI 鉴定所")
st.caption("GTZAN 10 流派 · 上传一段音乐，看不同模型各自怎么判断")

if not MODELS:
    st.error("还没有 checkpoint，先运行 `python -m src.train_classical` 和 `python -m src.train_cnn`。")
    st.stop()

uploaded = st.file_uploader("上传音频（wav / mp3 / flac / ogg）", type=["wav", "mp3", "flac", "ogg", "m4a"])
if uploaded is None:
    st.info("也可以直接用测试集里的曲目：`data/raw/gtzan/genres/blues/blues.00008.wav`（测试集，模型没见过）")
    st.stop()

y, sr = _load_bytes(uploaded.name, uploaded.getvalue())
st.audio(uploaded)
show_audio(y, sr)

col_pick, col_go = st.columns([1, 1])
with col_pick:
    picked = st.multiselect("参赛模型", MODELS, default=MODELS)
with col_go:
    st.write("")
    run = st.button("开始鉴定", type="primary", use_container_width=True)

if run:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        import soundfile as sf
        sf.write(t.name, y, sr)
        tmp = t.name
    results = {}
    for m in picked:
        results[m] = predict_audio(tmp, m)
    Path(tmp).unlink(missing_ok=True)

    tabs = st.tabs([LABEL.get(m, m) for m in picked] + ["🏆 多模型对比"])
    for tab, m in zip(tabs[:len(picked)], picked):
        with tab:
            top = results[m]["top3"]
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.metric("Top-1", top[0][0], f"{top[0][1]*100:.1f}%")
                st.table(pd.DataFrame([{"流派": g, "置信度": f"{p*100:.1f}%"} for g, p in top]))
            with c2:
                allp = results[m]["all"]
                fig, ax = plt.subplots(figsize=(4.2, 3))
                gs = list(allp)[::-1]
                ax.barh(gs, [allp[g] for g in gs], color="#c2703d")
                ax.set_xlim(0, 1)
                ax.set_xlabel("概率")
                ax.set_title("全类别概率", fontsize=10)
                st.pyplot(fig)
                plt.close(fig)

    with tabs[-1]:
        rows = [{"模型": LABEL.get(m, m), "Top-1": results[m]["top3"][0][0],
                 "置信度": f"{results[m]['top3'][0][1]*100:.1f}%"} for m in picked]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        tops = {results[m]["top3"][0][0] for m in picked}
        if len(tops) > 1:
            st.warning(f"模型意见不一致：{', '.join(sorted(tops))} — 这正是可以讲的故事。")
        else:
            st.success(f"所有模型都认为这是 {tops.pop()}。")

st.divider()
st.caption("预测来自真实 checkpoint；页面不包含任何预设结果。")
