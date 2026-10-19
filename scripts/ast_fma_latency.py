"""Measure end-to-end AST latency on 20 FMA test clips and fix metrics metadata."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import ASTModel, ASTFeatureExtractor

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/"results/fma"
MODEL_ID="MIT/ast-finetuned-audioset-10-10-0.4593"; SR=16000
df=pd.read_csv(ROOT/"data/splits/fma/splits.csv"); te=df[df.split=="test"].head(20)
model=ASTModel.from_pretrained(MODEL_ID).eval().cuda().half()
fe=ASTFeatureExtractor.from_pretrained(MODEL_ID)
import librosa
lat=[]
with torch.inference_mode():
    for r in te.itertuples():
        t=time.perf_counter()
        y,_=librosa.load(str(ROOT/r.path), sr=SR, mono=True,
                         offset=max(0,(float(r.duration)-5)/2), duration=5.0)
        y=np.pad(y,(0,int(SR*10.24)-len(y)))[:int(SR*10.24)]
        inp=fe(y, sampling_rate=SR, return_tensors="pt")["input_values"].cuda().half()
        model(input_values=inp)
        lat.append((time.perf_counter()-t)*1000)
median=float(np.median(lat)); n_backbone=sum(p.numel() for p in model.parameters())
n_head=768*8+8; size=sum(p.numel()*4 for p in model.parameters())/1e6+0.03
m=json.loads((RES/"ast_metrics.json").read_text())
m["infer_ms_per_clip"]=round(median,2); m["n_params"]=int(n_backbone+n_head)
m["model_size_mb"]=round(size,2); m["notes"]+=f";latency_median_ms={median:.1f};fp16_forward"
(RES/"ast_metrics.json").write_text(json.dumps(m,indent=2,ensure_ascii=False))
ev=json.loads((RES/"ast_eval.json").read_text())
ev.setdefault("cost",{}).update({"n_params":int(n_backbone+n_head),"model_size_mb":round(size,2)})
ev["timing"]={"latency_median_ms":round(median,2),"n_clips":len(lat)}
(RES/"ast_eval.json").write_text(json.dumps(ev,indent=2,ensure_ascii=False))
print("latency",round(median,2),"ms; params",n_backbone+n_head,"size",round(size,2))
