#!/bin/bash
# Website clips v3 (mechanism visibility): the v2 episodes (stage-1 N1 v2 policy cropen_ppo_e3nv2_s4 epoch 10000, physics v2
# at 255, seed 300, attempts 10) re-rendered with a close low camera, mechanism colours and a read-only state log.
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /workspace/sitemedia
D=/workspace/sitemedia/v2copy/dexrl-cropen-v2/cropen
CK=/workspace/sitemedia/ck/cropen_ppo_e3nv2_s4/ckpt_10000.pt
OUT=/workspace/sitemedia/final_v3
mkdir -p $OUT
sha256sum v6/tools/render_demo.py
for F in M1 M2 M3 M6; do
  f=${F,,}
  timeout 3600 nice -n 5 /venv/main/bin/python -B v6/tools/render_demo.py --cropen-dir $D --ckpt $CK --family $F --out $OUT --tag $f --pretty --mech-colors --eye 0.093 -0.1994 0.385 --lookat 0.0 0.0 0.335 --focal 24 --attempts 10 --seed 300 --reset-pose $D/data/pose_n1.json --flex-floor $D/data/flex_floor_n1.json --solver-pos-iters 255 --state-log $OUT/${f}_state.json --keep-frames > /workspace/sitemedia/v3_$F.log 2>&1
  echo "$F rc=$?"
  grep -E '^\{"attempt' /workspace/sitemedia/v3_$F.log
done
ls -la $OUT
echo RENDER_V3_DONE
