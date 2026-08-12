#!/usr/bin/env bash
# Build the antibody-codesign test split so it matches MFDesign's yaml inputs
# token for token, then keep only the targets that match exactly.
#
#   bash scripts/build_mfdesign_test_set.sh            # build + report, no install
#   INSTALL=1 bash scripts/build_mfdesign_test_set.sh  # also install into the data root
#
# Produces 180 targets from MFDesign's 204.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${PROTENIX_ROOT_DIR:-/home/dinge/data/proj/protenix_codesign/data}"
YAML_DIR="${YAML_DIR:-/home/dinge/MFDesign/examples/test_yaml_dir/full}"
STAGE="${STAGE:-/tmp/mfdesign_test_build}"
PY="${PY:-/home/dinge/miniconda3/envs/protenix/bin/python}"
NCPU="${NCPU:-30}"
INSTALL="${INSTALL:-0}"
# Roles never resolve: 9fve declares chains B/A that are absent from the file.
SKIP="${SKIP:-9fve_B__A}"

ENTRY="$ROOT/preproc_mfdesign/step6_output/mfdesign_test_split/test_entry.json"
SUMMARY="$ROOT/preproc_mfdesign/step1_output/summary.json"
CIFS="$ROOT/preproc_mfdesign/step1_input/raw_data/cif"

for f in "$ENTRY" "$SUMMARY" "$CIFS" "$YAML_DIR"; do
  [ -e "$f" ] || { echo "[error] missing: $f" >&2; exit 1; }
done

rm -rf "$STAGE"; mkdir -p "$STAGE"/{bio,cif,final_bio,final_cif}

echo "== 1/4  filtering input entries (skip: $SKIP)"
"$PY" - "$ENTRY" "$STAGE/entry.json" "$SKIP" <<'PY'
import json, sys
src, dst, skip = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
d = json.load(open(src))
out = [x for x in d if x not in skip] if isinstance(d, list) else {k: v for k, v in d.items() if k not in skip}
json.dump(out, open(dst, "w"))
print(f"   {len(d)} -> {len(out)} entries")
PY

echo "== 2/4  building bioassemblies (--mfdesign_chain_subset)"
PROTENIX_ROOT_DIR="$ROOT" LAYERNORM_TYPE=torch "$PY" scripts/prepare_training_data.py \
  -i "$STAGE/entry.json" \
  --complexes_json "$SUMMARY" \
  --mmcif_dir "$CIFS" \
  --output_csv "$STAGE/indices.csv" \
  --bio_output_dir "$STAGE/bio" \
  --cif_output_dir "$STAGE/cif" \
  --strip_antibody_cdr \
  --mfdesign_chain_subset \
  --n_cpu "$NCPU" 2>&1 | grep -oE "\[dedup\] test.*" || true

echo "== 3/4  keeping targets whose token count matches MFDesign's yaml"
"$PY" - "$STAGE" "$YAML_DIR" <<'PY'
import os, shutil, sys, yaml
import pandas as pd
stage, ydir = sys.argv[1], sys.argv[2]
df = pd.read_csv(f"{stage}/indices.csv")
tok = df.groupby("pdb_id").num_tokens.first()
keep, off = [], []
for pid, n in tok.items():
    f = f"{ydir}/{pid}.yaml"
    if not os.path.exists(f):
        continue
    th = sum(len(v.get("sequence", "")) for s in yaml.safe_load(open(f))["sequences"] for v in s.values())
    (keep if int(n) == th else off).append(pid)
keep.sort()
open(f"{stage}/exact_ids.txt", "w").write("\n".join(keep) + "\n")
df[df.pdb_id.isin(keep)].to_csv(f"{stage}/final_indices.csv", index=False)
for pid in keep:
    for sub, ext in (("bio", ".pkl.gz"), ("cif", ".cif")):
        src = f"{stage}/{sub}/{pid}{ext}"
        if os.path.exists(src):
            shutil.copy2(src, f"{stage}/final_{sub}/{pid}{ext}")
print(f"   exact {len(keep)}   discarded {len(off)}")
print(f"   tokens: median {tok[keep].median():.0f}  max {tok[keep].max()}")
PY

N=$(wc -l < "$STAGE/exact_ids.txt")
echo "== 4/4  $N targets staged in $STAGE/final_{bio,cif} + final_indices.csv"

if [ "$INSTALL" = "1" ]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  [ -d "$ROOT/test_bioassembly" ] && mv "$ROOT/test_bioassembly" "$ROOT/test_bioassembly.bak_$TS"
  [ -f "$ROOT/indices/test_indices.csv" ] && mv "$ROOT/indices/test_indices.csv" "$ROOT/indices/test_indices.bak_$TS.csv"
  [ -d "$ROOT/test_mmcif" ] && mv "$ROOT/test_mmcif" "$ROOT/test_mmcif.bak_$TS"
  cp -r "$STAGE/final_bio" "$ROOT/test_bioassembly"
  cp -r "$STAGE/final_cif" "$ROOT/test_mmcif"
  cp "$STAGE/final_indices.csv" "$ROOT/indices/test_indices.csv"
  echo "   installed into $ROOT (previous set kept as *.bak_$TS)"
else
  echo "   not installed; rerun with INSTALL=1 to replace the data root"
fi
