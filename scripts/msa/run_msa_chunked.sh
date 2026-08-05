#!/usr/bin/env bash
# Run the ColabFold MSA search in chunks.
#
# WHY: mmseqs `expandaln` segfaults on large query sets (ColabFold issue #682
# reports it across 5k-50k sequences). A ~9,336-sequence run of this pipeline
# completed previously; 11,389 dies at ~1%. Splitting the fasta into pieces below
# the size that last worked sidesteps it.
#
# NAMING: search.py writes final MSAs as `{job_number}.a3m`, numbered by position
# in the input fasta. Chunks are contiguous slices, so a chunk starting at global
# offset K produces local 0..n-1 that map exactly to global K..K+n-1. Renaming by
# that offset reproduces the numbering an unchunked run would have given, so
# downstream steps cannot tell the difference.
#
# Usage:
#   bash scripts/msa/run_msa_chunked.sh
#   CHUNK=3000 bash scripts/msa/run_msa_chunked.sh
#   USE_TEMPLATES=1 bash scripts/msa/run_msa_chunked.sh   # off by default
set -euo pipefail

R="${R:-/home/dinge/data/proj/protenix_codesign/data}"
CF="${CF:-/home/dinge/MMseqs2/ColabFold}"
FASTA="${FASTA:-$R/pdb_seqs/pdb_seq.fasta}"
OUT="${OUT:-$R/mmcif_msa_initial}"
WORK="${WORK:-$R/msa_chunks}"
MMSEQS="${MMSEQS:-/usr/local/bin/mmseqs}"   # must match the index generator
CHUNK="${CHUNK:-4000}"
PARALLEL="${PARALLEL:-6}"          # chunks run concurrently
NCORES=$(nproc)
# expandaln is forced single-threaded in search.py (it races and segfaults at any
# thread count > 1), so throughput comes from running chunks side by side instead.
# The other stages still thread, so divide cores across the concurrent chunks.
THREADS="${THREADS:-$(( NCORES / PARALLEL > 0 ? NCORES / PARALLEL : 1 ))}"
# Templates are off by default: Protenix runs with enable_prot_template false and
# MFDesign (Boltz-1) has no template pathway at all, so the pdb70 search is pure
# cost. Set USE_TEMPLATES=1 to restore it.
USE_TEMPLATES="${USE_TEMPLATES:-0}"

[[ -f "$FASTA" ]] || { echo "[error] no fasta: $FASTA" >&2; exit 1; }
[[ -x "$MMSEQS" ]] || { echo "[error] no mmseqs: $MMSEQS" >&2; exit 1; }

TOTAL=$(grep -c '^>' "$FASTA")
NCHUNK=$(( (TOTAL + CHUNK - 1) / CHUNK ))
echo "fasta   : $FASTA  ($TOTAL sequences)"
echo "chunks  : $NCHUNK x $CHUNK"
echo "work    : $WORK"
echo "final   : $OUT"
echo "parallel: $PARALLEL chunks x $THREADS threads (of $NCORES cores)"
echo "mmseqs  : $MMSEQS ($($MMSEQS version 2>/dev/null))"
echo

mkdir -p "$WORK" "$OUT"

# --- split, preserving record order (offsets must stay contiguous) ------------
echo "[split] writing chunk fastas..."
/home/dinge/miniconda3/envs/protenix/bin/python - "$FASTA" "$WORK" "$CHUNK" <<'PY'
import sys, os
fasta, work, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
recs, h, cur = [], None, []
for line in open(fasta):
    if line.startswith(">"):
        if h is not None: recs.append((h, "".join(cur)))
        h, cur = line.rstrip(), []
    else: cur.append(line.rstrip())
if h is not None: recs.append((h, "".join(cur)))
offsets = []
for i in range(0, len(recs), chunk):
    k = i // chunk
    with open(os.path.join(work, f"chunk_{k:03d}.fasta"), "w") as f:
        for hdr, seq in recs[i:i+chunk]:
            f.write(f"{hdr}\n{seq}\n")
    offsets.append((k, i, len(recs[i:i+chunk])))
with open(os.path.join(work, "offsets.tsv"), "w") as f:
    for k, off, n in offsets:
        f.write(f"{k}\t{off}\t{n}\n")
print(f"  {len(offsets)} chunks, {len(recs)} records total")
PY

TPL_FLAG=()
[[ "$USE_TEMPLATES" == "1" ]] && TPL_FLAG=(--use-templates 1)

# --- search each chunk into its own dir ---------------------------------------
run_chunk() {
  local K="$1" OFFSET="$2" N="$3"
  local CDIR="$WORK/out_$(printf '%03d' "$K")"
  local DONE="$CDIR/.complete"
  if [[ -f "$DONE" ]]; then
    echo "[chunk $K] already complete -- skipping"
    return 0
  fi
  echo "[chunk $K] start offset=$OFFSET n=$N"
  # mmseqs refuses to overwrite existing result dbs, so start each attempt clean
  rm -rf "$CDIR"
  mkdir -p "$CDIR"

  python3 "$CF/colabfold/mmseqs/search.py" \
    --mmseqs="$MMSEQS" \
    "$WORK/chunk_$(printf '%03d' "$K").fasta" \
    "$CF" \
    "$CDIR" \
    --db1 uniref30_2202_db \
    --db2 pdb70_220313 \
    --db3 colabfold_envdb_202108_db \
    "${TPL_FLAG[@]}" \
    --db-load-mode 2 \
    --threads "$THREADS"

  touch "$DONE"
  echo "[chunk $K] done"
}

# bounded concurrency: at most PARALLEL chunks in flight
while IFS=$'\t' read -r K OFFSET N; do
  while (( $(jobs -rp | wc -l) >= PARALLEL )); do wait -n; done
  run_chunk "$K" "$OFFSET" "$N" > "$WORK/chunk_${K}.log" 2>&1 &
done < "$WORK/offsets.tsv"
wait
echo
echo "[chunks] all finished; failures (if any):"
grep -Ll "done" "$WORK"/chunk_*.log 2>/dev/null | head || true

# --- merge with global offsets -------------------------------------------------
echo
echo "[merge] renaming into $OUT with global offsets..."
MOVED=0
while IFS=$'\t' read -r K OFFSET N; do
  CDIR="$WORK/out_$(printf '%03d' "$K")"
  for i in $(seq 0 $((N - 1))); do
    SRC="$CDIR/${i}.a3m"
    [[ -f "$SRC" ]] || { echo "[warn] missing $SRC" >&2; continue; }
    cp -n "$SRC" "$OUT/$((OFFSET + i)).a3m"
    MOVED=$((MOVED + 1))
  done
done < "$WORK/offsets.tsv"

echo "[merge] $MOVED a3m files -> $OUT"
echo "expected $TOTAL; missing $((TOTAL - MOVED))"
echo
echo "Next: python scripts/msa/build_uniref_tax_m8.py --db_prefix $CF/uniref30_2202_db --a3m_dir $OUT"
