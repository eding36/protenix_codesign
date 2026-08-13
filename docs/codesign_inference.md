# Antibody CDR design at inference

Instructions for designing CDR residues and structure simultaneously.

## 1. Input JSON

One entry per complex, one `proteinChain` per chain. **Mask the residues you want
designed with `X`.** Everything else is kept.

```json
[
  {
    "name": "my_complex",
    "sequences": [
      {"proteinChain": {"sequence": "EVQLQQSGPELVKPGASMKISCKTSXXXXXXXTMNWVKQ...", "count": 1, "id": ["H"]}},
      {"proteinChain": {"sequence": "DIVMTQSPASLAVSLGQRATISCXXXXXXXXXXXXXXXWYQQ...", "count": 1, "id": ["L"]}},
      {"proteinChain": {"sequence": "TVEELKKLLEQWNLVIGFLFLTWICLLQFAYANRNRF...", "count": 1, "id": ["A"]}}
    ]
  }
]
```

## 2. Run

```bash
export PROTENIX_ROOT_DIR=/path/to/data          # needs common/components.cif
export LAYERNORM_TYPE=torch                     # required on Blackwell (sm_120)

python runner/inference.py \
  --model_name protenix_base_default_v1.0.0_codesign \
  --load_checkpoint_path /path/to/stage_4.pt \
  --input_json_path input.json \
  --dump_dir out \
  --seeds 42 \
  --model.N_cycle 4 \
  --sample_diffusion.N_sample 20 \
  --sample_diffusion.N_step 200 \
  --triangle_attention cuequivariance \
  --triangle_multiplicative cuequivariance
```

`--sample_diffusion.N_sample` is the number of designs; they are confidence-ranked,
rank 0 first.

## 3. Output

```
out/<name>/seed_42/predictions/
  <name>_seed_42.seq                     designs, one row per rank
  <name>_seed_42.fasta                   same, FASTA
  <name>_sample_<i>.cif                  structure per design
  <name>_summary_confidence_sample_<i>.json
```

Design is driven solely by the `X` positions. With no `X` anywhere, nothing is designed: the sequence is returned unchanged and only the structure is predicted.
