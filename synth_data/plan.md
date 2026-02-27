Below is a concrete, “coding-agent-ready” plan that turns your notes into a reproducible pipeline you can iterate on. It’s written as **implementation tickets + repo skeleton + interfaces** so an agent can start building immediately and you can swap components later.

---

## 0) Goal and invariants

**Goal:** an end-to-end loop for multilingual MT SFT with synthetic data:

1. train student on OPUS-100 subset
2. evaluate on fixed FLORES-200 devtest (+ metrics)
3. run *candidate-pool* eval → compute hardness → select top-k
4. generate targeted synthetic pairs via teacher (vLLM)
5. filter + merge data
6. SFT student via LlamaFactory (apptainer)
7. repeat for N iterations

**Invariants:**

* FLORES devtest is **frozen** (never used for selection / generation).
* Every artifact is versioned per iteration: `iter_00`, `iter_01`, …
* Every dataset row has stable IDs + provenance metadata.

---

## 1) Repo skeleton (what the agent should create)

**New repo:** `active_mt_distill/`

```text
active_mt_distill/
  configs/
    exp_base.yaml
    llama_factory/
      sft_template.yaml
  data/
    raw/               # optional caches / downloaded datasets
    processed/
      flores_devtest.jsonl
      opus_train_pool.jsonl  # separate from candidate_pool.jsonl
      candidate_pool.jsonl
    iterations/
      iter_00/
        # evertythin we need to snapshort per iteration
        train.jsonl      # (ids of original set: what you would train on if you did “no active synthetic generation” this round)
        synth.jsonl      # (The new synthetic parallel data generated this iteration from selected.jsonl.), each row is a synthetic training pair: src_variant, tgt_translation, plus provenance: parent_id (which selected example it came from), strategy (e.g., active_variant_minimal_pairs), teacher_model, temperature, etc.
        merged_train.jsonl (The final training dataset used for SFT in this iteration.)
        selected.jsonl   # (The top-k “hard” examples chosen from the candidate_pool. These are the seeds you feed into the teacher prompts to generate targeted synthetic data.)
        student_outputs.jsonl      #  The student model’s translations on the candidate_pool for this iteration. (This is used to compute hardness (loss/uncertainty, QE, etc.) and to debug why items were selected.)
        hardness_scores.parquet    #  A table of hardness scores per candidate example. This is the basis for selecting top-k.
      iter_01/...
  src/
    orchestrator/
      run_iteration.py
      pipeline.py  # submits a GPU job for the whole pipeline (see how we implement sweep_runner.py in the main project for reference)
      io.py  # some file utils if needed
    datasets/
      flores.py  # Responsible for creating your frozen eval set. Load FLORES-200 devtest for chosen language pairs. Convert to your canonical JSONL schema. Write data/processed/flores_devtest.jsonl
      opus100.py # produces train pool (SFT data); candidate pool (selection data); Sample fixed-size subsets per pair (e.g., 20k train, 2k candidate). candidate pool must be disjoint from train pool!
      formatters.py  # Chat/SFT format for LlamaFactory (often “instruction + response” style). To save the samples for generation, should use the current formatting from src/ folder
      filters.py  # Applies quality gates to remove bad synthetic or noisy parallel examples. only length bounds (min/max tokens or chars) for now, and langid checks to make sure src/tgt are in the right languages. For now implement as simple functions, can later swap in a more complex class-based system if you want.
      dedup.py   # Removes duplicates (especially critical for synthetic generation loops). for now implement a simpe hash-based dedup (MinHash) to set up API, can later swap in a more complex embedding-based dedup if you want.
    eval/
      connect evaluation/ code from the main project;
      eval_runner.py submits eval jobs (see the main project sweep for reference);
    selection/
      hardness.py  # takes candidate_pool.jsonl; student_outputs.jsonl; computes hardness signals (self_nll, forced_nll_ref, QE if enabled); writes hardness_scores.parquet; need to set API;
      select_topk.py  # Implements the selection policy: pick the examples you’ll feed into the teacher.
    generation/
      TBD: should be based on src/ sampling primitives;
    sft/
      llamafactory_runner.py  # docker based SFT runner specified via a config;
      dataset_export.py   # Take merged_train.jsonl and and export it into whatever dataset format your LlamaFactory container expects, plus any small “glue” files/config references needed. Outputs: lf_train.jsonl (or .json) in LlamaFactory-compatible format; optionally lf_valid.jsonl; a manifest describing source files, export settings, counts per language pair, hash/checksum for reproducibility;
    utils/
      logging.py
      hashing.py # Text normalization + hashing; exact: hash(normalized src_text) or hash(src_text + tgt_text) / near-dup (optional later): shingles/minhash/simhash
      langid.py  # langid checks for filtering
  scripts/
    prepare_data.sh
    run_iter.sh
    eval_model.sh
  README.md
  pyproject.toml
```

**Key design:** Everything is driven from `configs/exp_base.yaml`, and `src/orchestrator/run_iteration.py` is the single entrypoint.

---

## 2) Config file spec (agent should implement)

`configs/exp_base.yaml` (example fields)

```yaml
experiment_name: mt_active_distill_v0
seed: 13

languages:
  pairs:
    - {src: "eng_Latn", tgt: "deu_Latn"}
    - {src: "eng_Latn", tgt: "fra_Latn"}
    - {src: "eng_Latn", tgt: "rus_Cyrl"}
    # add 8–12 pairs total

data:
  flores_devtest_path: "data/processed/flores_devtest.jsonl"
  opus_pool_path: "data/processed/opus_train_pool.jsonl"
  candidate_pool_size_per_pair: 2000
  opus_train_size_per_pair: 20000

loop:
  iterations: 5
  k_select_per_pair: 500
  synth_per_selected: 2
  keep_prev_synth: true

hardness:
  signal: "self_nll"         # "forced_nll_ref" | "self_nll" | "qe_kiwi"
  normalize_by_len: true
  quotas: "per_pair"         # or "global_with_min_quota"

teacher:
  backend: "vllm"
  model: "YOUR_TEACHER_MODEL"
  max_tokens: 512
  temperature: 0.2

student:
  base_model: "YOUR_STUDENT_BASE"
  init_checkpoint: null      # or path to prior
  generation_max_tokens: 512

sft:
  engine: "llamafactory_apptainer"
  apptainer_image: "/path/to/llamafactory.sif"
  output_root: "runs/"
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  learning_rate: 2e-5
  lora: true

evaluation:
  comet_project_root: "/home/stroshi/parallel_sequential/standalone/parallel-sequential-core/evaluation"
  generation_utils_root: "/home/stroshi/parallel_sequential/standalone/parallel-sequential-core/src"
  ...

<slurm configs like in sweeps in main project>
```

---

## 3) Data schema (critical for selection + provenance)

Every dataset row is JSONL with at least:

```json
{
  "id": "flores:devtest:eng_Latn-deu_Latn:000123",
  "src_lang": "eng_Latn",
  "tgt_lang": "deu_Latn",
  "src_text": "...",
  "tgt_text": "...",                 // reference if available, or synthetic label
  "split": "train|candidate|devtest",
  "provenance": {
    "source": "flores|opus100|synthetic",
    "iter": 1,
    "strategy": "static_bt|active_variant|...",
    "parent_id": "opus:...:00456"     // if synthetic derived
  }
}
```

---

## 4) Integration points (reuse your existing code)

### A) Evaluation 
  `/home/stroshi/parallel_sequential/standalone/parallel-sequential-core/evaluation`

### B) Generation wrappers

Reuse
`/home/stroshi/parallel_sequential/standalone/parallel-sequential-core/src`

### C) SFT runner (LlamaFactory via apptainer)

Implement `src/sft/llamafactory_runner.py` that:

* writes LlamaFactory dataset format (whatever you standardize on)
* writes a per-iteration LlamaFactory config YAML
* runs apptainer:

  * binds `data/iterations/iter_k/` into the container
  * writes checkpoint to `runs/iter_k/`

Expose:

```python
def run_sft(train_jsonl: str, base_model: str, out_dir: str, cfg: dict) -> str:
    # returns path to trained checkpoint
```

---

## 5) Iteration loop: what the agent should implement first

### `src/orchestrator/run_iteration.py`

Single command:

```bash
python -m orchestrator.run_iteration --config configs/exp_base.yaml --iter 0
```

**Pseudo-steps inside:**

1. Load config + set seeds
2. Resolve student checkpoint for this iter
3. Build candidate pool for language pairs (from OPUS pool, not FLORES devtest)
4. Run student translations on candidate pool → `student_outputs.jsonl`
5. Compute hardness scores:

   * **self_nll**: compute NLL of student’s generated output (length-normalized) Sould prototype the API only
   * **forced_nll_ref**: compute NLL on known references (if candidate pool has refs)  Sould prototype the API only
   * **qe_kiwi**: COMETKiwi scores (if enabled)  Can implement already!
6. Select top-k:

   * per pair quotas by default (`k_select_per_pair`)
   * optionally global selection with min-per-pair
7. Generate synthetic data for selected IDs:

   * strategy = `active_variant` (see below)
   * generate `{src_variant, tgt_translation}` pairs

   Should only draft scenario config (as in main project for now) and implement some API points, will decide later on details.
8. Filter:

   * langid checks on src/tgt (lightweight)
   * length ratio bounds
   * dedup by hash/simhash
9. Merge train set:

   * base OPUS subset + (optional previous synth) + new synth
10. Run LlamaFactory SFT (apptainer)
11. Evaluate on FLORES devtest (COMET/COMETKiwi + sacreBLEU)
12. Write `metrics.json` + append to `runs/metrics.csv`

---

## 6) “Hard example” signals for MT (implement in this order)

### Tier 1 (cheap, reliable) — implement first

1. **self_nll (student uncertainty)**

   * Generate translation; compute token NLL of that translation under the student
   * Use length-normalized NLL to avoid long-sentence bias
     Output: `score = avg_nll`

2. **forced_nll_ref** (if refs are available in the candidate pool)

   * Compute NLL of reference translation (teacher-forced)
     Output: `score = avg_nll(ref)`

### Tier 2 (stronger but costlier) — implement after you have Tier 1 stable

3. **COMETKiwi QE score**

   * Select low-QE examples, but be careful: QE models can be biased by style/length.

**Agent acceptance test:** correlation sanity checks per pair:

* self_nll should correlate negatively with BLEU/COMET on a held-out slice (not FLORES devtest).

---

## 7) Synthetic strategies to compare (make these pluggable)

Implement in `src/generation/synth_strategies.py`:

### Static baselines

* `static_bt`: back-translation (src→pivot→tgt or tgt→src→tgt)
* `static_paraphrase_src`: paraphrase source then translate

### Student-aware (active)
TO BE DONE LATER
* `active_variant_minimal_pairs`:

  * prompt teacher: “Create 2 variants of the source that preserve meaning but stress translation difficulty: named entities, numbers, syntactic reorderings, idioms, morphology, punctuation… Provide translation.”
  * returns 2 `(src_variant, tgt)` per selected sample

* `active_error_exploit` (optional later):

  * include student translation in prompt; ask teacher to generate “nearby” examples that trigger similar errors and provide correct translation.

**Filtering is essential here** (dedup + length ratio + langid), otherwise active generation can become noisy fast.

---

## 8) What the agent should deliver as milestones

### Start: together with agent, draft the skeleton, and set up UV environments if needed / download and verify LlamaFactory

### Milestone M0: Dataset prep + frozen eval

* `flores_devtest.jsonl` created and frozen
* OPUS pool prepared and subset sampling works (per-pair sizes)

### Milestone M1: SFT baseline (no synthetic)

* LlamaFactory apptainer run works end-to-end
* FLORES devtest evaluation works (COMET + sacreBLEU)
* Run outputs saved under `runs/iter_00/`