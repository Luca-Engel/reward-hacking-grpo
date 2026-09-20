# GPU_COMPAT — GPU stack pins, risks and fallbacks

Written on 2026-09-20 on a machine with **no GPU**: nothing below was run against vLLM or a CUDA device. This is
"settled on paper" (DESIGN §10 pre-mortem #2, §11); the first real GPU minutes (`scripts/smoke.sh`) exist to falsify it.

Tags used on every claim:

* **[verified from docs]** — read in official documentation, a package's PyPI metadata, or the *published source* of
  the pinned package version (`trl 1.13.0`, `vllm 0.28.0`; downloaded and read, not installed on a GPU).
* **[verified from issue tracker]** — read in a GitHub issue/PR of `huggingface/trl` or `vllm-project/vllm`.
* **[verified by CPU run]** — executed in a throwaway CPU environment (`uv run --no-project --isolated --with ...`).
* **[unverified]** — reasoning, memory of vendor documentation, or an assumption that only a GPU can confirm.

## 1. Pins

Exactly the lines of `requirements-gpu.txt` (`tests/test_trl_mapping.py` asserts that this table and the file list the
same `package==version`; the `## 1` heading bounds that check).

| pin | why | tag |
|---|---|---|
| `torch==2.13.0` | `vllm 0.28.0` hard-pins `torch==2.13.0` (also `torchvision==0.28.0`, `torchaudio==2.11.0`, `flashinfer-python==0.6.16.post3`, `compressed-tensors==0.17.0`) | verified from docs (PyPI metadata) |
| `vllm==0.28.0` | newest vLLM inside TRL 1.13.0's own range: its `vllm` extra is `vllm<=0.28.0,>=0.19.1`. The TRL vLLM-integration page (main) says "TRL currently only supports vLLM versions from `0.19.1` to `0.29.0`"; the PyPI metadata of the release we pin is the stricter, so we follow it. vLLM 0.29.0 exists but is outside that metadata. ≥0.8.5 / Qwen3 (DESIGN §2.1) is satisfied by a wide margin | verified from docs |
| `transformers==5.15.0` | `vllm 0.28.0` needs `transformers>=5.5.3` (release notes: "bumped Transformers to 5.15.0", i.e. the version vLLM itself was tested with); `trl 1.13.0` needs `>=4.56.2`; DESIGN's `>=4.51` is met. **Major-version note:** this is transformers 5, not 4.x (renamed `torch_dtype`→`dtype`, `warmup_ratio` removed) | verified from docs |
| `trl==1.13.0` | latest on PyPI (2026-09-10); the documentation of `main` describes colocate as the default mode, `vllm_enable_sleep_mode`, and the pinned signature was dumped from exactly this version (`docs/trl_grpoconfig_fields.json`, 189 `GRPOConfig` fields). Downside: 10 days old at pinning, so little soak time (see fallback 3) | verified from docs / by CPU run |
| `peft==0.20.0` | `trl` needs `peft>=0.13.0`; no upper bound anywhere. 0.20.0 (2026-07-28) predates transformers 5.15.0 by two weeks and had been out for a month at pinning; 0.21.0 (2026-09-15) is too new to trust, and the last TRL-issue report of a working stack used 0.19.1 | verified from docs / issue tracker |
| `accelerate==1.14.0` | `trl` needs `>=1.4.0`; 1.14.0 (2026-06-11) is the accelerate used by the working stack reported in trl#6688; 1.15.0 (2026-09-09) is too new | verified from issue tracker |
| `datasets==5.0.1` | equals the `datasets` in `uv.lock` (the CPU box that builds `data/processed`), so tokenising/filtering code sees the same library; `trl` needs `>=4.7.0` | verified from docs |

Resolution check: `uv pip compile requirements-gpu.txt --python-platform x86_64-manylinux_2_28 --python-version 3.12`
resolves these seven pins together with no conflict (transformers 5.15.0, tokenizers 0.22.2, huggingface-hub 1.32.0,
triton 3.7.1, xgrammar 0.2.7, flashinfer-python 0.6.16.post3, `nvidia-*-cu13` wheels) **[verified by CPU run]**. That
proves the metadata is consistent, not that the software works. Python 3.12 is inside vLLM's `>=3.10,<3.15` **[verified
from docs]**.

Transformers 5 also runs on the box for our own code: `rhg.data.prompts.render_chat` calls
`tokenizer.apply_chat_template(..., tokenize=False, ...)`; the `network`-marked test in `tests/test_prompts.py` that
compares our rendering with the real tokenizer must be run once on the box after setup (`uv run pytest -m network`).

## 2. Install order, CUDA wheels, driver

1. `uv venv --python 3.12 .venv-gpu && source .venv-gpu/bin/activate` (or reuse the project venv; the GPU stack is not in
   `uv.lock`).
2. `uv pip install -r requirements-gpu.txt` — **one command**, so a single resolver run lets vLLM's `torch==2.13.0` decide
   torch. Do not install torch first from another index and vLLM afterwards.
3. `uv pip freeze > results/bench/pip_freeze.txt` (records the transitive versions; the `libs` block of the run manifest
   records the direct ones).
4. Check: `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"`.

**Which CUDA build you get.** PyPI `torch==2.13.0` is the CUDA 13.0 build (its metadata pins `cuda-toolkit==13.0.3`,
`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`; the resolved lock contains only `*-cu13` packages) **[verified from docs / by
CPU run]**; the `v0.28.0` release notes say the default vLLM wheel uses CUDA 13.0 with CUDA 12.9 wheels as an
alternative **[verified from docs]**. The vLLM installation page still says the default PyPI wheels are compiled with
CUDA 12.9 — the two statements disagree and the page is probably stale; treat the wheel as CUDA 13 and confirm with
`torch.version.cuda` **[unverified until the install]**.

**Driver.** CUDA 13.0 needs a Linux driver ≥ 580.65.06 (NVIDIA CUDA 13.0 documentation, via search summary) **[verified from
docs, secondary]**. Rented 4090 hosts frequently run older drivers (535/550/570) **[unverified]**. Check `nvidia-smi`
before installing; if the driver is < 580 use the **cu129 variant**:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple torch==2.13.0
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
uv pip install --no-deps -r requirements-gpu.txt   # then the remaining pins, unchanged
```

`torch 2.13.0+cu129`/`+cu126`/`+cu130` wheels exist on the PyTorch index and the release asset
`vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` exists **[verified from docs]**; the install order above and the
minimum driver for CUDA 12.9 (≈575; CUDA 12.x minor-version compatibility allows ≥525 with feature limits) are
**[unverified]**. Keep the requirements file byte-identical (its sha256 is in the preregistration freeze); record the
cu129 deviation in `DEVIATIONS.md` instead.

RTX 4090 = compute capability 8.9; vLLM's minimum is 7.5 **[verified from docs]**. Whether the torch 2.13 / vLLM 0.28
kernels (incl. FlashInfer JIT, which needs `nvcc`, provided by the `nvidia-cuda-nvcc` wheel in the lock) run on sm_89
is **[unverified]** until the smoke run.

## 3. Known issues

* **GRPO + colocate + PEFT hangs (trl#3671).** Reported "only on multi-GPU setups" with vLLM tensor parallelism and no
  NVLink (an NCCL race between DDP's LoRA all-reduce and vLLM's TP collectives). Closed by PRs #6139/#6187/#6196
  (June 2026), which add a barrier that runs only when `is_peft_model(model) and tensor_parallel_size > 1`. **We run
  one process, `vllm_tensor_parallel_size=1`**, so neither the deadlock nor its workaround code path is reached
  **[verified from issue tracker]**. The fixes are in 1.13.0 (released after them) **[verified from docs]**. How a
  5-minute smoke run would reveal a *different* hang on one GPU: step 1 never completes (no `steps.jsonl` line; the
  stall watchdog, `run.step_timeout_s`, exits 75); a hang inside `LLM(...)` construction shows no output after "loading
  model"; a hang after step 1 points to `sync_weights`. Use `--set run.step_timeout_s=300` for the smoke run and read
  `status.json.phase`.
* **Sleep mode + colocate (trl#5312, #5142)**: after a refactor, weights were not re-synchronised after waking; closed
  (July 2026) and the 1.13.0 source re-pushes all weights after each sleep (level 2 discards them) **[verified from
  issue tracker / docs]**.
* **NaN vLLM logprob crash (trl#6166)**, closed June 2026, fixed before 1.13.0. Irrelevant while
  `vllm_importance_sampling_correction=False` **[verified from issue tracker]**.
* **bf16 merge/unmerge drift (trl#6688, open, TRL 1.9.2)**: repeated `merge_adapter()`/`unmerge_adapter()` with a bf16
  base model, no answer yet. See §4 for why this matters here **[verified from issue tracker; effect size unverified]**.
* **Importance-sampling ratio biased under top_p/top_k truncation (trl#6789, open)**: not applicable, correction off and
  top_p=1, top_k off **[verified from issue tracker]**.
* **vLLM-side `Engine core initialization failed` (trl#3632)** class of errors is almost always insufficient free
  memory at engine start: lower `vllm_gpu_mem_util` (fallback 1) **[verified from issue tracker, old]**.
* The pinned `GRPOConfig` has `max_completion_length` but **no prompt-length field** (TRL's docs say the legacy
  `max_prompt_length` was removed), and `grpo_trainer.py` contains no prompt truncation **[verified from docs / by CPU
  run]**: `rhg.train.trl_trainer` must drop or reject prompts above `grpo.max_prompt_tokens`.

## 4. How TRL syncs LoRA weights to vLLM in colocate mode  [verified from docs: `trl/generation/vllm_generation.py` 1.13.0]

vLLM is built inside the trainer process (`LLM(model=model.name_or_path, distributed_executor_backend="external_launcher",
gpu_memory_utilization=..., max_model_len=vllm_max_model_length, max_num_seqs=per_device_train_batch_size × TP ×
steps_per_generation, max_num_batched_tokens=4096, logprobs_mode="processed_logprobs", ...)`) **without `enable_lora`**:
vLLM never sees an adapter. Before each generation batch `sync_weights()` runs `model.merge_adapter()`, walks
`named_parameters()` (stripping `base_model.model.` and `.base_layer`, skipping `lora_*` and `original_module`), calls
`llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])` for **every** base parameter,
then `model.unmerge_adapter()` and `llm.reset_prefix_cache()`. So per training step: one full-model (≈3.2 GiB bf16) copy
into the engine plus a merge and an unmerge of 34.9 M LoRA parameters' worth of matmuls (`t_sync` in BUDGET §2). An
adapter-only path (PR trl#6007) is open and **not** in 1.13.0.

Two consequences to keep in mind **[unverified reasoning; smoke-testable]**:

1. vLLM samples from the *merged bf16* weights. A LoRA delta smaller than about 2⁻⁹ of a base weight is partly rounded
   away when added in bf16, so early in training (lr 7e-5, B initialised to 0) the sampling policy is a slightly
   quantised version of the policy being trained. This is the mismatch TRL's importance-sampling correction measures.
   We leave that correction off (see `ESTIMAND_NOTES` in `trl_config.py` for why); the smoke run should include one
   2–3 step diagnostic run with `extra={"vllm_importance_sampling_correction": True}` and read the logged
   `importance_sampling_ratio` statistics; a large spread means we must revisit the decision (SPEC_DEVIATIONS 10).
2. Repeated merge/unmerge in bf16 makes the frozen base weights drift by rounding. Cheap check for the smoke run: keep a
   CPU copy of one base `q_proj` weight before training and compare after 5 steps (`max |Δ|` should be ≈ 0 to bf16
   resolution and not grow with steps).

## 5. Expected memory on a 24 GB RTX 4090  [unverified; arithmetic in `rhg.train.gpu_memory`]

`uv run python -m rhg.train.gpu_memory` prints the table for the configured batch/length (micro-batch 4, prompt ≤768,
completion ≤1024, `vllm_gpu_mem_util=0.35`, gradient checkpointing). Default output, 24.00 GiB nominal (a 4090 has
24564 MiB, **[unverified]**):

| component | GiB |
|---|---|
| trainer weights (bf16, 1,720,574,976 parameters) | 3.20 |
| LoRA r=32 all-linear (34,865,152 params × 16 B: fp32 param+grad+2 Adam) | 0.52 |
| activations with gradient checkpointing (checkpoints + one live layer) | 1.29 |
| logits of the completion tokens (bf16 + grad + fp32 row copy) | 3.48 |
| allocator fragmentation (assumed 10%) | 0.85 |
| vLLM budget = 0.35 × 24 (weights 3.20 + assumed overhead 1.50 + KV cache 3.70) | 8.40 |
| CUDA context (assumed) | 1.00 |
| **expected peak** | **≈18.7** |

KV cache per token = 2 × 28 layers × 8 kv-heads × 128 × 2 B = 114,688 B (112 KiB); 3.70 GiB holds ≈34.6k tokens, i.e.
only ≈19 sequences of the maximum 1792 tokens at once, so vLLM will run a step's 128 rollouts in waves. That costs
generation throughput, not correctness (the bench measures it). All architecture numbers come from the model's
`config.json` (28 layers, hidden 2048, 16 q / 8 kv heads, head_dim 128, MLP 6144, vocab 151936, tied embeddings), fetched
without downloading weights **[verified from docs]**; overhead, context, fragmentation and the logits peak are
assumptions. `scripts/smoke.sh` compares the measured peak with this estimate (`--measured-peak-gib`).

## 6. Ranked fallbacks

**Order of use** (each step only after the previous fails in the smoke run; none touches the estimand except where marked):

1. **Lower `grpo.vllm_gpu_mem_util`** 0.35 → 0.30 → 0.25 (`--set grpo.vllm_gpu_mem_util=0.30`), and/or a smaller micro-batch
   (`build_grpo_config(..., micro_batch_cap=2)`: same rollouts per step, half the logits memory, more accumulation steps).
   Symptoms: CUDA OOM in the first backward, or `Engine core initialization failed` at start. Below ≈0.20 the KV cache
   in the estimator no longer holds a few sequences. This changes `config_hash` only through `vllm_gpu_mem_util`
   (a hyperparameter in the freeze) — decide it *before* the freeze.
2. **vLLM sleep mode:** `build_grpo_config(cfg, extra={"vllm_enable_sleep_mode": True})`. TRL sleeps the engine (level 2: weights
   and KV cache discarded) during the optimizer step and re-pushes all weights before each generation, so the KV budget
   need not coexist with training activations, at the price of a full weight push plus KV re-allocation per step
   (`t_sync` up) **[verified from docs]**.
3. **Alternative pinned combination** (already resolved with `uv pip compile`, and the exact set reported working in
   trl#6688 on an RTX 6000 Ada with colocate, `+cu129` wheels): `torch==2.11.0`, `vllm==0.20.2`, `transformers==5.14.1`,
   `trl==1.9.2`, `peft==0.19.1`, `accelerate==1.14.0`, `datasets==5.0.1` (vLLM 0.20.2 also ships a `+cu129` wheel)
   **[verified from issue tracker + by CPU resolve]**. Switching requires re-running the field dump (command in the
   JSON's `_about`) because TRL 1.9.2's `GRPOConfig` differs, updating `requirements-gpu.txt` **before** the freeze, and
   re-running `tests/test_trl_mapping.py`. A second, newer fallback inside TRL 1.13.0's vLLM range is `vllm==0.27.1`
   (`torch==2.13.0`, same everything else) **[verified by CPU resolve of the metadata; unverified in use]**.
4. **Minimal custom GRPO loop (last resort, specification only; not implemented).** Drop TRL and write ~200 lines: load
   Qwen3-1.7B in bf16 with a PEFT LoRA (r=32, α=32, all-linear) and run a **separate, non-colocated** vLLM engine for
   sampling (`enable_lora=True, max_lora_rank=32`, sync by saving the adapter to disk each step and passing a
   `LoRARequest` with a fresh `lora_int_id`/`load_inplace=True`, which is what `rhg.eval.generate.VLLMGenerator` already
   does) or, if memory allows, HF `generate`; per step take the driver's schedule (`prompts_per_step` distinct prompts),
   sample `gens_per_prompt` completions each (T=1.0, top_p=1, top_k off, `max_tokens=1024`, record truncation), compute
   rewards with `rhg.train.rollout_io.make_reward_fn`, form group-normalised advantages `(r − mean)/(std + 1e-4)`, run
   micro-batched forward/backward of the token-level policy-gradient loss `−A·logπ(token)` summed over completion tokens
   and divided by the constant `rollouts × max_completion_tokens` (Dr. GRPO), with `num_iterations=1` (no ratio, no
   clip, no KL, `beta=0`), clip the gradient norm at 1.0, take one AdamW step (constant lr 7e-5, no weight decay), and
   report the `steps.jsonl` fields through `ctx.end_step`. Correctness checks: on the mock reward with a tiny model the
   loss must equal the closed form on a hand-made batch, and a second run with the same seed must reproduce the prompt
   schedule exactly.

Not a fallback: `vllm_mode="server"` needs a second CUDA device (TRL docs: server and trainer "must run on separate CUDA
devices") **[verified from docs]**, so it does not apply to a single 4090.

## 7. Open items this document cannot close (owner: smoke run, TRL trainer and run plan)

* Whether the torch 2.13 + vLLM 0.28 wheels initialise on the rented box's driver (§2) and on sm_89.
* Peak memory versus §5; the KV-cache budget accounting when the trainer already holds memory in the same process.
* `t_sync`/`t_gen` per step (BUDGET §2) — nothing here is a throughput claim.
* Whether `merge_adapter` in bf16 changes the sampling policy measurably (§4).
