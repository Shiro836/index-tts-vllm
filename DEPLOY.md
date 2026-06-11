# index-tts-vllm-fast

Fork of `index-tts-vllm` focused on inference latency, VRAM footprint and boot
reliability. Runs on **vLLM 0.22.1** (was 0.10.2) in its own conda env
(`index-tts-vllm-fast`). Drop-in replacement for `index-tts.service` on the
**same port 5113** — the unit has `Conflicts=index-tts.service`, so starting
either service automatically stops the other; switching back and forth is a
single systemctl command and no consumer needs to change.

## What changed vs the original

Speed:
- **vLLM 0.10.2 → 0.22.1** — the GPT decode was per-step-overhead-bound
  (~250 tok/s for a 340M model); newer engine has a much leaner bs=1 loop.
- **Reference (speaker) conditioning cache** — w2v-bert, campplus, semantic
  codec, length regulator and the GPT conditioning latent are computed once
  per reference wav (LRU keyed on path+mtime+size) instead of per request
  (~0.3-0.4s saved on every request). The duplicate w2v-bert forward for the
  emotion reference (same file) is also gone.
- **Concurrent sentence generation** — all sentences of a request decode in
  vLLM simultaneously; s2mel/BigVGAN of finished sentences overlaps decodes
  still in flight. Long multi-sentence requests no longer pay
  sum-of-decodes.
- `merge_emovec` / `get_conditioning` hoisted out of the sentence loop;
  FINAL_ONLY output kind (no per-token asyncio streaming).

VRAM (target ~12-13GB total vs ~22.8GB):
- **Qwen emotion engine is lazy** (`--qwen_emo_mode lazy`): not built until the
  first emo_text request (prod has sent zero in 13k+ requests). −3.2GB.
- **Explicit KV budget** `--kv_cache_memory_bytes 2500000000` (~20k tokens,
  ~11 max-length sentences in flight) instead of 6GB worth of utilization
  fraction. −3.5GB.
- **w2v-bert + campplus on CPU** (`--ref_device cpu`) — they only run on
  speaker-cache misses. −2.5GB.
- Trimmed cudagraph capture sizes, CFM estimator caches 8192→4096 positions,
  `expandable_segments` allocator. −1GB-ish.

Boot reliability:
- `index-tts-updated.service`: starts after the other GPU services
  (`llm_text.service`, `style.service`), `ExecStartPre=wait_for_vram.sh`
  blocks until enough VRAM is actually free, `RestartSec=30s` (a dying
  EngineCore needs seconds to release VRAM; the old 5s raced it),
  `StartLimitIntervalSec=0` (never stop retrying), `TimeoutStartSec=900`.
- The explicit KV budget makes the vLLM allocation independent of boot order.

vLLM port notes:
- `patch_vllm.py` no longer vendors `GPUModelRunner._prepare_inputs`; it wraps
  the original and shifts the model-input positions afterwards (slot
  mapping/attention metadata keep true positions). Should survive future
  upgrades.
- `index_tts_gpt2_vllm_v1.py` ported to the 0.22 model API
  (`embed_multimodal`/`embed_input_ids`/`compute_logits` without sampler,
  `get_placeholder_str`, new dummy-builder signature).
- `enable_mm_embeds=True` is required on 0.22 for the embeds prompt.
- transformers 5.x: vendored maskgct `LlamaConfig(...)` calls converted to
  keyword args.

## Deploy

```bash
# 0) build fake_dns.so if missing (it is not committed)
gcc -shared -fPIC fake_dns.c -o fake_dns.so -ldl

# 1) install the unit
sudo cp index-tts-updated.service /etc/systemd/system/
sudo systemctl daemon-reload

# 2) switch over (Conflicts= stops the old service automatically)
sudo systemctl start index-tts-updated.service
journalctl -u index-tts-updated.service -f

# 3) once validated, make it the boot default
sudo systemctl disable index-tts.service
sudo systemctl enable index-tts-updated.service
```

Rollback (one command, the Conflicts= relation stops the new one for you):

```bash
sudo systemctl start index-tts.service
# and if it should stay that way across reboots:
sudo systemctl enable index-tts.service && sudo systemctl disable index-tts-updated.service
```

The old repo/env/checkpoints are untouched (checkpoints are shared via
symlink, read-only).

## Validation checklist (partially run live on 2026-06-10, samples in outputs/)

- [x] Service boots; vLLM logs show `kv_cache_memory` honored and the model
      loading as `GPT2TTSModel`.
- [x] Request returns real speech (477 tok/s decode, RTF 0.17 vs prod 0.31); listen and compare
      against prod output for the same text+voice (positions patch regression
      would sound garbled — this is the critical check after the 0.22 port).
- [ ] Multi-sentence request: total time ≈ slowest sentence + s2mel, not the
      sum; audio order and 200ms gaps intact.
- [x] Second request with the same voice logs no `reference cache miss` and
      is ~0.3s faster.
- [x] `nvidia-smi`: total footprint ~8.7GB (even better than the 12-13GB target).
- [ ] emo_control_method=3 request (optional): Qwen engine lazy-initializes,
      then responds normally.
- [ ] `kill -9` the EngineCore once: service restarts cleanly after ~30s
      without manual help.

## Streaming endpoint

`POST /tts_stream` — same request fields as `/tts_url`. Responds with NDJSON,
one line per finished sentence chunk:

```json
{"text": "...", "start": 0.0, "end": 4.34, "sampling_rate": 22050, "audio": "<b64 standalone wav>"}
```

terminated by `{"done": true}` (or `{"error": "..."}` on mid-stream failure —
the HTTP status is already 200 by then, so consumers must check for it).
Chunks include the 200ms inter-sentence silence; decoding and concatenating
all chunks reproduces the `/tts_url` output exactly. `start`/`end` bound the
speech within the concatenated stream (silence excluded). First chunk arrives
after the first sentence is synthesized (~1.3s) while the rest keep decoding;
lower `max_text_tokens_per_sentence` for finer chunks / faster first audio.
Client disconnect aborts the in-flight vLLM decodes.
