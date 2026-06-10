"""vLLM integration patches for the IndexTTS GPT2 model (vLLM 0.22.x).

Two things happen here:
1. The custom GPT2TTSModel is registered under the architecture name found in
   the converted checkpoint's config.json ("GPT2InferenceModel").
2. GPUModelRunner._prepare_inputs is wrapped so that, for the TTS model only,
   the model-input position ids are shifted by -(prompt_len - 1) per request.
   The TTS GPT2 indexes its mel positional embedding with these positions:
   the first generated mel token must see position 1 (position 0 belongs to
   the start-mel token whose positional embedding is baked into the prompt
   embeds). Prompt-token positions become <= 0 and are clamped to 0 inside the
   model, where embedding row 0 is pinned to zeros.

   The wrapper runs AFTER the original _prepare_inputs, so KV slot mapping and
   attention metadata are built from the true positions; only the tensor the
   model consumes is shifted. This replaces the old approach of vendoring the
   entire _prepare_inputs body, which broke on every vLLM upgrade.
"""

import importlib

import numpy as np
import torch
from packaging import version

vllm_version = version.parse(importlib.import_module("vllm").__version__)

from vllm import ModelRegistry
from indextts.gpt.index_tts_gpt2_vllm_v1 import GPT2TTSModel

ModelRegistry.register_model("GPT2InferenceModel", GPT2TTSModel)
print("✅  Registered GPT2TTSModel with vLLM")


from vllm.v1.worker.gpu_model_runner import GPUModelRunner

_orig_prepare_inputs = GPUModelRunner._prepare_inputs


def _prepare_inputs(self, scheduler_output, num_scheduled_tokens, *args, **kwargs):
    ret = _orig_prepare_inputs(self, scheduler_output, num_scheduled_tokens, *args, **kwargs)

    model = self.get_model()
    if isinstance(model, GPT2TTSModel):
        total_num_scheduled_tokens = int(scheduler_output.total_num_scheduled_tokens)
        num_reqs = self.input_batch.num_reqs

        offsets_np = np.empty(num_reqs, dtype=np.int64)
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            req_state = self.requests[req_id]
            offsets_np[i] = -(req_state.num_prompt_tokens - 1)

        offsets = torch.from_numpy(offsets_np).to(self.positions.device)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]
        self.positions[:total_num_scheduled_tokens].add_(offsets[req_indices_gpu])

    return ret


GPUModelRunner._prepare_inputs = _prepare_inputs
print("✅  GPUModelRunner._prepare_inputs wrapped (TTS position offsets)")
