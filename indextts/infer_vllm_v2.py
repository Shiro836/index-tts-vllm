import asyncio
import os
import random
import re
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List
import uuid

import librosa
import torch
import torchaudio
# from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor
from transformers import AutoTokenizer
from modelscope import AutoModelForCausalLM
import safetensors
from loguru import logger

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model_vllm_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.front import TextNormalizer, TextTokenizer

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

import torch.nn.functional as F

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


def _half_with_fp32_layernorm(model):
    """Convert model to FP16 but keep LayerNorm in FP32 for numerical stability."""
    model.half()
    for module in model.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
            module.float()
            module.register_forward_pre_hook(
                lambda mod, inp: tuple(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in inp)
            )
            module.register_forward_hook(
                lambda mod, inp, out: out.half() if isinstance(out, torch.Tensor) and out.is_floating_point() else out
            )
    return model


class IndexTTS2:
    def __init__(
        self, model_dir="checkpoints", is_fp16=False, device=None, use_cuda_kernel=None, gpu_memory_utilization=0.25, qwenemo_gpu_memory_utilization=0.10, offload_device=None,
        kv_cache_memory_bytes=None, qwen_emo_mode="lazy", ref_device=None, ref_cache_size=8, mel_workers=1,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            kv_cache_memory_bytes (int): explicit KV-cache budget for the GPT vLLM engine. Overrides
                gpu_memory_utilization, making the allocation independent of what else is resident on
                the GPU at startup (avoids boot-order races between services).
            qwen_emo_mode (str): "lazy" (default) builds the Qwen emotion engine on first emo-text
                request, "eager" builds it at startup, "disabled" never builds it.
            ref_device (str): device for the reference-audio-only models (w2v-bert, campplus).
                E.g. "cpu" keeps ~2.5GB off the GPU; with the reference cache the forward only
                runs once per new speaker.
            ref_cache_size (int): number of reference audio files whose conditioning is kept cached.
            mel_workers (int): threads running the synchronous GPU tail of a sentence
                (GPT forward, s2mel, BigVGAN) and reference computation. Even 1 keeps the
                event loop responsive (the GIL is released during CUDA ops); more workers
                let concurrent requests overlap kernel-launch overhead, but all threads
                share the default CUDA stream (kernels still serialize) and activation
                VRAM grows per concurrent worker — measure before raising.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
            logger.info(">> Be patient, it may take a while to run in CPU mode.")

        if offload_device:
            self.gpt_device = self.device
            self.device = offload_device
        else:
            self.gpt_device = self.device

        cfg_path = os.path.join(model_dir, "config.yaml")
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        vllm_dir = os.path.join(model_dir, "gpt")
        engine_kwargs = dict(
            model=vllm_dir,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            # prompts carry unique per-request conditioning embeds, so block-level
            # prefix caching can never hit; it only adds mm-hashing overhead
            enable_prefix_caching=False,
            max_num_seqs=16,
            # vLLM >= 0.11 rejects multimodal *embedding* inputs (our conditioning
            # prompt) unless explicitly enabled
            enable_mm_embeds=True,
            # every request's embeds are unique; caching processed mm items only
            # burns CPU RAM on dead entries
            mm_processor_cache_gb=0,
        )
        if kv_cache_memory_bytes:
            engine_kwargs["kv_cache_memory_bytes"] = int(kv_cache_memory_bytes)
        try:
            from vllm.config import CompilationConfig
            from vllm.config.compilation import CUDAGraphMode
            # decode batch is tiny (concurrent sentences); capturing graphs up to
            # the default 256 wastes several hundred MB of VRAM.
            # FULL_AND_PIECEWISE captures whole decode steps as single CUDA graphs —
            # this 340M model is kernel-launch-bound at bs=1, so eliminating per-step
            # launches is the main remaining decode speedup (costs a bit more VRAM,
            # numerically identical).
            engine_kwargs["compilation_config"] = CompilationConfig(
                cudagraph_capture_sizes=[1, 2, 4, 8, 16],
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            )
            # overlap CPU scheduling of step N+1 with GPU execution of step N
            engine_kwargs["async_scheduling"] = True
        except Exception:
            logger.warning(">> CompilationConfig unavailable; using default cudagraph capture sizes")
        engine_args = AsyncEngineArgs(**engine_kwargs)
        indextts_vllm = AsyncLLM.from_engine_args(engine_args)

        self.qwen_emo = None
        self.qwen_emo_mode = qwen_emo_mode
        self._qwen_emo_dir = os.path.join(self.model_dir, self.cfg.qwen_emo_path)
        self._qwen_emo_gpu_memory_utilization = qwenemo_gpu_memory_utilization
        self._qwen_emo_lock = asyncio.Lock()
        if qwen_emo_mode == "eager":
            self.qwen_emo = QwenEmotion(
                self._qwen_emo_dir,
                gpu_memory_utilization=qwenemo_gpu_memory_utilization,
            )

        self.gpt = UnifiedVoice(indextts_vllm, **self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.gpt_device)
        if self.is_fp16:
            _half_with_fp32_layernorm(self.gpt)
        self.gpt.eval()
        logger.info(f">> GPT weights restored from: {self.gpt_path}")

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load

                anti_alias_activation_cuda = load.load()
                logger.info(f">> Preload custom CUDA kernel for BigVGAN {anti_alias_activation_cuda}")
            except Exception as ex:
                traceback.print_exc()
                logger.info(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            # "facebook/w2v-bert-2.0"
            os.path.join(self.model_dir, "w2v-bert-2.0")
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat),
            os.path.join(self.model_dir, "w2v-bert-2.0")
        )
        self.ref_device = ref_device or self.device
        self.semantic_model = self.semantic_model.to(self.ref_device)
        if self.is_fp16:
            _half_with_fp32_layernorm(self.semantic_model)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.ref_device)
        self.semantic_std = self.semantic_std.to(self.ref_device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = os.path.join(self.model_dir, "semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        if self.is_fp16:
            _half_with_fp32_layernorm(self.semantic_codec)
        self.semantic_codec.eval()
        logger.info('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        if self.is_fp16:
            _half_with_fp32_layernorm(self.s2mel)
        # real upper bound is prompt_condition + 1.72 * codes ≈ 2.4k positions at the
        # 120-token sentence cap; 8192 wasted a few hundred MB of preallocated cache
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=4096)
        self.s2mel.eval()
        logger.info(f">> s2mel weights restored from: {s2mel_path}")

        # load campplus_model
        # campplus_ckpt_path = hf_hub_download(
        #     "funasr/campplus", filename="campplus_cn_common.bin", cache_dir=os.path.join(self.model_dir, "campplus")
        # )
        campplus_ckpt_path = os.path.join(self.model_dir, "campplus/campplus_cn_common.bin")
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.ref_device)
        if self.is_fp16:
            _half_with_fp32_layernorm(self.campplus_model)
        self.campplus_model.eval()
        logger.info(f">> campplus_model weights restored from: {campplus_ckpt_path}")

        bigvgan_name = self.cfg.vocoder.name
        # self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False, cache_dir=os.path.join(self.model_dir, "bigvgan"))
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(os.path.join(self.model_dir, "bigvgan"))
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        if self.is_fp16:
            _half_with_fp32_layernorm(self.bigvgan)
        self.bigvgan.eval()
        logger.info(f">> bigvgan weights restored from: {bigvgan_name}")

        self.bpe_path = os.path.join(self.model_dir, "bpe.model")  # self.cfg.dataset["bpe_model"]
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        logger.info(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        logger.info(f">> bpe model loaded from: {self.bpe_path}")

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # LRU cache of everything derived from a reference wav (w2v-bert embedding,
        # s2mel prompt condition, campplus style, GPT conditioning latent). All of it
        # is a pure function of the audio file, so it only needs to be computed once
        # per speaker instead of on every request.
        self._ref_cache = OrderedDict()
        self._ref_cache_size = ref_cache_size

        # the synchronous GPU sections (sentence tail, reference computation) run
        # here instead of on the event loop, which otherwise freezes every
        # concurrent request's stream for the duration of an s2mel/vocoder pass
        self._mel_executor = ThreadPoolExecutor(max_workers=max(1, mel_workers), thread_name_prefix="mel")
        self._ref_inflight = {}

    async def _ensure_qwen_emo(self):
        if self.qwen_emo is not None:
            return self.qwen_emo
        if self.qwen_emo_mode == "disabled":
            raise RuntimeError(
                "emo_text mode requested but the Qwen emotion engine is disabled "
                "(qwen_emo_mode='disabled'); pass an explicit emotion vector instead"
            )
        async with self._qwen_emo_lock:
            if self.qwen_emo is None:
                logger.info(">> lazily initializing Qwen emotion engine (first emo-text request)...")
                stt = time.perf_counter()
                self.qwen_emo = QwenEmotion(
                    self._qwen_emo_dir,
                    gpu_memory_utilization=self._qwen_emo_gpu_memory_utilization,
                )
                logger.info(f">> Qwen emotion engine ready in {time.perf_counter() - stt:.1f}s")
        return self.qwen_emo

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        if self.dtype is not None:
            feat = feat.to(self.dtype)
        return feat

    @staticmethod
    def _ref_cache_key(audio_path):
        st = os.stat(audio_path)
        return (os.path.abspath(audio_path), st.st_mtime_ns, st.st_size)

    @torch.no_grad()
    def _compute_ref(self, audio_path):
        """Compute all conditioning derived from a reference wav (the per-speaker,
        per-request-invariant part of the old infer() preamble)."""
        audio, sr = librosa.load(audio_path)
        audio = torch.tensor(audio).unsqueeze(0)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.ref_device)
        attention_mask = inputs["attention_mask"].to(self.ref_device)
        spk_cond_emb = self.get_emb(input_features, attention_mask).to(self.device)
        spk_cond_emb_gpt = spk_cond_emb.to(self.gpt_device)

        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.to(self.device).float())
        if self.dtype is not None:
            ref_mel = ref_mel.to(self.dtype)
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(self.ref_device),
                                                 num_mel_bins=80,
                                                 dither=0,
                                                 sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        if self.dtype is not None:
            feat = feat.to(self.dtype)
        style = self.campplus_model(feat.unsqueeze(0)).to(self.device)

        prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                 ylens=ref_target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]

        # GPT conditioning latent (was recomputed inside inference_speech per sentence).
        # Length semantics (shape[-1]) intentionally match the original call sites.
        cond_lengths_gpt = torch.tensor([spk_cond_emb.shape[-1]], device=self.gpt_device)
        cond_latent = self.gpt.get_conditioning(spk_cond_emb_gpt.transpose(1, 2), cond_lengths_gpt)

        return {
            "spk_cond_emb": spk_cond_emb,
            "spk_cond_emb_gpt": spk_cond_emb_gpt,
            "prompt_condition": prompt_condition,
            "ref_mel": ref_mel,
            "style": style,
            "cond_latent": cond_latent,
        }

    async def _get_ref(self, audio_path):
        key = self._ref_cache_key(audio_path)
        entry = self._ref_cache.get(key)
        if entry is not None:
            self._ref_cache.move_to_end(key)
            return entry

        # concurrent misses for the same voice compute once; the rest await the
        # same future instead of duplicating ~0.3s of GPU work per request
        fut = self._ref_inflight.get(key)
        if fut is not None:
            return await fut

        stt = time.perf_counter()
        fut = asyncio.get_running_loop().run_in_executor(self._mel_executor, self._compute_ref, audio_path)
        self._ref_inflight[key] = fut
        try:
            entry = await fut
        finally:
            self._ref_inflight.pop(key, None)

        self._ref_cache[key] = entry
        while len(self._ref_cache) > self._ref_cache_size:
            self._ref_cache.popitem(last=False)
        logger.info(f">> reference cache miss for {audio_path} (computed in {time.perf_counter() - stt:.2f}s)")
        return entry

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between sentences.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list
    
    def _sentence_to_wav(self, codes, text_tokens, speech_conditioning_latent,
                         spk_cond_emb, emo_cond_emb_gpt, cond_lengths, emo_cond_lengths,
                         emovec, prompt_condition, ref_mel, style, verbose=False):
        """GPU tail of one sentence: GPT forward, s2mel diffusion, BigVGAN.

        Runs on the mel executor — the block has no await points, so inline it
        would pin the event loop for the whole pass. All inputs and outputs are
        per-sentence locals and the shared modules are pure forwards, so
        concurrent invocations from different requests are safe; they share the
        default CUDA stream, so GPU kernels serialize while the loop stays free
        (the GIL is released during CUDA ops).

        Returns (wav_cpu, gpt_forward_dt, s2mel_dt, bigvgan_dt).
        """
        with torch.no_grad():
            code_lens = []
            for code in codes:
                if self.stop_mel_token not in code:
                    code_len = len(code)
                else:
                    len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                    code_len = len_ - 1
                code_lens.append(code_len)
            codes = codes[:, :code_len]
            code_lens = torch.LongTensor(code_lens)
            code_lens = code_lens.to(self.device)
            if verbose:
                print(codes, type(codes))
                print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                print(f"code len: {code_lens}")

            m_start_time = time.perf_counter()
            use_speed = torch.zeros(spk_cond_emb.size(0)).to(self.gpt_device).long()
            latent = self.gpt(
                speech_conditioning_latent,
                text_tokens,
                torch.tensor([text_tokens.shape[-1]], device=self.gpt_device),
                codes,
                torch.tensor([codes.shape[-1]], device=self.gpt_device),
                emo_cond_emb_gpt,
                cond_mel_lengths=cond_lengths,
                emo_cond_mel_lengths=emo_cond_lengths,
                emo_vec=emovec,
                use_speed=use_speed,
            )
            gpt_forward_dt = time.perf_counter() - m_start_time

            latent = latent.to(self.device)
            codes = codes.to(self.device)

            with torch.amp.autocast(self.device.split(":")[0] if isinstance(self.device, str) else self.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                m_start_time = time.perf_counter()
                diffusion_steps = 25
                inference_cfg_rate = 0.7
                latent = self.s2mel.models['gpt_layer'](latent)
                S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                S_infer = S_infer.transpose(1, 2)
                S_infer = S_infer + latent
                target_lengths = (code_lens * 1.72).long()

                cond = self.s2mel.models['length_regulator'](S_infer,
                                                             ylens=target_lengths,
                                                             n_quantizers=3,
                                                             f0=None)[0]
                cat_condition = torch.cat([prompt_condition, cond], dim=1)
                vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                               torch.LongTensor([cat_condition.size(1)]).to(
                                                                   cond.device),
                                                               ref_mel, style, None, diffusion_steps,
                                                               inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                s2mel_dt = time.perf_counter() - m_start_time

                m_start_time = time.perf_counter()
                wav = self.bigvgan(vc_target).squeeze().unsqueeze(0)
                bigvgan_dt = time.perf_counter() - m_start_time
                wav = wav.squeeze(1)

            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            if verbose:
                print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
            wav = wav.cpu()

        return wav, gpt_forward_dt, s2mel_dt, bigvgan_dt

    async def infer_stream(self, spk_audio_prompt, text,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_sentence=120, **generation_kwargs):
        """Async generator yielding one finished sentence chunk at a time:

            (segment, sampling_rate, chunk)

        - segment: {"text", "start", "end"} — positions in the concatenated
          stream; start/end bound the speech itself, excluding the
          inter-sentence silence that is appended to every non-final chunk.
        - chunk: float tensor [1, N] scaled to ±32767; concatenating all
          chunks reproduces exactly what infer() returns.

        First chunk is ready after one sentence is synthesized, while the
        remaining sentences keep decoding in vLLM concurrently.
        """
        logger.info(">> start inference...")
        start_time = time.perf_counter()

        if use_emo_text:
            emo_audio_prompt = None
            emo_alpha = 1.0
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0
            if emo_text is None:
                emo_text = text
            qwen_emo = await self._ensure_qwen_emo()
            emo_dict, content = await qwen_emo.inference(emo_text)
            # logger.info(emo_dict)
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            emo_audio_prompt = None
            emo_alpha = 1.0
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0

        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0
            # assert emo_alpha == 1.0

        spk_ref = await self._get_ref(spk_audio_prompt)
        spk_cond_emb = spk_ref["spk_cond_emb"]
        spk_cond_emb_gpt = spk_ref["spk_cond_emb_gpt"]
        prompt_condition = spk_ref["prompt_condition"]
        ref_mel = spk_ref["ref_mel"]
        style = spk_ref["style"]

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector).to(self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        # the emo reference defaults to the speaker wav, in which case the speaker
        # entry is reused instead of running w2v-bert a second time on the same audio
        emo_ref = spk_ref if emo_audio_prompt == spk_audio_prompt else await self._get_ref(emo_audio_prompt)
        emo_cond_emb = emo_ref["spk_cond_emb"]
        emo_cond_emb_gpt = emo_ref["spk_cond_emb_gpt"]

        sentence_pairs = self.tokenizer.split_text_with_originals(text, max_text_tokens_per_sentence)
        if verbose:
            print("sentences count:", len(sentence_pairs))
            print("max_text_tokens_per_sentence:", max_text_tokens_per_sentence)
            print(*sentence_pairs, sep="\n")

        sampling_rate = 22050

        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0

        # request-invariant conditioning, hoisted out of the sentence loop
        # (the old code recomputed it per sentence with identical inputs)
        cond_lengths = torch.tensor([spk_cond_emb.shape[-1]], device=self.gpt_device)
        emo_cond_lengths = torch.tensor([emo_cond_emb.shape[-1]], device=self.gpt_device)
        with torch.no_grad():
            emovec = self.gpt.merge_emovec(
                spk_cond_emb_gpt,
                emo_cond_emb_gpt,
                cond_lengths,
                emo_cond_lengths,
                alpha=emo_alpha
            )

            if emo_vector is not None:
                emovec = emovec_mat.to(self.gpt_device) + (1 - torch.sum(weight_vector.to(self.gpt_device))) * emovec
                # emovec = emovec_mat

        # launch every sentence's GPT decode at once — sentences are conditioned
        # only on the reference audio, not on each other, and vLLM batches the
        # concurrent requests. s2mel/vocoder for finished sentences then overlaps
        # with the decodes still in flight.
        sent_text_tokens = []
        sent_texts = []
        for orig_text, sent_tokens in sentence_pairs:
            token_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_text_tokens.append(torch.tensor(token_ids, dtype=torch.int32, device=self.gpt_device).unsqueeze(0))
            sent_texts.append(orig_text)

        gen_tasks = [
            asyncio.create_task(self.gpt.inference_speech(
                spk_cond_emb_gpt,
                text_tokens,
                emo_cond_emb_gpt,
                cond_lengths=cond_lengths,
                emo_cond_lengths=emo_cond_lengths,
                emo_vec=emovec,
                speech_conditioning_latent=spk_ref["cond_latent"],
            ))
            for text_tokens in sent_text_tokens
        ]

        sil_samples = int(sampling_rate * interval_silence / 1000.0) if interval_silence > 0 else 0
        sil_tensor = torch.zeros(1, sil_samples)
        cursor = 0
        total_audio_samples = 0

        try:
            for sent_idx, (sent_text, text_tokens, gen_task) in enumerate(zip(sent_texts, sent_text_tokens, gen_tasks)):
                m_start_time = time.perf_counter()
                codes, speech_conditioning_latent = await gen_task
                gpt_gen_time += time.perf_counter() - m_start_time

                wav, fwd_dt, mel_dt, voc_dt = await asyncio.get_running_loop().run_in_executor(
                    self._mel_executor,
                    self._sentence_to_wav,
                    codes, text_tokens, speech_conditioning_latent,
                    spk_cond_emb, emo_cond_emb_gpt, cond_lengths, emo_cond_lengths,
                    emovec, prompt_condition, ref_mel, style, verbose,
                )
                gpt_forward_time += fwd_dt
                s2mel_time += mel_dt
                bigvgan_time += voc_dt

                n = wav.shape[-1]
                segment = {
                    "text": sent_text,
                    "start": round(cursor / sampling_rate, 3),
                    "end": round((cursor + n) / sampling_rate, 3),
                }
                is_last = sent_idx == len(gen_tasks) - 1
                chunk = wav if (is_last or sil_samples == 0) else torch.cat([wav, sil_tensor], dim=1)
                cursor += chunk.shape[-1]
                total_audio_samples += chunk.shape[-1]

                yield segment, sampling_rate, chunk
        except BaseException:
            # also covers GeneratorExit when the consumer disconnects mid-stream:
            # abort the decodes still running in vLLM
            for task in gen_tasks:
                task.cancel()
            raise
        end_time = time.perf_counter()

        wav_length = total_audio_samples / sampling_rate
        logger.info(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        logger.info(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        logger.info(f">> s2mel_time: {s2mel_time:.2f} seconds")
        logger.info(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        logger.info(f">> Total inference time: {end_time - start_time:.2f} seconds")
        logger.info(f">> Generated audio length: {wav_length:.2f} seconds")
        logger.info(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

    async def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_sentence=120, return_segments=False, **generation_kwargs):
        wavs = []
        segments = []
        sampling_rate = 22050

        async for segment, sampling_rate, chunk in self.infer_stream(
            spk_audio_prompt, text,
            emo_audio_prompt=emo_audio_prompt, emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text, emo_text=emo_text, use_random=use_random,
            interval_silence=interval_silence,
            verbose=verbose, max_text_tokens_per_sentence=max_text_tokens_per_sentence,
            **generation_kwargs,
        ):
            segments.append(segment)
            wavs.append(chunk)

        wav = torch.cat(wavs, dim=1)

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                logger.info(f">> remove old wav file: {output_path}")
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            logger.info(f">> wav file saved to: {output_path}")
            return output_path
        else:
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            if return_segments:
                return (sampling_rate, wav_data, segments)
            return (sampling_rate, wav_data)


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index

class QwenEmotion:
    def __init__(self, model_dir, gpu_memory_utilization=0.1, device_id=None):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        env_backup = os.environ.get("CUDA_VISIBLE_DEVICES")
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        engine_args = AsyncEngineArgs(
            model=model_dir,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,
        )
        self.model = AsyncLLM.from_engine_args(engine_args)

        if device_id is not None:
            if env_backup is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = env_backup
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self.prompt = "文本情感分类"
        self.convert_dict = {
            "愤怒": "angry",
            "高兴": "happy",
            "恐惧": "fear",
            "反感": "hate",
            "悲伤": "sad",
            "低落": "low",
            "惊讶": "surprise",
            "自然": "neutral",
        }
        self.backup_dict = {"happy": 0, "angry": 0, "sad": 0, "fear": 0, "hate": 0, "low": 0, "surprise": 0,
                            "neutral": 1.0}
        self.max_score = 1.2
        self.min_score = 0.0

    def convert(self, content):
        content = content.replace("\n", " ")
        content = content.replace(" ", "")
        content = content.replace("{", "")
        content = content.replace("}", "")
        content = content.replace('"', "")
        parts = content.strip().split(',')
        # print(parts)
        parts_dict = {}
        desired_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        for part in parts:
            key_value = part.strip().split(':')
            if len(key_value) == 2:
                parts_dict[key_value[0].strip()] = part
        # 按照期望顺序重新排列
        ordered_parts = [parts_dict[key] for key in desired_order if key in parts_dict]
        parts = ordered_parts
        if len(parts) != len(self.convert_dict):
            return self.backup_dict

        emotion_dict = {}
        for part in parts:
            key_value = part.strip().split(':')
            if len(key_value) == 2:
                try:
                    key = self.convert_dict[key_value[0].strip()]
                    value = float(key_value[1].strip())
                    value = max(self.min_score, min(self.max_score, value))
                    emotion_dict[key] = value
                except Exception:
                    continue

        for key in self.backup_dict:
            if key not in emotion_dict:
                emotion_dict[key] = 0.0

        if sum(emotion_dict.values()) <= 0:
            return self.backup_dict

        return emotion_dict

    async def inference(self, text_input):
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(text)["input_ids"]
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        # generated_ids = self.model.generate(
        #     **model_inputs,
        #     max_new_tokens=32768,
        #     pad_token_id=self.tokenizer.eos_token_id
        # )
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        
        sampling_params = SamplingParams(
            max_tokens=2048,  # 32768
        )
        tokens_prompt = TokensPrompt(prompt_token_ids=model_inputs)
        output_generator = self.model.generate(tokens_prompt, sampling_params=sampling_params, request_id=uuid.uuid4().hex)
        async for output in output_generator:
            pass
        output_ids = output.outputs[0].token_ids[:-2]

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        emotion_dict = self.convert(content)
        return emotion_dict, content