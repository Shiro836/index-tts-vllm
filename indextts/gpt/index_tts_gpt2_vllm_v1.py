"""Custom GPT2-based TTS model registered into vLLM (ported to vLLM 0.22.x).

The prompt is a sequence of precomputed conditioning+text embeddings delivered
via the "audio" multimodal modality; generated tokens are mel codes embedded
through `audio_emb` with a learned positional embedding indexed by the
(runner-shifted, see patch_vllm.py) position ids.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import torch
from torch import nn
from transformers import BatchFeature

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings,
                                                   SupportsMultiModal,
                                                   SupportsPP)
from vllm.model_executor.models.utils import (
    _merge_multimodal_embeddings,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.gpt2 import GPT2Block

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.multimodal.parse import (DictEmbeddingItems, ModalityDataItems,
                                   MultiModalDataItems, MultiModalDataParser)

PLACEHOLDER_TOKEN = "!"
PLACEHOLDER_TOKEN_ID = 0


def _audio_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(audio_embeds=MultiModalFieldConfig.batched("audio"))


class GPT2TTSProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    # since vLLM ~0.11 the data parser is supplied by ProcessingInfo,
    # not by the processor
    def get_data_parser(self) -> "MultiModalDataParser":
        return GPT2TTSDataParser()


class GPT2TTSDummyInputsBuilder(BaseDummyInputsBuilder[GPT2TTSProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return PLACEHOLDER_TOKEN * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        num_items = mm_counts.get("audio", 0)
        if num_items == 0:
            return {}

        config = self.info.get_hf_config()
        dummy_seq_len = 1024
        dummy_embed = torch.rand(
            (dummy_seq_len, config.n_embd),
            dtype=torch.float16,
        )

        return {"audio": {"audio_embeds": [dummy_embed] * num_items}}


class GPT2TTSDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: Union[Dict[str, torch.Tensor], Any],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_audio_field_config,
            )

        raise TypeError(
            "For 'audio' modality, expected a dict like "
            f"{{'audio_embeds': tensor}}, but got {type(data)}")


class GPT2TTSMultiModalProcessor(BaseMultiModalProcessor[GPT2TTSProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _audio_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_mm_data = out_mm_kwargs.get_data()

        def get_replacement(item_idx: int):
            embeds = out_mm_data["audio_embeds"][item_idx]
            num_features = embeds.shape[0]
            return [PLACEHOLDER_TOKEN_ID] * num_features

        return [
            PromptReplacement(
                modality="audio",
                target=PLACEHOLDER_TOKEN,
                replacement=get_replacement,
            )
        ]


@support_torch_compile
class GPT2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.n_embd
        # token/positional embeddings live on the wrapper (GPT2TTSModel); the
        # prompt arrives as precomputed embeds, so this stack starts at layer 0
        self.start_layer, self.end_layer, self.h = make_layers(
            config.n_layer,
            lambda prefix: GPT2Block(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h")
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.n_embd))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = inputs_embeds

        for layer in self.h[self.start_layer:self.end_layer]:
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            # The HF's GPT-2 implementation uses Conv1D instead of Linear.
            # Because of this, we need to transpose the weights.
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


@MULTIMODAL_REGISTRY.register_processor(GPT2TTSMultiModalProcessor,
                                        info=GPT2TTSProcessingInfo,
                                        dummy_inputs=GPT2TTSDummyInputsBuilder)
class GPT2TTSModel(nn.Module, SupportsPP, SupportsMultiModal):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return PLACEHOLDER_TOKEN
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.transformer = GPT2Model(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "transformer"))
        self.text_pos_embedding = LearnedPositionEmbeddings(self.config.n_positions, self.config.n_embd)
        with torch.no_grad():
            self.text_pos_embedding.emb.weight[0].zero_()
        self.audio_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.final_norm = nn.LayerNorm(self.config.n_embd, bias=True)
        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.n_embd,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.lm_head",
                                      bias=True)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

    def get_language_model(self) -> torch.nn.Module:
        return self.transformer

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_embeds = kwargs.get("audio_embeds")
        if audio_embeds is None:
            return []

        processed_embeds = []
        for embed in audio_embeds:
            if embed.dim() == 3 and embed.shape[0] == 1:
                processed_embeds.append(embed.squeeze(0))
            elif embed.dim() == 2:
                processed_embeds.append(embed)
            else:
                raise ValueError(
                    "Expected audio embeddings to be 2D or 3D with a "
                    f"leading dimension of 1, but got shape: {embed.shape}")

        return processed_embeds

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        *,
        is_multimodal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # decode tokens are mel codes -> audio_emb; prompt placeholder tokens
        # get overwritten by the conditioning embeds
        inputs_embeds = self.audio_emb(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        if is_multimodal is None:
            is_multimodal = input_ids == PLACEHOLDER_TOKEN_ID

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # positions were shifted by the patched runner so that the first
        # generated token gets index 1; prompt tokens clamp to 0, whose
        # embedding is pinned to zero (no-op add)
        positions = torch.clamp(positions, min=0)
        pos_emb = self.text_pos_embedding.emb(positions)

        # kusuriuri: 这里必须使用 += ，否则计算结果会错误
        inputs_embeds += pos_emb

        transformer_output = self.transformer(
            input_ids=None,
            position_ids=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds
        )

        transformer_output = self.final_norm(transformer_output)

        return transformer_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue
            if ".wte" in name:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # position 0 must stay a zero vector: prompt tokens clamp to it and
        # rely on the add being a no-op
        with torch.no_grad():
            self.text_pos_embedding.emb.weight[0].zero_()
        return loaded_params
