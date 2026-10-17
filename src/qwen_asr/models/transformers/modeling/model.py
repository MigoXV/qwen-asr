# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union
import torch
from torch import nn
from transformers.generation import GenerationMixin
from transformers.utils import auto_docstring, can_return_tuple
from ..configuration_qwen3_asr import Qwen3ASRConfig, Qwen3ASRThinkerConfig
from .layers import (
    Qwen3ASRThinkerTextAttention,
)
from .base import (
    Qwen3ASRPreTrainedModel,
    Qwen3ASRThinkerCausalLMOutputWithPast,
    Qwen3ASRPreTrainedModelForConditionalGeneration,
)
from .audio import (
    Qwen3ASRAudioEncoder,
)
from .text import (
    Qwen3ASRThinkerTextDecoderLayer,
    Qwen3ASRThinkerTextModel,
)


@auto_docstring(
    custom_intro="""
    The Qwen3ASRThinker model which consists of a audio backbone and a language model.
    """
)
class Qwen3ASRThinkerForConditionalGeneration(Qwen3ASRPreTrainedModelForConditionalGeneration, GenerationMixin):
    config: Qwen3ASRThinkerConfig
    base_model_prefix = "thinker"
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]
    _no_split_modules = [
        "Qwen3ASRAudioEncoderLayer",
        "Qwen3ASRThinkerTextDecoderLayer",
    ]
    _can_record_outputs = {
        "hidden_states": Qwen3ASRThinkerTextDecoderLayer,
        "attentions": Qwen3ASRThinkerTextAttention,
    }

    def __init__(self, config):
        super().__init__(config)
        self.audio_tower = Qwen3ASRAudioEncoder._from_config(config.audio_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen3ASRThinkerTextModel._from_config(config.text_config)
        if "forced_aligner" in config.model_type:
            self.lm_head = nn.Linear(config.text_config.hidden_size, config.classify_num, bias=False)
        else:
            self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)

        # audio encoder do not support batch inference to keep precision
        audio_features = []
        for input_feature, feature_len in zip(input_features, feature_lens):
            audio_output = self.audio_tower(
                input_feature[:, :feature_len],
                feature_lens=feature_len.unsqueeze(0),
            )
            audio_feature = audio_output.last_hidden_state
            audio_features.append(audio_feature)
        audio_features = torch.cat(audio_features, dim=0)

        return audio_features

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_audio_mask = input_ids == self.config.audio_token_id

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_audio_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> Union[tuple, Qwen3ASRThinkerCausalLMOutputWithPast]:
        r"""
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text, audios
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    attention_mask,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        return Qwen3ASRThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        input_features=None,
        feature_attention_mask=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["input_features"] = None

        return model_inputs


class Qwen3ASRForConditionalGeneration(Qwen3ASRPreTrainedModel, GenerationMixin):
    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.config = config

        self.thinker = Qwen3ASRThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.post_init()

    def get_support_languages(self):
        return self.config.support_languages

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 4096,
        eos_token_id: int | list[int] = [151645, 151643],
        **kwargs,
    ):
        shared_kwargs = {}
        thinker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_id,
        }

        for key, value in kwargs.items():
            # Process special input values
            if key == "feature_attention_mask":
                thinker_kwargs[key] = value
            elif key in ("input_features", "attention_mask"):
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value

        thinker_result = self.thinker.generate(input_ids=input_ids, return_dict_in_generate=True, **thinker_kwargs)

        return thinker_result
