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

from transformers.configuration_utils import PretrainedConfig
from .config_audio import (
    Qwen3ASRAudioEncoderConfig,
)
from .config_text import (
    Qwen3ASRTextConfig,
)


class Qwen3ASRThinkerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3ASRThinker`]. It is used to instantiate a
    Qwen3-ASR-Thinker model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the thinker component of the Qwen3-Omni
    architecture.

    e.g. [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`dict`, *optional*):
            The config dictionary of the audio backbone.
        text_config (`dict`, *optional*):
            The config dictionary of the text backbone.
        audio_token_id (`int`, *optional*, defaults to 151646):
            The audio token id to encode the audio prompt.
        audio_start_token_id (`int`, *optional*, defaults to 151647):
            The audio start token id to encode the audio prompt.
        user_token_id (`int`, *optional*, defaults to 872):
            The user token id to encode the user token.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import Qwen3ASRThinkerModel, Qwen3ASRThinkerConfig

    >>> # Initializing a default Qwen3ASRThinkerConfig
    >>> configuration = Qwen3ASRThinkerConfig()

    >>> # Initializing a model (with random weights) from the default configuration
    >>> model = Qwen3ASRThinkerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_asr_thinker"

    attribute_map = {}
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3ASRTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151646,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range

        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id


class Qwen3ASRConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Qwen3ASRForConditionalGeneration`]. It is used to instantiate a Qwen3ASR
    model according to the specified sub-models configurations, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        thinker_config (`dict`, *optional*): Configuration of the underlying thinker sub-model.
        support_languages (`List[str]`, *optional*): The languages supported by the model.

    Example:

    ```python
    >>> from transformers import (
    ...     Qwen3ASRThinkerConfig,
    ...     Qwen3ASRForConditionalGeneration,
    ...     Qwen3ASRConfig,
    ... )

    >>> # Initializing a Qwen3ASR style configuration
    >>> configuration = Qwen3ASRConfig()

    >>> # Initializing a model from the configuration
    >>> model = Qwen3ASRForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_asr"
    sub_configs = {
        "thinker_config": Qwen3ASRThinkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        support_languages=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}

        self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        self.support_languages = support_languages

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        """
        Returns the config that is meant to be used with text IO. On most models, it is the original config instance
        itself. On specific composite models, it is under a set of valid names.

        Args:
            decoder (`Optional[bool]`, *optional*, defaults to `False`):
                If set to `True`, then only search for decoder config names.
        """
        # Overridden for deeply nested config like Qwen2.5-Omni. We don't have any omni model
        # except for Qwen yet. This has to be generalized if more deeply nested configs are
        # added. NOTE: currently method used only by vLLM
        return self.thinker_config.get_text_config()
