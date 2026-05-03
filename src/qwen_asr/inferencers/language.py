# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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

SUPPORTED_LANGUAGES: list[str] = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]

# Language code mapping: ISO 639-1 / BCP-47 / common aliases -> canonical name.
# All keys must be lowercase.
LANGUAGE_CODE_MAP: dict[str, str] = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-sg": "Chinese",
    "zh-tw": "Chinese",
    "zh-hant": "Chinese",
    "cmn": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "en-au": "English",
    "en-in": "English",
    "english": "English",
    "yue": "Cantonese",
    "zh-hk": "Cantonese",
    "zh-yue": "Cantonese",
    "cantonese": "Cantonese",
    "ar": "Arabic",
    "ar-sa": "Arabic",
    "ar-eg": "Arabic",
    "arabic": "Arabic",
    "de": "German",
    "de-de": "German",
    "de-at": "German",
    "de-ch": "German",
    "german": "German",
    "fr": "French",
    "fr-fr": "French",
    "fr-ca": "French",
    "french": "French",
    "es": "Spanish",
    "es-es": "Spanish",
    "es-mx": "Spanish",
    "es-ar": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "portuguese": "Portuguese",
    "id": "Indonesian",
    "id-id": "Indonesian",
    "indonesian": "Indonesian",
    "it": "Italian",
    "it-it": "Italian",
    "italian": "Italian",
    "ko": "Korean",
    "ko-kr": "Korean",
    "korean": "Korean",
    "ru": "Russian",
    "ru-ru": "Russian",
    "russian": "Russian",
    "th": "Thai",
    "th-th": "Thai",
    "thai": "Thai",
    "vi": "Vietnamese",
    "vi-vn": "Vietnamese",
    "vietnamese": "Vietnamese",
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "japanese": "Japanese",
    "tr": "Turkish",
    "tr-tr": "Turkish",
    "turkish": "Turkish",
    "hi": "Hindi",
    "hi-in": "Hindi",
    "hindi": "Hindi",
    "ms": "Malay",
    "ms-my": "Malay",
    "malay": "Malay",
    "nl": "Dutch",
    "nl-nl": "Dutch",
    "nl-be": "Dutch",
    "dutch": "Dutch",
    "sv": "Swedish",
    "sv-se": "Swedish",
    "swedish": "Swedish",
    "da": "Danish",
    "da-dk": "Danish",
    "danish": "Danish",
    "fi": "Finnish",
    "fi-fi": "Finnish",
    "finnish": "Finnish",
    "pl": "Polish",
    "pl-pl": "Polish",
    "polish": "Polish",
    "cs": "Czech",
    "cs-cz": "Czech",
    "czech": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "filipino": "Filipino",
    "tagalog": "Filipino",
    "fa": "Persian",
    "fa-ir": "Persian",
    "persian": "Persian",
    "farsi": "Persian",
    "el": "Greek",
    "el-gr": "Greek",
    "greek": "Greek",
    "ro": "Romanian",
    "ro-ro": "Romanian",
    "romanian": "Romanian",
    "hu": "Hungarian",
    "hu-hu": "Hungarian",
    "hungarian": "Hungarian",
    "mk": "Macedonian",
    "mk-mk": "Macedonian",
    "macedonian": "Macedonian",
}


def normalize_language_name(language: str) -> str:
    """
    Normalize a language name to the canonical format used by Qwen3-ASR.

    Examples:
        ``"cHINese"`` -> ``"Chinese"``
    """
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def resolve_language_code(code: str | None) -> str | None:
    """
    Resolve a language code or name to the canonical English full name.

    Accepts ISO 639-1 codes, BCP-47 tags, or supported English language names.
    Unknown values return ``None`` so callers can fall back to auto-detection.
    """
    if code is None:
        return None
    key = str(code).strip()
    if not key:
        return None

    result = LANGUAGE_CODE_MAP.get(key.lower())
    if result is not None:
        return result

    normalized = normalize_language_name(key)
    if normalized in SUPPORTED_LANGUAGES:
        return normalized

    return None
