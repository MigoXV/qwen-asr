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
from typing import Optional, Tuple

from qwen_asr.inferencers.language import normalize_language_name

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def detect_and_fix_repetitions(text, threshold=20):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
            
        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                    
                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break
                
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            
            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return ''.join(result)
    
    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def parse_asr_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
    """
    将 Qwen3-ASR 的原始输出解析为 ``(language, text)``。

    支持的情况：
      - 带标签：``"language Chinese<asr_text>...."``
      - 带换行：``"language Chinese\\n...\\n<asr_text>...."``
      - 无标签：将整个字符串视为转写文本。
      - ``"language None<asr_text>"``：视为空音频，返回 ``("", "")``。

    如果提供了 ``user_language``，返回的语言会强制使用该值。即便如此，
    仍会防御性地剥离泄漏的 ``language X<asr_text>`` 前缀，因为部分模型输出
    可能会回显 assistant prompt，而不是只返回纯文本。

    参数：
        raw: 解码后的模型原始字符串。
        user_language: 用户强制指定语言时使用的规范语言名。

    返回：
        Tuple[str, str]: ``(language, text)``。
    """
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    s = detect_and_fix_repetitions(s)

    if user_language:
        text = s
        if _ASR_TEXT_TAG in text:
            _, text = text.split(_ASR_TEXT_TAG, 1)
        else:
            forced_prefix = f"{_LANG_PREFIX}{user_language}"
            if text.lower().startswith(forced_prefix.lower()):
                text = text[len(forced_prefix):]
        return user_language, text.strip()

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        # 没有标签时，将完整输出视为纯文本。
        return "", s.strip()

    meta_lower = meta_part.lower()

    # 空音频启发式判断。
    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        # 如果模型仍然返回了文本，则保留文本，但语言未知。
        return "", t

    # 从元信息中提取 "language xxx"。
    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()
