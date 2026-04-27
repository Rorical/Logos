"""
Tiktoken Tokenizer Wrapper
==========================
A thin wrapper around OpenAI's tiktoken that provides a HuggingFace-tokenizer-like
interface for training and inference.

- Defaults to ``cl100k_base`` (GPT-4 / GPT-3.5-turbo encoding).
- Uses ``<|endoftext|>`` as both EOS and PAD token.
- Provides batch encoding with automatic padding / truncation.
"""

from typing import List, Union, Optional, Dict

import torch
import tiktoken


class TiktokenTokenizer:
    """
    Args:
        encoding_name: tiktoken encoding name.  Common choices:
            * ``cl100k_base`` – GPT-4, GPT-3.5-turbo  (default)
            * ``o200k_base``  – GPT-4o
            * ``p50k_base``   – GPT-3 / Codex
            * ``gpt2``        – GPT-2
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

        # tiktoken uses the same id for end-of-text (pad == eos)
        self.eos_token: str = "<|endoftext|>"
        self.eos_token_id: int = self.encoding.eot_token
        self.pad_token_id: int = self.eos_token_id
        self.vocab_size: int = self.encoding.n_vocab

    # ------------------------------------------------------------------ #
    # Single-text API
    # ------------------------------------------------------------------ #
    def encode(
        self,
        text: str,
        allowed_special: Union[str, set, None] = None,
        disallowed_special: Union[str, set] = (),
    ) -> List[int]:
        """Encode a single string to token ids.

        Literal special-token strings in raw corpora are treated as normal
        text by default. Pass ``allowed_special="all"`` or a specific token
        set when those strings should become special token ids.
        """
        # tiktoken expects allowed_special as "all" or a set of strings
        if allowed_special is None:
            allowed_special = set()
        elif isinstance(allowed_special, str) and allowed_special != "all":
            allowed_special = {allowed_special}
        return self.encoding.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        )

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token ids back to a string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.encoding.decode(token_ids)

    # ------------------------------------------------------------------ #
    # Batch API (training / DataLoader)
    # ------------------------------------------------------------------ #
    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts and optionally pad / truncate.

        Returns a dict with keys:
            * ``input_ids``      – (batch, seq_len)
            * ``attention_mask`` – (batch, seq_len)  1=real, 0=pad
            * ``labels``         – (batch, seq_len)  same as input_ids but
              pad positions are replaced with ``-100`` for ``nn.CrossEntropyLoss``.
        """
        batch_ids = []
        for text in texts:
            ids = self.encode(text)
            # Append EOS
            ids.append(self.eos_token_id)
            if truncation and max_length:
                ids = ids[: max_length]
            batch_ids.append(ids)

        if not padding or max_length is None:
            # Return ragged tensors (useful for packing)
            return {
                "input_ids": [torch.tensor(ids, dtype=torch.long) for ids in batch_ids],
                "attention_mask": [
                    torch.ones(len(ids), dtype=torch.long) for ids in batch_ids
                ],
                "labels": [torch.tensor(ids, dtype=torch.long) for ids in batch_ids],
            }

        # Pad to max_length
        padded_ids = []
        attention_masks = []
        labels = []
        for ids in batch_ids:
            pad_len = max_length - len(ids)
            if pad_len > 0:
                padded_ids.append(ids + [self.pad_token_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
                labels.append(ids + [-100] * pad_len)
            else:
                padded_ids.append(ids)
                attention_masks.append([1] * len(ids))
                labels.append(ids)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # ------------------------------------------------------------------ #
    # Chat Template (ChatML)
    # ------------------------------------------------------------------ #
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Render a list of messages into a ChatML-formatted string.

        Args:
            messages: list of dicts with ``role`` and ``content`` keys.
                      Roles: ``system``, ``user``, ``assistant``.
        Returns:
            ChatML string, e.g.::

                <|im_start|>system
                You are a helpful assistant.<|im_end|>
                <|im_start|>user
                Hello!<|im_end|>
                <|im_start|>assistant
                Hi there!<|im_end|>
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)

    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenise a conversation with ChatML formatting.

        Only **assistant** content tokens are included in labels;
        system / user turns and all structural markers are masked
        with ``-100``.

        Returns a dict with keys: ``input_ids``, ``attention_mask``, ``labels``.
        """
        input_ids: List[int] = []
        labels: List[int] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Structural marker + role header
            header = f"<|im_start|>{role}\n"
            header_ids = self.encode(header)

            # Message body
            content_ids = self.encode(content)

            # End marker
            end = "<|im_end|>"
            end_ids = self.encode(end)

            input_ids.extend(header_ids)
            input_ids.extend(content_ids)
            input_ids.extend(end_ids)

            if role == "assistant":
                # Train on content only; mask header and end marker
                labels.extend([-100] * len(header_ids))
                labels.extend(content_ids)
                labels.extend([-100] * len(end_ids))
            else:
                # Mask entire turn
                labels.extend([-100] * (len(header_ids) + len(content_ids) + len(end_ids)))

        # Append EOS
        input_ids.append(self.eos_token_id)
        labels.append(self.eos_token_id)

        # Truncate from the LEFT to preserve assistant content at the end
        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"encoding='{self.encoding_name}', "
            f"vocab_size={self.vocab_size}, "
            f"eos_id={self.eos_token_id})"
        )
