from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer
from utils.logger import setup_logger

logger = setup_logger("vlm_tokenizer")

DEFAULT_PROMPT = "Generate a detailed radiology findings and impression report for this chest X-ray image:"


class VLMTokenizerWrapper:
    """
    Tokenizer wrapper for FLAN-T5 medical report generation.
    Handles prompt encoding, target report encoding with -100 padding, and string decoding.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        prompt: str = DEFAULT_PROMPT,
        max_prompt_len: int = 32,
        max_target_len: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt = prompt
        self.max_prompt_len = max_prompt_len
        self.max_target_len = max_target_len

        # Pre-encode default prompt
        encoded_prompt = self.tokenizer(
            self.prompt,
            max_length=self.max_prompt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.prompt_ids = encoded_prompt["input_ids"].squeeze(0)
        self.prompt_mask = encoded_prompt["attention_mask"].squeeze(0)

    def encode_target_report(self, report_text: str) -> torch.Tensor:
        """
        Encodes target report text into label IDs, replacing pad tokens with -100 for Cross-Entropy loss.
        """
        if not report_text or not isinstance(report_text, str):
            report_text = "No acute cardiopulmonary findings."

        tokens = self.tokenizer(
            report_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # Replace pad_token_id with -100
        labels = tokens.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def decode_generated_ids(self, token_ids: torch.Tensor) -> str:
        """
        Decodes token IDs or generated predictions back to string.
        """
        if token_ids.dim() == 2:
            token_ids = token_ids.squeeze(0)

        # Replace -100 with pad_token_id before decoding
        clean_ids = token_ids.clone()
        clean_ids[clean_ids == -100] = self.tokenizer.pad_token_id

        return self.tokenizer.decode(clean_ids, skip_special_tokens=True)
