import re
import collections
from typing import List, Dict, Tuple
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger("text_preprocessing")


class RadiologyTextPreprocessor:
    """
    Handles medical report cleaning, section extraction (Findings / Impression),
    tokenization, vocabulary building, and duplicate report removal.
    """

    def __init__(
        self,
        min_word_freq: int = 3,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        self.min_word_freq = min_word_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.vocab_freq: collections.Counter = collections.Counter()

        self._build_special_tokens()

    def _build_special_tokens(self):
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for idx, tok in enumerate(special_tokens):
            self.word2idx[tok] = idx
            self.idx2word[idx] = tok

    @staticmethod
    def clean_report_text(text: str) -> str:
        """
        Cleans raw radiology text: lowercasing, whitespace normalization, punctuation standardizing.
        """
        if not text or pd.isna(text):
            return ""

        text = str(text)
        text = text.lower()
        # Remove de-identification brackets e.g. [**2019-01-01**]
        text = re.sub(r"\[\*\*.*?\*\*\]", "", text)
        # Remove digits / numbers if appropriate or keep spaces around symbols
        text = re.sub(r"([.,;:?!\(\)])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_findings_and_impression(raw_report: str) -> Tuple[str, str]:
        """
        Extracts FINDINGS and IMPRESSION sections from raw medical text.
        """
        if not raw_report or pd.isna(raw_report):
            return "", ""

        report_upper = str(raw_report).upper()

        findings = ""
        impression = ""

        # Match FINDINGS section
        findings_match = re.search(r"FINDINGS:\s*(.*?)(?=\s*(?:IMPRESSION:|RECOMMENDATION:|NOTIFICATION:|$))", report_upper, re.DOTALL)
        if findings_match:
            findings = findings_match.group(1).strip()

        # Match IMPRESSION section
        impression_match = re.search(r"IMPRESSION:\s*(.*?)(?=\s*(?:RECOMMENDATION:|NOTIFICATION:|$))", report_upper, re.DOTALL)
        if impression_match:
            impression = impression_match.group(1).strip()

        # Fallback if section headers missing
        if not findings and not impression:
            findings = str(raw_report)

        return RadiologyTextPreprocessor.clean_report_text(findings), RadiologyTextPreprocessor.clean_report_text(impression)

    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean_report_text(text)
        return cleaned.split()

    def build_vocabulary(self, corpus: List[str]) -> Dict[str, int]:
        """
        Builds vocabulary dictionary from a corpus of text reports based on min frequency.
        """
        counter = collections.Counter()
        for text in corpus:
            tokens = self.tokenize(text)
            counter.update(tokens)

        self.vocab_freq = counter

        idx = len(self.word2idx)
        for word, count in counter.most_common():
            if count >= self.min_word_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        logger.info(f"Vocabulary Built: {len(self.word2idx)} unique tokens (Min Freq: {self.min_word_freq})")
        return self.word2idx

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(tok, self.word2idx[self.unk_token]) for tok in tokens]

        if add_special_tokens:
            ids = [self.word2idx[self.bos_token]] + ids + [self.word2idx[self.eos_token]]

        return ids

    def decode(self, token_ids: List[int]) -> str:
        words = [self.idx2word.get(tid, self.unk_token) for tid in token_ids]
        # Remove BOS/EOS/PAD tokens for clean string display
        filtered = [w for w in words if w not in [self.bos_token, self.eos_token, self.pad_token]]
        return " ".join(filtered)

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, text_col: str = "full_report") -> pd.DataFrame:
        """
        Removes exact duplicate reports to ensure clean training corpora.
        """
        initial_len = len(df)
        df_clean = df.drop_duplicates(subset=[text_col]).copy()
        logger.info(f"Duplicate Removal: {initial_len} -> {len(df_clean)} records (Removed {initial_len - len(df_clean)} duplicates)")
        return df_clean
