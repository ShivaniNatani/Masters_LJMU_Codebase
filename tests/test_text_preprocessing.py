import pandas as pd

from preprocessing.text_preprocessing import RadiologyTextPreprocessor


def test_encode_decode_roundtrip_preserves_known_tokens():
    pre = RadiologyTextPreprocessor(min_word_freq=1)
    pre.build_vocabulary(["the lungs are clear", "no acute cardiopulmonary disease"])

    ids = pre.encode("the lungs are clear", add_special_tokens=True)
    assert ids[0] == pre.word2idx[pre.bos_token]
    assert ids[-1] == pre.word2idx[pre.eos_token]

    decoded = pre.decode(ids)
    assert decoded == "the lungs are clear"


def test_unseen_words_map_to_unk():
    pre = RadiologyTextPreprocessor(min_word_freq=1)
    pre.build_vocabulary(["the lungs are clear"])

    ids = pre.encode("pneumothorax detected", add_special_tokens=False)
    assert all(tid == pre.word2idx[pre.unk_token] for tid in ids)


def test_min_word_freq_excludes_rare_tokens():
    pre = RadiologyTextPreprocessor(min_word_freq=2)
    pre.build_vocabulary(["rare_token common common", "common common"])

    assert "common" in pre.word2idx
    assert "rare_token" not in pre.word2idx


def test_extract_findings_and_impression_splits_sections():
    report = "FINDINGS: Lungs are clear. IMPRESSION: No acute disease."
    findings, impression = RadiologyTextPreprocessor.extract_findings_and_impression(report)

    assert "lungs are clear" in findings
    assert "no acute disease" in impression


def test_remove_duplicates_drops_exact_matches_only():
    df = pd.DataFrame({"full_report": ["report a", "report a", "report b"]})
    deduped = RadiologyTextPreprocessor.remove_duplicates(df)

    assert len(deduped) == 2
