from __future__ import annotations

import os
from typing import Any



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from cs336_data.help import extract_text_from_html
    return extract_text_from_html(html_bytes)

def run_identify_language(text: str) -> tuple[Any, float]:
    from cs336_data.help import language_identify
    languages = ["en", "zh"]
    while True:
        results = language_identify(text)
        if results[0] in languages:
            return results 

def run_mask_emails(text: str) -> tuple[str, int]:
    from cs336_data.help import mask_emails
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    from cs336_data.help import mask_phonenumber
    return mask_phonenumber(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    from cs336_data.help import mask_ips
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    from cs336_data.help import classify_nsfw
    return classify_nsfw(text)

def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    from cs336_data.help import classify_toxic_speech
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    from cs336_data.help import run_quality_classifier
    return run_quality_classifier(text)

def run_gopher_quality_filter(text: str) -> bool:
    from cs336_data.help import gopher_quality_filters
    return gopher_quality_filters(text)



def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    from cs336_data.help import exact_deduplication
    exact_deduplication(input_files, output_directory)

def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    from cs336_data.help import minhash_deduplication
    minhash_deduplication(
        input_files,
        num_hashes,
        num_bands,
        ngrams,
        jaccard_threshold,
        output_directory,
    )