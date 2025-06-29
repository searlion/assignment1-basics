import regex as re
from typing import Counter

# Use the EXACT pattern from the assignment PDF
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _get_pair_stats(word_counts: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    """Helper to count adjacent pairs from the word frequency map."""
    pair_counts = Counter()
    for word_tuple, count in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair_counts[(word_tuple[i], word_tuple[i+1])] += count
    return pair_counts


def _process_chunk(
        input_path: str, start: int, end: int, unicode_map: dict[int, str], special_tokens: list[str]
) -> Counter[tuple[str, ...]]:
    """
    A worker function for a single process. Reads a chunk of the file,
    pre-tokenizes it, and returns the frequency of each pre-token.
    """
    word_counts = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        sub_chunks = re.split(f"({special_pattern})", chunk_text)
    else:
        sub_chunks = [chunk_text]
    for i in range(0, len(sub_chunks), 2):
        text_part = sub_chunks[i]
        if not text_part: continue
        pre_tokens = re.findall(PAT, text_part)
        for pre_token in pre_tokens:
            word_counts[tuple(unicode_map[b] for b in pre_token.encode("utf-8"))] += 1
    return word_counts