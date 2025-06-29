import re
from typing import Counter

# Modified pre-tokenization regex pattern compatible with Python's re module
# \p{L} -> [a-zA-ZÀ-ÿĀ-žА-я一-龯] (covers most common Unicode letter ranges)
# \p{N} -> [0-9] (covers ASCII digits)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-ZÀ-ÿĀ-žА-я一-龯]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""

def _get_pair_stats(word_counts: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    """Helper to count adjacent pairs from the word frequency map."""
    pair_counts = Counter()
    for word_tuple, count in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] += count
    return pair_counts


def _process_chunk(
        input_path: str, start: int, end: int, unicode_map: dict[int, str]
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
    pre_tokens = re.findall(PAT, chunk_text)

    for pre_token in pre_tokens:
        pre_token_bytes = pre_token.encode("utf-8")
        # Convert to tuple of printable unicode strings
        word_tuple = tuple(unicode_map[b] for b in pre_token_bytes)
        word_counts[word_tuple] += 1

    return word_counts