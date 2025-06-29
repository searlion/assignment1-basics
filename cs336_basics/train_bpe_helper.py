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

    # --- [START] CORRECTED LOGIC FOR SPECIAL TOKENS ---

    # 1. Create a regex pattern to split the text by any of the special tokens.
    # We must escape them in case they contain special regex characters.
    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        sub_chunks = re.split(f"({special_pattern})", chunk_text)
    else:
        sub_chunks = [chunk_text]

    # 2. Process each sub-chunk.
    # The output of re.split with a capturing group is [text, separator, text, separator, ...].
    # We only want to run our BPE pre-tokenizer on the 'text' parts.
    for i in range(0, len(sub_chunks), 2):  # Iterate over the text parts
        text_part = sub_chunks[i]
        if not text_part:
            continue

        pre_tokens = re.findall(PAT, text_part)

        for pre_token in pre_tokens:
            pre_token_bytes = pre_token.encode("utf-8")
            word_tuple = tuple(unicode_map[b] for b in pre_token_bytes)
            word_counts[word_tuple] += 1

    # --- [END] CORRECTED LOGIC ---

    return word_counts