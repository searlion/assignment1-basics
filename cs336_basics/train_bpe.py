import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Tuple, Counter

from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.train_bpe_helper import _process_chunk, _get_pair_stats
from tests.common import gpt2_bytes_to_unicode


@dataclass
class BpeTrainingResult:
    """
    A container for the results of BPE tokenizer training.

    Attributes:
        vocab (Dict[int, bytes]):
            The tokenizer vocabulary, mapping token IDs to bytes.
        merges (List[Tuple[bytes, bytes]]):
            The ordered list of BPE merges created during training.
    """
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> BpeTrainingResult:
    """
    Trains a byte-level Byte Pair Encoding (BPE) tokenizer from a text file.

    This function reads a text file, builds an initial vocabulary from all
    possible byte values (0-255), and iteratively merges the most frequent
    pair of adjacent tokens to build a final vocabulary of the desired size.

    Args:
        input_path (str):
            Path to the training text file. The file will be read as raw bytes.
        vocab_size (int):
            The target vocabulary size. This must be a positive integer >= 256.
            The final vocabulary will contain the initial 256 byte tokens,
            newly merged tokens, and any special tokens, up to this size.
        special_tokens (List[str]):
            A list of special token strings (e.g., ["<|endoftext|>"]) to add
            to the vocabulary. These are added after the BPE merges are
            complete and do not affect the training process itself.

    Returns:
        Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
            A tuple containing the trained vocabulary and the merge rules:
            - vocab (Dict[int, bytes]): A mapping from token IDs (int) to their
              byte representation (bytes).
            - merges (List[Tuple[bytes, bytes]]): An ordered list of BPE merges.
              Each element is a tuple of two byte strings that were merged,
              in the order the merges were created.
    """
    # --- 1. Setup and Initial Vocabulary ---
    assert vocab_size >= 256 + len(special_tokens)

    # 1. Setup
    unicode_map = gpt2_bytes_to_unicode()

    # --- THE FIX: Create and maintain a string-to-byte mapping ---
    ## purpose: for tie-breaking on the original byte values and construct the final output
    str_to_bytes_map = {v: bytes([k]) for k, v in unicode_map.items()}

    # 2. Initial Vocab and Word Counts
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token_str in enumerate(special_tokens):
        token_bytes = token_str.encode("utf-8")
        vocab[256 + i] = token_bytes
        # Add special tokens to our string-to-byte map as well
        str_to_bytes_map[token_str] = token_bytes

    merges: list[tuple[bytes, bytes]] = []

    # Parallel Pre-tokenization (This part is correct)
    print("Starting parallel pre-tokenization and counting...")
    num_processes = multiprocessing.cpu_count()
    ## To split a large file for parallel processing, we can't just cut it at arbitrary byte positions (we might split a multi-byte character in half).
    ## Using a document separator like <|endoftext|> or a line break (\n) is a smart heuristic to ensure chunks start and end at meaningful places.
    split_token_for_chunking = special_tokens[0].encode("utf-8") if special_tokens else b'\n'
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token_for_chunking)
    chunk_args = [(str(input_path), start, end, unicode_map, special_tokens) for start, end in
                  zip(boundaries[:-1], boundaries[1:])]
    word_counts: Counter[tuple[str, ...]] = Counter()
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.starmap(_process_chunk, chunk_args)
        for result_counter in results: word_counts.update(result_counter)
    print(f"Finished pre-tokenization. Found {len(word_counts)} unique pre-tokens.")

    # 3. The Iterative Merging Loop
    num_merges = vocab_size - len(vocab)
    print(f"Starting {num_merges} BPE merge operations...")

    for i in tqdm(range(num_merges), desc="BPE Merges"):
        ## This counts the frequency of each pair of tokens
        pair_stats = _get_pair_stats(word_counts)
        if not pair_stats:
            break

        # Tie-breaking logic (this was always correct)
        # Logic: first compare frequency
        # If frequency tie, compare first token of the pair
        # If still tie, compare second token of the pair
        best_pair = max(
            pair_stats,
            key=lambda p: (pair_stats[p], str_to_bytes_map[p[0]], str_to_bytes_map[p[1]]),
        )

        # --- THE FIX in action ---
        # Use the str_to_bytes_map to correctly get the byte representation of Found {len(word_counts)} unique pre-tokens.the pair
        part1_bytes = str_to_bytes_map[best_pair[0]]
        part2_bytes = str_to_bytes_map[best_pair[1]]

        # Add the correct bytes to the final merges list
        merges.append((part1_bytes, part2_bytes))

        # Create the new token and update all our mappings
        new_token_str = "".join(best_pair)
        new_token_bytes = part1_bytes + part2_bytes
        vocab[len(vocab)] = new_token_bytes
        str_to_bytes_map[new_token_str] = new_token_bytes

        # --- OPTIMIZATION: The incremental update logic ---
        # Find only the words that are affected by this merge
        # By using a set, we are looking only at unique words to update, not how many times they appear (in word_counts)
        # Note that this is an imperfect filter as it does not take into account ordering of the tokens
        ## This is a classic engineering trade-off between perfect filtering and "good enough: filtering for the sake of speed
        ## While there are false positives, these will be handled in the next stage of code
        words_to_update = {word for word in word_counts if best_pair[0] in word and best_pair[1] in word}

        # words_to_update: The small set of candidate words that might contain our best_pair.
        # best_pair: The pair we are merging, e.g., ('t', 'h').
        # new_token_str: The string for the new token, e.g., 'th'.
        # word_counts: The master dictionary of word_tuple -> frequency.
        # pair_stats: The master dictionary of pair_tuple -> frequency.
        for word in words_to_update:
            count = word_counts[word]
            j = 0
            new_word = []

            # This loop creates the new version of the word
            while j < len(word):
                if j < len(word) - 1 and (word[j], word[j + 1]) == best_pair:
                    # Decrement stats for pairs being destroyed by the merge
                    if j > 0:
                        pair_stats[word[j - 1], word[j]] -= count
                    if j < len(word) - 2:
                        pair_stats[word[j + 1], word[j + 2]] -= count

                    new_word.append(new_token_str)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1

            new_word_tuple = tuple(new_word)

            # Update word_counts: remove the old word, add the new one
            del word_counts[word]
            word_counts[new_word_tuple] += count

            # Increment stats for newly created pairs
            for k in range(len(new_word_tuple) - 1):
                pair_stats[new_word_tuple[k], new_word_tuple[k + 1]] += count

        # Clean up the pair we just merged
        del pair_stats[best_pair]

    print("BPE training complete.")
    return BpeTrainingResult(vocab=vocab, merges=merges)
