import regex as re
import pickle
from typing import Iterable, Iterator, List, Dict, Tuple, Optional

# The same regex pattern used in training, as required by the spec
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    A BPE Tokenizer that encodes text into token IDs and decodes token IDs back to text.
    """

    def __init__(
            self,
            vocab: Dict[int, bytes],
            merges: List[Tuple[bytes, bytes]],
            special_tokens: Optional[List[str]] = None,
    ):
        """
        Constructs the tokenizer from a vocabulary and a list of merges.

        Args:
            vocab: A mapping from token ID (int) to token bytes.
            merges: A list of BPE merges, ordered by creation.
            special_tokens: A list of special token strings.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # --- Pre-compute data structures for efficient encoding/decoding ---

        # 1. An inverted vocabulary for fast encoding (bytes -> ID)
        self.encoder: Dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # 2. A dictionary mapping merge pairs to their rank (order of creation)
        # Lower rank = merged earlier = higher priority
        self.bpe_ranks: Dict[Tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}

        # 3. A set of special tokens for fast lookups
        self.special_tokens_set = set(self.special_tokens)

        # 4. A compiled regex pattern to split text by special tokens.
        # The capturing group `()` is crucial to keep the delimiters in the output list.
        # --- THE FIX IS HERE ---
        # A compiled regex pattern to split text by special tokens.
        if self.special_tokens:
            # Sort special tokens by length, from longest to shortest.
            # This ensures the regex engine tries to match longer tokens first,
            # correctly handling overlapping tokens.
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

            special_pattern = "|".join(re.escape(s) for s in sorted_special_tokens)
            self.special_token_pattern = re.compile(f"({special_pattern})")
        else:
            self.special_token_pattern = None

    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """
        Class method to construct a Tokenizer from serialized files.
        Assumes the files were created with a library like pickle.
        """
        print(f"Loading vocabulary from {vocab_filepath}")
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        print(f"Loading merges from {merges_filepath}")
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _get_pairs(self, parts: List[bytes]) -> set:
        """Helper to get all adjacent pairs from a list of token parts."""
        return set(zip(parts, parts[1:]))

    def _bpe_encode_bytes(self, token_bytes: bytes) -> List[int]:
        """
        Applies the BPE merge algorithm to a single sequence of bytes (a pre-token).
        """
        if not token_bytes:
            return []

        # Initially, the token is broken into its constituent single-byte parts
        parts: List[bytes] = [bytes([b]) for b in token_bytes]

        while True:
            pairs = self._get_pairs(parts)
            if not pairs:
                break

            # Find the best pair to merge (the one with the lowest rank/priority)
            best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))

            # If no pairs are in our merge list, we're done with this token.
            if best_pair not in self.bpe_ranks:
                break

            # --- Merge the best pair ---
            new_parts: List[bytes] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    # If we find the pair, merge them and advance index by 2
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    # Otherwise, copy the token and advance by 1
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        # Convert the final token parts to their integer IDs
        return [self.encoder[part] for part in parts]

    def encode(self, text: str) -> List[int]:
        """
        Encodes an input text string into a sequence of token IDs.
        """
        token_ids: List[int] = []

        # First, split the text by special tokens if they exist
        if self.special_token_pattern:
            text_chunks = self.special_token_pattern.split(text)
        else:
            text_chunks = [text]

        for chunk in text_chunks:
            if not chunk:
                continue

            # If the chunk is a special token, encode it directly
            if chunk in self.special_tokens_set:
                token_ids.append(self.encoder[chunk.encode("utf-8")])
            else:
                # If it's a regular text chunk, apply pre-tokenization and BPE
                pre_tokens = re.findall(PAT, chunk)
                for pre_token in pre_tokens:
                    pre_token_bytes = pre_token.encode("utf-8")
                    token_ids.extend(self._bpe_encode_bytes(pre_token_bytes))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, returns a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files.
        """
        buffer = ""
        # We use a large buffer to make processing efficient, but not so large
        # that it consumes all memory. 1MB is a reasonable size.
        BUFFER_SIZE = 1_048_576

        for text_chunk in iterable:
            buffer += text_chunk
            if len(buffer) >= BUFFER_SIZE:
                # Find the last safe split point (e.g., a newline or space)
                # to avoid cutting a pre-token in half.
                split_pos = buffer.rfind(' ', 0, len(buffer) - 200)
                if split_pos == -1:
                    split_pos = buffer.rfind('\n', 0, len(buffer) - 200)

                if split_pos != -1:
                    text_to_process = buffer[:split_pos]
                    buffer = buffer[split_pos:]
                    for token_id in self.encode(text_to_process):
                        yield token_id

        # Process any remaining text in the buffer after the loop finishes
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs back into a text string.
        """
        # Look up each token ID to get its byte representation
        # Use .get() with a default of b'' to handle potential unknown IDs gracefully
        token_bytes_list = [self.vocab.get(i, b'') for i in ids]

        # Concatenate all byte chunks into one
        total_bytes = b"".join(token_bytes_list)

        # Decode the final byte sequence into a string, replacing invalid UTF-8
        # sequences with the Unicode replacement character '�'.
        return total_bytes.decode("utf-8", errors="replace")