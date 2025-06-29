import regex as re
import pickle
from typing import Iterable, Iterator, List, Dict, Tuple, Optional

# The same regex pattern used in training, as required by the spec
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Text -> Split by Special Tokens -> Pre-tokenize with Regex -> Apply BPE Merges -> Get Token IDs
class Tokenizer:
    """
    A BPE Tokenizer that encodes text into token IDs and decodes token IDs back to text.
    """

    # Overall Purpose: The constructor's main job isn't just to store the inputs,
    # but to pre-process them into efficient data structures.
    # This initial investment of computation makes the frequently-called encode and decode methods much faster.
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
        # These lines simply store the vocabulary, merge rules, and special tokens list as instance attributes.
        # The or [] is a safety measure to ensure self.special_tokens is always a list, even if None was passed.
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # --- Pre-compute data structures for efficient encoding/decoding ---

        # 1. An inverted vocabulary for fast encoding (bytes -> ID)
        # This creates a new dictionary called encoder. It iterates through the input vocab (which maps id -> bytes) and reverses it to create a map from bytes -> id.
        # The decode method needs to go from an ID to bytes (vocab).
        # The encode method needs to go from a final merged byte chunk to its ID.
        # This inverted encoder map makes that lookup extremely fast (O(1) on average).
        self.encoder: Dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # 2. A dictionary mapping merge pairs to their rank (order of creation)
        # This creates a "priority map" for the merge rules.
        # It takes the list of merges and turns it into a dictionary where the key is the merge pair (b't', b'h') and the value is its index in the list (its "rank").
        # Lower rank = merged earlier = higher priority
        self.bpe_ranks: Dict[Tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}

        # 3. A set of special tokens for fast lookups
        # Converts the list of special tokens into a set.
        # Checking if an item is in a set is much faster (O(1) on average) than checking if it's in a list (O(N)).
        # Since the encode method frequently checks if a chunk of text is a special token, this is a simple but effective optimization.
        self.special_tokens_set = set(self.special_tokens)

        # 4. A compiled regex pattern to split text by special tokens.
        # The capturing group `()` is crucial to keep the delimiters in the output list.
        # --- THE FIX IS HERE ---
        # A compiled regex pattern to split text by special tokens.
        #  This pre-compiled pattern allows the encode method to correctly and efficiently split the input text by special tokens as its very first step,
        #  ensuring longer tokens are matched before their shorter substrings.
        if self.special_tokens:
            # Sort special tokens by length, from longest to shortest.
            # This ensures the regex engine tries to match longer tokens first,
            # correctly handling overlapping tokens.
            # Example of an overlapping token is `Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>`
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # "|".join(...): This builds a regex "alternation" pattern.
            # For example, ["<|eot|>", "hi"] becomes the pattern string <\\|eot\\|>|hi.
            # re.escape is used to safely handle any special regex characters within the tokens themselves.
            # The re.escape() function is the perfect tool for this job.
            # Its sole purpose is to take a string and put a backslash (\) in front of every character that has a special meaning in regex.
            # It "neutralizes" the string, telling the regex engine "treat these characters as the literal characters they are, not as instructions."
            special_pattern = "|".join(re.escape(s) for s in sorted_special_tokens)
            # re.compile(f"({special_pattern})"):
            # This "compiles" the regex string into a highly efficient pattern object for repeated use.
            # The outer parentheses () are critical:
            # they turn the pattern into a capturing group, which tells re.split() to keep the delimiters (the special tokens) in the resulting list.
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

    # Purpose: It takes a single chunk of bytes (representing one "pre-token" like b"testing")
    # and applies the merge rules iteratively until no more merges are possible.
    def _bpe_encode_bytes(self, token_bytes: bytes) -> List[int]:
        """
        Applies the BPE merge algorithm to a single sequence of bytes (a pre-token).
        """
        if not token_bytes:
            return []

        # Initially, the token is broken into its constituent single-byte parts
        # Decomposes the input token_bytes into a list of its fundamental, single-byte components. b"cat" becomes [b'c', b'a', b't'].
        parts: List[bytes] = [bytes([b]) for b in token_bytes]

        # In a loop, it finds all adjacent pairs in the current parts.
        # It then uses the pre-computed bpe_ranks dictionary to find the pair with the lowest rank (highest priority).
        while True:
            pairs = self._get_pairs(parts)
            if not pairs:
                break

            # Find the best pair to merge (the one with the lowest rank/priority)
            # The .get(p, float("inf")) is a clever trick: if a pair is not in our merge rules, it gets a rank of infinity,
            # ensuring it will never be chosen as the minimum.
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
            # It rebuilds the parts list, replacing the best_pair with the newly merged token.
            # For example, [b'c', b'a', b't'] might become [b'ca', b't'] if (b'c', b'a') was the best pair.
            parts = new_parts

        # Convert the final token parts to their integer IDs
        # Once the loop is finished, the parts list contains the final, fully merged byte chunks.
        # This line uses the encoder map to convert each of these final byte chunks into its corresponding integer ID.
        return [self.encoder[part] for part in parts]

    # Turn human-readable text into a list of integer token IDs.
    def encode(self, text: str) -> List[int]:
        """
        Encodes an input text string into a sequence of token IDs.
        """
        token_ids: List[int] = []

        # First, split the text by special tokens if they exist
        if self.special_token_pattern:
            # Step 1: Split by Special Tokens.
            # It uses the pre-compiled regex to split the input text. Hello<|eot|> becomes ['Hello', '<|eot|>', ''].
            text_chunks = self.special_token_pattern.split(text)
        else:
            text_chunks = [text]

        for chunk in text_chunks:
            if not chunk:
                continue

            # If the chunk is a special token, encode it directly
            if chunk in self.special_tokens_set:
                # Step 2: Handle Special Tokens. It checks if a chunk is a special token.
                # If so, it encodes it directly using the encoder map and moves on.
                token_ids.append(self.encoder[chunk.encode("utf-8")])
            else:
                # Step 3: Handle Regular Text.
                # If a chunk is normal text, it first applies the PAT regex to get "pre-tokens" (words/word-parts).
                pre_tokens = re.findall(PAT, chunk)
                for pre_token in pre_tokens:
                    pre_token_bytes = pre_token.encode("utf-8")
                    # Step 4: BPE Merge and Final Encoding.
                    # For each of these pre-tokens,
                    # it calls the _bpe_encode_bytes engine to perform the merges and get the final IDs,
                    # extending the main list with the result.
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

    # Turn a list of integer token IDs back into human-readable text.
    def decode(self, ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs back into a text string.
        """
        # Look up each token ID to get its byte representation
        # Use .get() with a default of b'' to handle potential unknown IDs gracefully
        # Step 1: Convert IDs to Bytes.
        # It iterates through the input list of integer IDs and uses the self.vocab map (id -> bytes) to look up the byte representation for each one.
        # .get() provides safety against invalid IDs.
        token_bytes_list = [self.vocab.get(i, b'') for i in ids]

        # Concatenate all byte chunks into one
        # Step 2: Concatenate Bytes. It joins all the small byte chunks into a single bytes object. b"".join() is the most efficient way to do this.
        total_bytes = b"".join(token_bytes_list)

        # Decode the final byte sequence into a string, replacing invalid UTF-8
        # sequences with the Unicode replacement character '�'.
        # Step 3: Decode to String. It decodes the final byte sequence into a standard Python string.
        # errors="replace" ensures it won't crash on invalid byte sequences, instead inserting the � character.
        return total_bytes.decode("utf-8", errors="replace")