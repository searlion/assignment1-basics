from __future__ import annotations

import multiprocessing
import os
from typing import IO, Any, BinaryIO, Counter
from collections.abc import Iterable

from einops import repeat, rearrange
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import Softmax2d
from tqdm import tqdm

from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.learning_rate_scheduler import learning_rate_scheduler
from cs336_basics.linear import Linear
from cs336_basics.multihead_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.positionwise_feedforward import PositionwiseFeedForward
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.softmax import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe_helper import _process_chunk, _get_pair_stats
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.transformer_lm import TransformerLM
from embedding import Embedding
from tests.common import gpt2_bytes_to_unicode


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    # 1. Instantiate your custom Linear module with the given dimensions.
    # The device and dtype of the weights and input features will be inferred.
    device = weights.device
    dtype = weights.dtype
    linear_module = Linear(in_features=d_in, out_features=d_out, device=device, dtype=dtype)

    # 2. Prepare the state dictionary for loading. The key 'W' must match
    #    the name of the parameter in your Linear class (self.W).
    state_dict_to_load = {"W": weights}

    # 3. Load the provided weights into your module instance.
    #    This is a key method provided by the nn.Module base class.
    linear_module.load_state_dict(state_dict_to_load)

    # 4. Apply the linear transformation by calling the module.
    #    This invokes the forward() method of your linear_module.
    output = linear_module(in_features)

    return output


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    device = weights.device
    dtype = weights.dtype
    embedding_module = Embedding(vocab_size, d_model, device=device, dtype=dtype)
    embedding_module.load_state_dict({"weight": weights})
    output = embedding_module(token_ids)
    return output


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # 1. Instantiate our feed-forward network with the specified dimensions.
    swiglu_ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

    # 2. Manually load the provided weights into the model's layers.
    # We use torch.no_grad() to perform these operations without tracking gradients.
    # .copy_() is an in-place operation that is robust for this task.
    with torch.no_grad():
        swiglu_ffn.gate_proj.weight.copy_(w1_weight)
        swiglu_ffn.down_proj.weight.copy_(w2_weight)
        swiglu_ffn.up_proj.weight.copy_(w3_weight)

    # 3. Run the forward pass with the input features.
    output = swiglu_ffn(in_features)

    return output


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    attention_module = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

    # --- THE FIX ---
    # The provided weights are already for all heads.
    # q_proj_weight is shape (d_model, d_model) = (64, 64)
    # k_proj_weight is shape (d_model, d_model) = (64, 64)
    # v_proj_weight is shape (d_model, d_model) = (64, 64)
    # We simply concatenate them.
    W_qkv = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    # The resulting shape is (3 * d_model, d_model) = (192, 64). This now matches
    # the shape of the `to_qkv.weight` in our module.
    # --- END FIX ---

    attention_module.to_qkv.weight.data.copy_(W_qkv)
    attention_module.to_out.weight.data.copy_(o_proj_weight)

    attention_module.eval()
    return attention_module(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # This function uses the same logic but without a helper class.
    # The logic here must also be updated.

    *_, seq_len, _ = in_features.shape
    d_k = d_model // num_heads

    # --- THE FIX ---
    # Same fix as above: just concatenate the provided full projection weights.
    W_qkv = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    # Shape of W_qkv is now (192, 64)
    # --- END FIX ---

    # in_features: (b, t, 64), W_qkv: (192, 64) -> qkv: (b, t, 192)
    qkv = torch.einsum('... t i, o i -> ... t o', in_features, W_qkv)

    # Now, rearrange will work because qkv's last dim (192)
    # matches the product of qkv=3, h=4, d=16.
    q, k, v = rearrange(qkv, '... t (qkv h d) -> qkv ... h t d', qkv=3, h=num_heads)

    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_features.device
    )

    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device)

    q = rope(q, token_positions)
    k = rope(k, token_positions)

    dots = torch.einsum('... h i d, ... h j d -> ... h i j', q, k) / (d_k ** 0.5)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=1).bool()
    dots.masked_fill_(causal_mask, float('-inf'))

    attn_weights = dots.softmax(dim=-1)
    attended_v = torch.einsum('... h i j, ... h j d -> ... h i d', attn_weights, v)

    concatenated_heads = rearrange(attended_v, '... h t d -> ... t (h d)')

    output = torch.einsum('... t i, o i -> ... t o', concatenated_heads, o_proj_weight)

    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    result = rope(in_query_or_key, token_positions)
    return result


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    # Instantiate the TransformerBlock with the given hyperparameters.
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    )

    # The provided weights dictionary has keys that need to be mapped to the
    # parameter names in our implementation.
    # - `RMSNorm` uses a gain parameter named `g`.
    # - `PositionwiseFeedForward` uses `gate_proj`, `up_proj`, `down_proj`.
    state_dict = {
        "ln1.g": weights["ln1.weight"],
        "attn.q_proj.weight": weights["attn.q_proj.weight"],
        "attn.k_proj.weight": weights["attn.k_proj.weight"],
        "attn.v_proj.weight": weights["attn.v_proj.weight"],
        "attn.output_proj.weight": weights["attn.output_proj.weight"],
        "ln2.g": weights["ln2.weight"],
        "ffn.gate_proj.weight": weights["ffn.w1.weight"],
        "ffn.up_proj.weight": weights["ffn.w3.weight"],
        "ffn.down_proj.weight": weights["ffn.w2.weight"],
    }
    block.load_state_dict(state_dict)

    # Move the model to the same device as the input features and set to evaluation mode.
    device = in_features.device
    block.to(device)
    block.eval()

    # RoPE requires the absolute positions of tokens in the sequence.
    _, seq_len, _ = in_features.shape
    token_positions = torch.arange(seq_len, device=device)

    # Perform the forward pass.
    with torch.no_grad():
        output = block(in_features, token_positions)

    return output


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    # Instantiate the full language model.
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )

    # Create a new state dictionary by mapping the provided weight keys
    # to the parameter names in our model implementation.
    state_dict = {}
    state_dict["token_embeddings.weight"] = weights["token_embeddings.weight"]
    state_dict["ln_final.g"] = weights["ln_final.weight"]
    state_dict["lm_head.W"] = weights["lm_head.weight"]

    for i in range(num_layers):
        layer_prefix = f"layers.{i}"
        # Attention block weights
        state_dict[f"{layer_prefix}.attn.q_proj.weight"] = weights[f"{layer_prefix}.attn.q_proj.weight"]
        state_dict[f"{layer_prefix}.attn.k_proj.weight"] = weights[f"{layer_prefix}.attn.k_proj.weight"]
        state_dict[f"{layer_prefix}.attn.v_proj.weight"] = weights[f"{layer_prefix}.attn.v_proj.weight"]
        state_dict[f"{layer_prefix}.attn.output_proj.weight"] = weights[f"{layer_prefix}.attn.output_proj.weight"]
        # FFN block weights (mapping w1, w2, w3 to our names)
        state_dict[f"{layer_prefix}.ffn.gate_proj.weight"] = weights[f"{layer_prefix}.ffn.w1.weight"]
        state_dict[f"{layer_prefix}.ffn.up_proj.weight"] = weights[f"{layer_prefix}.ffn.w3.weight"]
        state_dict[f"{layer_prefix}.ffn.down_proj.weight"] = weights[f"{layer_prefix}.ffn.w2.weight"]
        # Normalization block weights (mapping .weight to .g)
        state_dict[f"{layer_prefix}.ln1.g"] = weights[f"{layer_prefix}.ln1.weight"]
        state_dict[f"{layer_prefix}.ln2.g"] = weights[f"{layer_prefix}.ln2.weight"]

    # Load the remapped weights into the model.
    model.load_state_dict(state_dict)

    # Move model to the correct device and set to evaluation mode.
    device = in_indices.device
    model.to(device)
    model.eval()

    # Perform the forward pass without tracking gradients.
    with torch.no_grad():
        output = model(in_indices)

    return output


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    # 1. Instantiate your custom Linear module with the given dimensions.
    # The device and dtype of the weights and input features will be inferred.
    device = weights.device
    dtype = weights.dtype
    rms_norm = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)

    # 2. Prepare the state dictionary for loading. The key 'W' must match
    #    the name of the parameter in your Linear class (self.W).
    state_dict_to_load = {"g": weights}

    # 3. Load the provided weights into your module instance.
    #    This is a key method provided by the nn.Module base class.
    rms_norm.load_state_dict(state_dict_to_load)

    # 4. Apply the linear transformation by calling the module.
    #    This invokes the forward() method of your linear_module.
    output = rms_norm(in_features)

    return output


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim)


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return learning_rate_scheduler(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
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
    return vocab, merges
