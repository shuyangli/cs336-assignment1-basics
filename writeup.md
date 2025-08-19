## unicode1

a. `chr(0)` returns the Null unicode character.
b. The string representation is a 0 byte (\x00), and the printed representation is the empty string.
c. In text it functions the same way: string representation is a 0 byte, and printed representation is empty string.


## unicode2

a. UTF-8 uses fewer bytes per character; UTF-16 and UTF-32 uses a minimum of 2/4 bytes per character, and UTF-8 uses only one. This means our vocabulary size for UTF-8 encoded strings can be smaller.
b. The functions decodes each byte one by one, but not each character is one byte in UTF-8. Any character taking more than 1 byte (e.g. a Chinese character) will break that function.
c. '\xff\x01': byte 0 doesn't follow the expected encoding.


## Transformer resource accounting

Let's write down all the operations involved in a transformer LM in the forward pass:

- Tokenization (no matmul) - output matrix is Int[batch_size seq_len]
- Embed (just a lookup, no matmul) - output matrix is Float[batch_size seq_len d_model]
- For K layers of the transformer block, each layer performs:
  * RMSNorm: (no matmul)
  * Multihead Self-Attention: input matrix is Float[batch_size seq_len d_model]
    * Query Projection: Float[batch_size seq_len d_model] @ Float[d_model hd_qk] => Float[batch_size seq_len hd_qk]
    * Key Projection: Float[batch_size seq_len d_model] @ Float[d_model hd_qk] => Float[batch_size seq_len hd_qk]
    * Value Projection: Float[batch_size seq_len d_model] @ Float[d_model hd_v] => Float[batch_size seq_len hd_v]
    * Output Projection: Float[batch_size seq_len hd_v] @ Float[hd_v d_model] => Float[batch_size seq_len d_model]
        * Projection FLOPs is 2 * batch_size * seq_len * d_model * (2 hd_qk + 2 hd_v) = 8 * batch_size * seq_len * d_model^2
    * RoPE: mostly a pointwise and stacking operation, no matmul
    * Attention:
        * query @ key = Float[batch_size seq_len hd_qk] @ Float[batch_size seq_len hd_qk] => Float[batch_size seq_len seq_len]
        * softmax
        * softmax(qk^T) @ value = Float[batch_size seq_len seq_len] @ Float[batch_size seq_len hd_v] => Float[batch_size seq_len hd_v]
    * Attention FLOPs is 2 * 2 * batch_size * seq_len^2 * d_model
  * RMSNorm: (no matmul)
  * FFN (SwiGLU):
    * Float[batch_size seq_len d_model] @ Float[d_model d_hidden] => Float[batch_size seq_len d_hidden]
    * Float[batch_size seq_len d_model] @ Float[d_model d_hidden] => Float[batch_size seq_len d_hidden]
    * elementwise
    * Float[batch_size seq_len d_hidden] @ Float[d_hidden d_model] => Float[batch_size seq_len d_model]
    * FFN FLOPs: 3 * 2 * batch_size * seq_len * d_model * d_hidden; d_hidden = 8/3 * d_model
  * **Total Per-layer FLOPs:** 8 * batch_size * seq_len * d_model^2 + 4 * batch_size * seq_len^2 * d_model + 16 * batch_size * seq_len * d_model^2
    = 24 * batch_size * seq_len * d_model^2 + 4 * batch_size * seq_len^2 * d_model
- Final normalization (RMSNorm): (no matmul)
- Final linear layer for output: Float[batch_size seq_len d_model] @ Float[d_model vocab_size] => Float[batch_size seq_len vocab_size]
    * LM head FLOPs: 2 * batch_size * seq_len * d_model * vocab_size
- Final softmax
- Final unembed (just lookup, not a matmul)

## transformer_accounting problem

(a) For GPT-2 XL, params accounting:
* Embed: Float[vocab_size d_model] = vocab_size * d_model params
* For each attention layer:
  * q_proj = k_proj: d_model * num_heads * d_qk == d_model^2
  * v_proj = output_proj: d_model * num_heads * d_v = d_model^2
  * FFN: linear1 = linear2 = linear3 = d_model * d_hidden
* LM head layer: d_model * vocab_size params
* Unembed: d_model * vocab_size params (not always the same as the embed matrix; for larger models this hurts performance)

Total trainable parameters: vocab_size * d_model * 2 + num_layers * (4 * d_model^2 + 3 * d_model * d_hidden)
    = 1.608e8 + 1.967e9 ~= 2.1e9 (2 billion parameters)

If each is a single precision floating point: each param is 4 bytes (32 bits), so this is 8B bytes ~= 8GB of memory.

(b) For GPT-2 XL, each forward pass needs num_layers * (24 * batch_size * seq_len * d_model^2 + 4 * batch_size * seq_len^2 * d_model) + 2 * batch_size * seq_len * d_model * vocab_size

This is 4.5e12 FLOPs.
~ Projection is 2.1e10 FLOPs per layer
~ Attn is 6.7e9 FLOPs per layer
~ FFN is 6.3e10 FLOPs per layer
~ LM head is 1.6e11 FLOPs

(c) FFN is the most expensive FLOPs-wise.

(d)

| | **GPT-2 Small** | **GPT-2 Medium** | **GPT-2 Large** | **GPT-2 XL** |
| :--- | :---: | :---: | :---: | :---: |
| **vocab_size** | 50257 | 50257 | 50257 | 50257 |
| **context_length** | 1024 | 1024 | 1024 | 1024 |
| **num_layers** | 12 | 24 | 36 | 48 |
| **d_model** | 768 | 1024 | 1280 | 1600 |
| **num_heads** | 12 | 16 | 20 | 25 |
| **d_ff** | 6400 | 6400 | 6400 | 6400 |
| **d_qk = d_v** | 64 | 64 | 64 | 64 |
| **batch_size** | 1 | 1 | 1 | 1 |
| **Trainable parameters** | | | | |
| **Embed** | 3.860E+07 | 5.146E+07 | 6.433E+07 | 8.041E+07 |
| **Per-layer Q/K Proj** | 5.898E+05 | 1.049E+06 | 1.638E+06 | 2.560E+06 |
| **Per-layer V/O Proj** | 5.898E+05 | 1.049E+06 | 1.638E+06 | 2.560E+06 |
| **Each of the 3 linear layer in FFN** | 4.915E+06 | 6.554E+06 | 8.192E+06 | 1.024E+07 |
| **LM head** | 3.860E+07 | 5.146E+07 | 6.433E+07 | 8.041E+07 |
| **Total parameters** | 2.825E+08 | 6.754E+08 | 1.249E+09 | 2.127E+09 |
| **FLOPs** | | | | |
| **Per-layer QKVO Proj (total)** | 4.832E+09 | 8.590E+09 | 1.342E+10 | 2.097E+10 |
| **Per-layer Attention** | 3.221E+09 | 4.295E+09 | 5.369E+09 | 6.711E+09 |
| **Per-layer FFN** | 3.020E+10 | 4.027E+10 | 5.033E+10 | 6.291E+10 |
| **LM head** | 7.905E+10 | 1.054E+11 | 1.317E+11 | 1.647E+11 |
| **Total FLOPs** | 5.381E+11 | 1.381E+12 | 2.620E+12 | 4.513E+12 |

(e) If we scale the context-length to 16,384: attention becomes the bottleneck (it scales quadratically, whereas the rest scales linearlly)


## Learning Rate tuning

With learning rate = 1e0, loss gradually decays; with learning rate = 1e1, loss decays faster; at 1e2 it decays very rapidly to 0 for the simple case, and at 1e3 it never converged.