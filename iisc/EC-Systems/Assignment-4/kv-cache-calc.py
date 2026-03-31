import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KV Cache Calculator")

    # Model Specification
    model = parser.add_argument_group("Model Specification")
    model.add_argument("-l", "--layers", type=int, default=32, help="Number of layers")
    model.add_argument("-qh", "--q-heads", type=int, default=32, help="Number of query heads")
    model.add_argument("-kvh", "--kv-heads", type=int, default=8, help="Number of KV heads")
    model.add_argument("-h1", "--h1", type=int, default=4096, help="Embedding dimension (h1)")
    model.add_argument("-h2", "--h2", type=int, default=14336, help="Inner dimension (h2)")
    model.add_argument("-v", "--vocab", type=int, default=128256, help="Vocab size")

    # Device Specification
    device = parser.add_argument_group("Device Specification")
    device.add_argument("-mem", "--gpu-mem", type=float, default=80, help="GPU memory in GB")
    device.add_argument("-tp", "--tp-size", type=int, default=1, help="Tensor parallel size")

    # Input Specification
    inp = parser.add_argument_group("Input Specification")
    inp.add_argument("-s", "--seq-len", type=int, required=True, help="Input Sequence Length")

    # Quantization
    quant = parser.add_argument_group("Quantization")
    quant.add_argument("-wb", "--w-bytes", type=int, required=True, help="Bytes per weight (e.g., 2 for 16-bit)")
    quant.add_argument("-kvb", "--kv-bytes", type=int, required=True, help="Bytes per KV cache element")

    return parser.parse_args()


def calculate_kv_cache(
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    inner_dim: int,
    vocab_size: int,
    gpu_memory_gb: float,
    tp_size: int,
    input_seq_len: int,
    weight_bytes: int,
    kv_cache_bytes: int,
) -> dict:
    cols_per_head = embed_dim // num_query_heads
    
    # A transformer architecture has multiple layers.
    # Each layer contains multiple transformer blocks.
    # Each transformer blocks is made of multiple attention heads, an output weight matrix, and a Feed-Forward layer.
    # The parameters also include the entire Vocab Size and it's corresponding embeddings.
    
    # Number of Vocab Embeddings - Input embeddings for each token ID
    num_params_embed_vocab = vocab_size * embed_dim
    
    # Number of Attention Parameters in a single transformer block
    num_weights_q_heads = num_query_heads * (embed_dim * cols_per_head) # Dimension of each head matrix
    num_weights_kv_heads = 2 * num_kv_heads * (embed_dim * cols_per_head) # 2 --> Key and Value
    num_weights_output_matrix = embed_dim * embed_dim # Single matrix ---> Same size as Concat(Heads)
    num_params_attn_per_layer = num_weights_q_heads + num_weights_kv_heads + num_weights_output_matrix
    
    # Number of Parameters in Feed Forward - 2 Layers - Up and Down Projection
    # In models using SwiGLU activation, Up Projection uses 2 matrices - Up and Gate Projection - Both of size (embed_dim x inner_dim)
    # We calculate using a single matrix for up projection here.
    # Down Projection uses 1 matrix of size (inner_dim x embed_dim)
    num_weights_up_proj = 2 * embed_dim * inner_dim
    num_weights_down_proj = inner_dim * embed_dim
    num_params_ffn_per_layer = num_weights_up_proj + num_weights_down_proj
    
    # We add all the parameters together.
    # We also add the vocab embedding parameters at the end for conversion of output logits to token IDs, common in many transformers.
    total_params = num_params_embed_vocab + (num_params_attn_per_layer + num_params_ffn_per_layer) * num_layers + num_params_embed_vocab
    params_bytes = total_params * weight_bytes # Memory required by weights
    
    # With tensor parallelism, model weights are split across tp_size GPUs
    params_bytes_per_gpu = params_bytes / tp_size
    
    # Total KV Cache Elements - Each key and value head stores a matrix of size (input_seq_len x cols_per_head) for each layer
    # With tensor parallelism, KV heads are split across tp_size GPUs
    kv_cache_elements = (2 * num_kv_heads) * (input_seq_len * cols_per_head) * num_layers
    # KV Cache memory required per request per GPU
    kvc_mem_per_req_gpu = (kv_cache_elements * kv_cache_bytes) / tp_size

    max_batch_size = int((gpu_memory_gb * (1024 ** 3) - params_bytes_per_gpu) / kvc_mem_per_req_gpu)
    
    return {
        "Total Model Parameters": total_params,
        "Total Model Parameter memory (GB)": params_bytes / (1024 ** 3),
        "Model Memory per GPU (GB)": params_bytes_per_gpu / (1024 ** 3),
        "KV Cache Memory per Request per GPU (GB)": kvc_mem_per_req_gpu / (1024 ** 3),
        "Maximum Batch Size": max_batch_size,
    }

if __name__ == "__main__":
    args = parse_args()
    results = calculate_kv_cache(
        num_layers=args.layers,
        num_query_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        embed_dim=args.h1,
        inner_dim=args.h2,
        vocab_size=args.vocab,
        gpu_memory_gb=args.gpu_mem,
        tp_size=args.tp_size,
        input_seq_len=args.seq_len,
        weight_bytes=args.w_bytes,
        kv_cache_bytes=args.kv_bytes,
    )
    for key, value in results.items():
        print(f"{key}: {value}")
