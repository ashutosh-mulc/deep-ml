import torch
import torch.nn.functional as F

def grouped_query_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """
    Compute Grouped Query Attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    
    ## Algo:
    # - Compute attention scores for each head and take product with value vectors for each
    # - Concatenate the output vectors fro all 

    B, L, total_q_dim = Q.shape
    _, S, total_kv_dim = K.shape
    
    head_dim = total_q_dim // num_heads
    num_queries_per_kv = num_heads // num_kv_heads
    
    # 1. Reshape Q to explicitly separate the KV groups from the Q heads inside them
    # Shape: (B, num_kv_heads, num_queries_per_kv, L, head_dim)
    Q = Q.view(B, L, num_kv_heads, num_queries_per_kv, head_dim).permute(0, 2, 3, 1, 4)

    # 2. Reshape K and V, adding a dummy dimension of size 1 where the Q heads are
    # Shape: (B, num_kv_heads, 1, S, head_dim)
    K = K.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3).unsqueeze(2)
    V = V.view(B, S, num_kv_heads, head_dim).permute(0, 2, 1, 3).unsqueeze(2)

    # 3. Compute scores using broadcasting!
    # PyTorch automatically matches the '1' in K with the 'num_queries_per_kv' in Q 
    # WITHOUT copying any data in VRAM.
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
  
    output = output.permute(0, 3, 1, 2, 4).contiguous()
    output = output.view(B, L, total_q_dim)

    return output