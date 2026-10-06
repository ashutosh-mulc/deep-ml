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
    # - Chunk query tensor into num_kv_heads groups (torch.chunk)
    # - Compute attention scores for each head and take product with value vectors for each
    # - Concatenate the output vectors fro all 

    B, L, total_q_dim = Q.shape
    _, S, total_kv_dim = K.shape
    
    head_dim = total_q_dim // num_heads
    num_queries_per_kv = num_heads // num_kv_heads
    
    Q = Q.view(B, L, num_heads, head_dim).transpose(1, 2)
    K = K.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    V = V.view(B, S, num_kv_heads, head_dim).transpose(1, 2)


    K = torch.repeat_interleave(K, repeats=num_queries_per_kv, dim=1)
    V = torch.repeat_interleave(V, repeats=num_queries_per_kv, dim=1)


    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    output = output.transpose(1, 2).contiguous().view(B, L, total_q_dim)

    return output

     

    
