"""Drop-in for transformers' torch_chunk_gated_delta_rule (Qwen3.5 GDN, no fla): the 63-step forward substitution
that builds (I + M)^-1 for the strictly lower chunk matrix M is one batched unit-triangular solve (same math; much
less autograd state, far fewer kernels)."""
import torch
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def fast_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                                use_qk_l2norm_in_kernel=False, **kwargs):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size)); key = F.pad(key, (0, 0, 0, pad_size)); value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size)); g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    m = ((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)  # strictly lower M
    eye = torch.eye(chunk_size, dtype=m.dtype, device=m.device)
    attn = torch.linalg.solve_triangular(m + eye, eye.expand_as(m), upper=False, unitriangular=True)  # (I + M)^-1
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
                            if initial_state is None else initial_state.to(value))
    outs = []
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]  # decay_mask is lower triangular
        v_new = v_i - k_cumdecay[:, :, i] @ last_recurrent_state
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        outs.append(attn_inter + attn_i @ v_new)
        last_recurrent_state = (last_recurrent_state * g[:, :, i, -1, None, None].exp()
                                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = torch.stack(outs, 2)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length].transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
