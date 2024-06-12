import math,torch,numpy as np

def ts2tsemb1d(ts, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    ts = ts * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ts.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos = ts[..., 0, None] / dim_t
    posemb = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb

def time_position_embedding(track_num, frame_num, embed_dims, device):
    ts = torch.arange(0, 1 + 1e-5, 1/(frame_num - 1), dtype=torch.float32, device=device)
    ts = ts[None, :] * torch.ones((track_num, frame_num), dtype=torch.float32, device=device)
    ts_embed = ts2tsemb1d(ts.view(track_num * frame_num, 1), num_pos_feats=embed_dims).view(track_num, frame_num, embed_dims)
    return ts_embed