import torch
  
def pixel_wise_dot_product(x, K):
    """x.shape NCHW, 
    K.shape N,S,E"""
    n, c, h, w = x.size()
    _, cout, ck = K.size()
    assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
    y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
    return y.permute(0, 2, 1).view(n, cout, h, w)