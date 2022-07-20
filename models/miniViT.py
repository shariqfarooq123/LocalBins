import torch
import torch.nn as nn

from .layers import PatchTransformerEncoder, PixelWiseDotProduct


class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.embedding_dim = embedding_dim
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)  # TODO use conv3x3 output for dot product calculation. Use direct supervision for delta R
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x, return_D=False, return_queries=False, return_target_rest=False):
        # n, c, h, w = x.size()
        outputs = []
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)
        if return_D:
            outputs.append(x)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]
        tgt_rest = tgt[1+self.n_query_channels:, ... ]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2).contiguous()
        tgt_rest = tgt_rest.permute(1, 0, 2).contiguous()

        if return_queries:
            outputs.append(queries)
        if return_target_rest:
            outputs.append(tgt_rest)
        
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)

        outputs.extend([y, range_attention_maps])
        return tuple(outputs)