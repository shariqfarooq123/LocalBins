import torch


class RandomBBoxQueries(object):
    def __init__(self, batch_size, h, w, window_sizes, N=100):
        b = batch_size
        self.h, self.w = h, w
        queries = {}
        self.window_sizes = window_sizes

        for win_size in window_sizes:
            # queries[win_size]
            k = win_size // 2
            x = torch.randint(k+1, w - k, (b, N, 1))
            y = torch.randint(k+1, h - k, (b, N, 1))
            queries[win_size] = torch.cat((x,y), dim=-1)

        self.absolute = queries
        self.normalized = self._normalized()

    def _normalized(self):
        """returns queries in -1,1 range"""
        normed = {}
        for win_size, coords in self.absolute.items():
            c = coords.clone().float()
            c[:,:,0] = c[:,:,0] / (self.w - 1)  # w - 1 because range is [-1,1]
            c[:,:,1] = c[:,:,1] / (self.h - 1)
            normed[win_size] = c
        return normed

    def to(self, device):
        for win_size in self.window_sizes:
            self.absolute[win_size] = self.absolute[win_size].to(device)
            self.normalized[win_size] = self.normalized[win_size].to(device)
        return self

    def __repr__(self):
        return str(self.normalized)


def fetch_depth_values(depths, bbox_queries, min_depth=1e-3, max_depth=10):
    b, c, h, w = depths.shape
    window_sizes = bbox_queries.window_sizes
    points = {}
    for win_size in window_sizes:
        # target points
        with torch.no_grad():
            # # target points
            # # to get target points, create an list of indices that cover the bboxes and index the depth map
            # # mask out that are invalid
            # # skip if too many are masked or all are masked
            abs_coords = bbox_queries.absolute[win_size]  # B, N, 2; babs[b,n] = [x,y]
            # print(abs_coords.shape)
            # xc, yc = abs_coords[...,0], abs_coords[...,1]
            B, N, _two = abs_coords.shape
            assert B == b, f"Batch sizes should match, got {b} whereas queries have batch size {B}"

            k = win_size // 2
            # base_xx, baseyy = np.tile(range(width), height), np.repeat(range(height), width)
            x = torch.arange(-k, k+1)
            y = torch.arange(-k, k+1)
            Y, X = torch.meshgrid(y, x)
            base_coords = torch.stack((X, Y), dim=0)[None, None,...].to(depths.device)  # .shape 1, 1, 2, k, k
            
            coords = abs_coords[...,None,None] + base_coords  # shape B, N, 2, k, k
            
            x = coords[:,:,0,:,:]
            y = coords[:,:,1,:,:]
            flatten_indices = y * w + x  # .shape B, N, k, k
            
            flatten_flatten_indices = flatten_indices.flatten(2)  # .shape B, N, kxk
            d = depths.expand(-1,N,-1,-1).flatten(2)  # .shape B, N, HxW
            # print(depths.shape, flatten_flatten_indices.shape, flatten_flatten_indices.max(), flatten_flatten_indices.min())
            
            # target_points = depths[flatten_flatten_indices]  # .shape B, N, kxk
            target_points = torch.gather(d, dim=-1, index=flatten_flatten_indices)

            points[win_size] = target_points
    return points


def fetch_features_at_bbox_center(feat, bbox_queries):
    b, c, h, w = feat.shape
    window_sizes = bbox_queries.window_sizes
    points = {}
    for win_size in window_sizes:
        # target points
        with torch.no_grad():
            # # target points
            # # to get target points, create an list of indices that cover the bboxes and index the depth map
            # # mask out that are invalid
            # # skip if too many are masked or all are masked
            abs_coords = bbox_queries.absolute[win_size]  # B, N, 2; babs[b,n] = [x,y]
            # print(abs_coords.shape)
            # xc, yc = abs_coords[...,0], abs_coords[...,1]
            B, N, _two = abs_coords.shape
            assert B == b, f"Batch sizes should match, got {b} whereas queries have batch size {B}"

            # k = win_size // 2
            # # base_xx, baseyy = np.tile(range(width), height), np.repeat(range(height), width)
            # x = torch.arange(-k, k+1)
            # y = torch.arange(-k, k+1)
            # Y, X = torch.meshgrid(y, x)
            # base_coords = torch.stack((X, Y), dim=0)[None, None,...].to(depths.device)  # .shape 1, 1, 2, k, k
            
            # coords = abs_coords[...,None,None]  # shape B, N, 2, k, k
            
            x = abs_coords[:,:,0]
            y = abs_coords[:,:,1]
            flatten_indices = y * w + x  # .shape B, N
            flatten_indices = flatten_indices[...,None, None].expand(-1,-1,c,-1) #.shape B, N, C, 1
            
            d = feat.flatten(2,3).unsqueeze(1)  # B,1, C, HW
            d = d.expand(-1,N,-1,-1)  # .shape B, N, C, HxW
            
            fetched = torch.gather(d, dim=-1, index=flatten_indices)

            points[win_size] = fetched
    return points
        



class RandomCenteredBBoxQueries(object):
    def __init__(self, batch_size, h, w, window_sizes, N=100):
        b = batch_size
        self.h, self.w = h, w
        queries = {}
        self.window_sizes = window_sizes

        max_win_size = max(window_sizes)
        k = max_win_size // 2
        x = torch.randint(k+1, w - k, (b, N, 1))
        y = torch.randint(k+1, h - k, (b, N, 1))
        for win_size in window_sizes:
            # queries[win_size]

            queries[win_size] = torch.cat((x,y), dim=-1)

        self.absolute = queries
        self.normalized = self._normalized()

    def _normalized(self):
        """returns queries in -1,1 range"""
        normed = {}
        for win_size, coords in self.absolute.items():
            c = coords.clone().float()
            c[:,:,0] = c[:,:,0] / (self.w - 1)  # w - 1 because range is [-1,1]
            c[:,:,1] = c[:,:,1] / (self.h - 1)
            normed[win_size] = c
        return normed

    def to(self, device):
        for win_size in self.window_sizes:
            self.absolute[win_size] = self.absolute[win_size].to(device)
            self.normalized[win_size] = self.normalized[win_size].to(device)
        return self

    def __repr__(self):
        return str(self.normalized)



class CenteredBBoxQueries(object):
    def __init__(self, xc, yc, batch_size, h, w, window_sizes, N=100):
        b = batch_size
        self.h, self.w = h, w
        queries = {}
        self.window_sizes = window_sizes

        # max_win_size = max(window_sizes)
        # k = max_win_size // 2
        x = torch.Tensor([xc]).view(1,1,1).expand(batch_size, N, -1)
        y = torch.Tensor([yc]).view(1,1,1).expand(batch_size, N, -1)
        for win_size in window_sizes:
            # queries[win_size]

            queries[win_size] = torch.cat((x,y), dim=-1)

        self.absolute = queries
        self.normalized = self._normalized()

    def _normalized(self):
        """returns queries in -1,1 range"""
        normed = {}
        for win_size, coords in self.absolute.items():
            c = coords.clone().float()
            c[:,:,0] = c[:,:,0] / (self.w - 1)  # w - 1 because range is [-1,1]
            c[:,:,1] = c[:,:,1] / (self.h - 1)
            normed[win_size] = c
        return normed

    def to(self, device):
        for win_size in self.window_sizes:
            self.absolute[win_size] = self.absolute[win_size].to(device)
            self.normalized[win_size] = self.normalized[win_size].to(device)
        return self

    def __repr__(self):
        return str(self.normalized)


def bboxes_to_masks(bbox_queries, h, w):
    
    window_sizes = bbox_queries.window_sizes
    masks = {}
    for win_size in window_sizes:
        with torch.no_grad():
            abs_coords = bbox_queries.absolute[win_size]  # B, N, 2; babs[b,n] = [x,y]
            # print(abs_coords.shape)
            # xc, yc = abs_coords[...,0], abs_coords[...,1]
            B, N, _two = abs_coords.shape

            k = win_size // 2
            # base_xx, baseyy = np.tile(range(width), height), np.repeat(range(height), width)
            x = torch.arange(-k, k+1)
            y = torch.arange(-k, k+1)
            Y, X = torch.meshgrid(y, x)
            base_coords = torch.stack((X, Y), dim=0)[None, None,...] # .shape 1, 1, 2, k, k
            
            coords = abs_coords[...,None,None] + base_coords  # shape B, N, 2, k, k
            
            x = coords[:,:,0,:,:]
            y = coords[:,:,1,:,:]
            flatten_indices = y * w + x  # .shape B, N, k, k
            
            flatten_flatten_indices = flatten_indices.flatten(2).long()  # .shape B, N, kxk
            mask = torch.zeros((B, N, h*w))
            mask.scatter_(-1, flatten_flatten_indices, 1)

            masks[win_size] = mask.view(B, N, h, w)
    return masks