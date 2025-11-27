# import tqdm
# import pickle
import numpy as np
import torch
# import PIL.Image
# import dnnlib
import torch.nn.functional as F
# from torch_utils import distributed as dist
# import scipy.io
def get_ns_bounded_loss( u,v, device=torch.device('cuda'), mask_region=None):
    """Return the loss of the bounded NS equation and the observation loss.
    
    Args:
        u, v: Input tensors
        device: Device to use
        mask_region: Tuple of (x_start, x_end, y_start, y_end) coordinates to mask to zero
                    If None, no masking is applied
    """
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float32, device=device).view(1, 1, 3, 1) / 2
    deriv_x=deriv_x.repeat(1,10,1,1)
    deriv_y=deriv_y.repeat(1,10,1,1)
    grad_x_next_x_u = F.conv2d(u, deriv_x, padding=(0, 1))
    pde_loss_u = grad_x_next_x_u 
    print("pde_loss_u",pde_loss_u.shape)
    grad_x_next_y_v = F.conv2d(v, deriv_y, padding=(1, 0))
    pde_loss_v = grad_x_next_y_v
    pde_loss=pde_loss_u-pde_loss_v
    pde_loss = pde_loss.squeeze()
   # Apply boundary masking
    pde_loss[0, :] = 0
    pde_loss[-1, :] = 0
    pde_loss[:, 0] = 0
    pde_loss[:, -1] = 0
    
    # Apply custom mask region if specified
    if mask_region is not None:
        x_start, x_end, y_start, y_end = mask_region
        pde_loss[x_start:x_end, y_start:y_end] = 0
    
    return pde_loss

def get_ns_bounded_loss_with_mask(u, v, mask, mask_region=None,device=torch.device('cuda')):
    """
    Compute bounded NS loss with per-frame mask.
    """
    # 确保输入和 mask 在同一个 device
    u = u.to(device)
    v = v.to(device)
    mask = 1.0-mask.to(device).float()

    B, C, H, W = u.shape
    assert v.shape == u.shape, "u and v must have same shape"
    assert mask.shape == u.shape, "mask must have same shape as u and v"

    # Gradient kernels
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float32, device=device).view(1, 1, 3, 1) / 2
    deriv_x = deriv_x.repeat(C, 1, 1, 1)  # [C,1,1,3]
    deriv_y = deriv_y.repeat(C, 1, 1, 1)  # [C,1,3,1]

    # Compute gradients per channel
    grad_x_next_x_u = F.conv2d(u, deriv_x, padding=(0, 1), groups=C)
    grad_x_next_y_v = F.conv2d(v, deriv_y, padding=(1, 0), groups=C)

    pde_loss = grad_x_next_x_u - grad_x_next_y_v  # [1,10,96,96]

    # Boundary masking
    pde_loss[:, :, 0, :] = 0
    pde_loss[:, :, -1, :] = 0
    pde_loss[:, :, :, 0] = 0
    pde_loss[:, :, :, -1] = 0
    if mask_region is not None:
        x_start, x_end, y_start, y_end = mask_region
        pde_loss[:, :,x_start:x_end, y_start:y_end] = 0
        pde_loss = pde_loss * mask

    # Apply mask
    else:
        pde_loss = pde_loss * mask

    return pde_loss

def get_ns_bounded_loss_perframe(u, v, mask, mask_region=None,device=torch.device('cuda')):
    """
    Compute bounded NS loss with per-frame mask.
    """
    # 确保输入和 mask 在同一个 device
    u = u.to(device)
    v = v.to(device)
    # mask = 1.0-mask.to(device).float()

    B, C, H, W = u.shape
    assert v.shape == u.shape, "u and v must have same shape"
    # assert mask.shape == u.shape, "mask must have same shape as u and v"

    # Gradient kernels
    deriv_x = torch.tensor([[1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor([[1], [0], [-1]], dtype=torch.float32, device=device).view(1, 1, 3, 1) / 2
    deriv_x = deriv_x.repeat(C, 1, 1, 1)  # [C,1,1,3]
    deriv_y = deriv_y.repeat(C, 1, 1, 1)  # [C,1,3,1]

    # Compute gradients per channel
    grad_x_next_x_u = F.conv2d(u, deriv_x, padding=(0, 1), groups=C)
    grad_x_next_y_v = F.conv2d(v, deriv_y, padding=(1, 0), groups=C)

    pde_loss = grad_x_next_x_u - grad_x_next_y_v  # [1,10,96,96]

    # Boundary masking
    pde_loss[:, :, 0, :] = 0
    pde_loss[:, :, -1, :] = 0
    pde_loss[:, :, :, 0] = 0
    pde_loss[:, :, :, -1] = 0
    # if mask_region is not None:
    #     x_start, x_end, y_start, y_end = mask_region
    #     pde_loss[:, :,x_start:x_end, y_start:y_end] = 0
    #     pde_loss = pde_loss * mask

    # # Apply mask
    # else:
    #     pde_loss = pde_loss * mask

    return pde_loss