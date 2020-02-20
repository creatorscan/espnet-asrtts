import torch


def subsequent_mask(size, device="cpu", dtype=torch.uint8):
    """Create mask for subsequent steps (1, size, size)

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)

def non_subsequent_mask(size, ctxt=-2, device="cpu", dtype=torch.uint8):
    """Create mask for subsequent steps (1, size, size)

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ones = torch.ones(size, size, device=device, dtype=dtype)
    ltri_ = torch.tril(ones, diagonal=0)
    ltri_d = torch.tril(ones, diagonal=ctxt)
    ret = ltri_ - ltri_d
    return ret
