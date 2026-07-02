import warnings

import torch
import emd_cuda


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


def _normalize_point_limit(point_limit):
    if point_limit is None:
        return None
    point_limit = int(point_limit)
    if point_limit <= 0:
        return None
    return point_limit


def _downsample_points(xyz, point_limit, name):
    point_limit = _normalize_point_limit(point_limit)
    if point_limit is None or xyz.shape[1] <= point_limit:
        return xyz

    indices = torch.linspace(0, xyz.shape[1] - 1, point_limit, device=xyz.device).long()
    warnings.warn(
        f"Downsampling {name} for EMD from {xyz.shape[1]} to {point_limit} points.",
        RuntimeWarning,
        stacklevel=2,
    )
    return xyz.index_select(1, indices)


def earth_mover_distance(xyz1, xyz2, transpose=True, point_limit=None):
    """Earth Mover Distance (Approx).

    Args:
        xyz1 (torch.Tensor): (B, 3, N1) if transpose=True, else (B, N1, 3).
        xyz2 (torch.Tensor): (B, 3, N2) if transpose=True, else (B, N2, 3).
        transpose (bool): whether to transpose inputs from BCN to BNC format.
            Extensions only support BNC format.
        point_limit (Optional[int]): maximum number of points per cloud before
            calling the CUDA extension. ``None`` or ``<= 0`` disables limiting.

    Returns:
        cost (torch.Tensor): normalized cost with shape (B,).
    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)

    xyz1 = _downsample_points(xyz1, point_limit, "xyz1")
    xyz2 = _downsample_points(xyz2, point_limit, "xyz2")

    cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
    # Normalize by the size of the source point cloud, following
    # https://cocalc.com/github/alexzhou907/pvd/blob/main/metrics/PyTorchEMD/emd.py
    cost = cost / xyz1.shape[1]
    return cost
