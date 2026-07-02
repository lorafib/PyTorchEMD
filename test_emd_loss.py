import torch
import numpy as np
from emd import earth_mover_distance


if not torch.cuda.is_available():
    raise SystemExit("SKIP: PyTorchEMD test requires a visible CUDA GPU.")


def make_points():
    p1 = torch.from_numpy(
        np.array([[[1.7, -0.1, 0.1], [0.1, 1.2, 0.3]]], dtype=np.float32)
    ).cuda()
    p2 = torch.from_numpy(
        np.array([[[0.3, 1.8, 0.2], [1.2, -0.2, 0.3]]], dtype=np.float32)
    ).cuda()
    return p1.repeat(3, 1, 1), p2.repeat(3, 1, 1)


def expected_normalized_emd(p1, p2):
    # The wrapper normalizes the raw matching cost by the number of source
    # points, so the expected value is mean squared transport cost per point.
    raw_cost = (
        ((p1[:, 0] - p2[:, 1]) ** 2).sum(dim=-1)
        + ((p1[:, 1] - p2[:, 0]) ** 2).sum(dim=-1)
    )
    return raw_cost / p1.shape[1]


weights = torch.tensor([0.5, 2.0, 1.0 / 3.0], device="cuda")

expected_p1, expected_p2 = make_points()
expected_p1.requires_grad = True
expected_p2.requires_grad = True
expected_dist = expected_normalized_emd(expected_p1, expected_p2)
expected_loss = (expected_dist * weights).sum()
expected_loss.backward()

emd_p1, emd_p2 = make_points()
emd_p1.requires_grad = True
emd_p2.requires_grad = True
emd_dist = earth_mover_distance(emd_p1, emd_p2, transpose=False)
emd_loss = (emd_dist * weights).sum()
emd_loss.backward()

torch.testing.assert_close(emd_dist, expected_dist, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(emd_loss, expected_loss, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(emd_p1.grad, expected_p1.grad, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(emd_p2.grad, expected_p2.grad, rtol=1e-5, atol=1e-5)

print("Input shapes:")
print(f"  p1: {tuple(emd_p1.shape)}")
print(f"  p2: {tuple(emd_p2.shape)}")
print()
print("Normalized EMD per batch item:")
print(f"  expected: {expected_dist.detach().cpu().tolist()}")
print(f"  actual:   {emd_dist.detach().cpu().tolist()}")
print()
print("Weighted normalized loss:")
print(f"  expected: {expected_loss.detach().cpu().item():.6f}")
print(f"  actual:   {emd_loss.detach().cpu().item():.6f}")
print()
print("Gradient check:")
print(f"  p1 max abs diff: {(emd_p1.grad - expected_p1.grad).abs().max().detach().cpu().item():.6e}")
print(f"  p2 max abs diff: {(emd_p2.grad - expected_p2.grad).abs().max().detach().cpu().item():.6e}")
print()
print("OK")
