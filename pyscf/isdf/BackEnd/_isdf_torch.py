import torch
import numpy


def distance_translation(pa: torch.Tensor, pb: torch.Tensor, a: torch.Tensor):
    """
    Calculate the distances between multiple points pa and pb, taking periodic boundary conditions into account.
    PyTorch version.

    :param pa: Coordinates of points a, shape (n, 3)
    :param pb: Coordinates of points b, shape (m, 3)
    :param a: Lattice vectors (assumed to be a 3x3 tensor)
    :return: Minimum distances considering periodic boundary conditions, shape (n, m)
    """
    # Reshape pa and pb for broadcasting
    pa = pa.unsqueeze(1)  # Shape becomes (n, 1, 3)
    pb = pb.unsqueeze(0)  # Shape becomes (1, m, 3)

    # Calculate differences
    diff = pa - pb  # Shape becomes (n, m, 3)

    # Apply periodic boundary conditions
    for i in range(3):
        diff[:, :, i] = torch.min(
            torch.stack(
                [
                    diff[:, :, i].abs(),
                    (diff[:, :, i] - a[i, i]).abs(),
                    (diff[:, :, i] + a[i, i]).abs(),
                ]
            ),
            dim=0,
        ).values

    # Calculate Euclidean distances
    distances = torch.sqrt(torch.sum(diff**2, dim=2))  # Shape becomes (n, m)

    return distances


def add_to_indexed_submatrix_(
    a: torch.Tensor, idx: torch.Tensor, idy: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    idx_grid = idx.unsqueeze(1).expand(-1, len(idy))
    idy_grid = idy.unsqueeze(0).expand(len(idx), -1)
    a[idx_grid, idy_grid] += b
    return a


def copy_indexed_submatrix(
    a: torch.Tensor, idx: torch.Tensor, idy: torch.Tensor, out: torch.Tensor = None
) -> torch.Tensor:
    if out is None:
        out = torch.empty((len(idx), len(idy)), dtype=a.dtype, device=a.device)
    out.copy_(a[idx[:, None], idy])
    return out
