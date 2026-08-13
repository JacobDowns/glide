"""Pieces shared by the twin and Greenland surface-velocity inversions."""
import torch


def surface_velocity(u, v, u_s):
    """The model counterpart of an observed surface velocity vector.

    The DIVA closure returns a surface SPEED (u_s = U_b + tau_b*I1); its direction is
    inherited from the depth-averaged flow.  So the surface velocity vector is the
    depth-averaged direction scaled to the surface speed:

        u_surf = u_s * (u_c, v_c)/|(u_c, v_c)|

    Composing it here rather than in the kernel means autograd carries the derivative
    through both factors -- the closure path for u_s, and the ordinary momentum path
    for the direction.

    u, v are on facets; u_s is cell-centred; the result is cell-centred.
    """
    u_c = 0.5 * (u[:, 1:] + u[:, :-1])
    v_c = 0.5 * (v[1:] + v[:-1])
    spd = torch.sqrt(u_c ** 2 + v_c ** 2 + 1e-12)
    return u_s * u_c / spd, u_s * v_c / spd


def depth_averaged_velocity(u, v):
    """The SSA-style counterpart: the depth average itself, cell-centred.

    This is what GLIDE inversions have compared against surface observations up to
    now.  It is exact for SSA and biased under DIVA -- the point of the comparison
    run in the twin experiment.
    """
    return 0.5 * (u[:, 1:] + u[:, :-1]), 0.5 * (v[1:] + v[:-1])


def elastic_net(log_beta, dx, w_l2=1e-5, w_tv=1e-8, eps=1e-6):
    """Tikhonov + total-variation penalty on log(beta), as in the Greenland example.

    Returned in the same units as the data term so the two can simply be added.
    """
    gy = torch.diff(log_beta, dim=0) / dx
    gx = torch.diff(log_beta, dim=1) / dx
    J_l2 = w_l2 * ((gy ** 2).sum() + (gx ** 2).sum()) * dx ** 2
    gy_, gx_ = gy[:, :-1], gx[:-1]
    J_tv = w_tv * torch.sqrt(gy_ ** 2 + gx_ ** 2 + eps ** 2).sum() * dx ** 2
    return J_l2, J_tv
