import warnings

import torch
import cupy as cp

# Distinguishes "argument not supplied" from "argument supplied as None".  autograd sizes
# the backward return tuple by the number of arguments actually passed to apply(), so the
# two cases are not interchangeable -- passing m=None explicitly used to make the returned
# gradient count disagree with the input count.
_UNSET = object()


class GlideStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx,t,dt,model,level,H_prev,bed,beta,smb,
                m=_UNSET,u_c=_UNSET,return_u_s=_UNSET):
        # Trailing optional arguments are positional, so the number supplied is what
        # backward must return gradients for.
        ctx.n_inputs = 8 + sum(a is not _UNSET for a in (m, u_c, return_u_s))
        m = None if m is _UNSET else m
        u_c = None if u_c is _UNSET else u_c
        return_u_s = False if return_u_s is _UNSET else bool(return_u_s)

        ctx.t = t
        ctx.dt = dt
        ctx.model = model
        ctx.level = level

        # Optional differentiable sliding-law parameters: the Weertman exponent m
        # (scalar) and the regularized-Coulomb rate-transition u_c (field). Omitted
        # -> the model's existing values are used and no grad is returned for them
        # (byte-for-byte the previous behavior). Passed as requires_grad tensors ->
        # backward returns dJ/dm and/or dJ/du_c via the exact adjoint. (The law
        # itself is selected by model.mg.sliding.sliding_law, set separately.)
        ctx.m_provided = m is not None
        ctx.m_needs_grad = isinstance(m, torch.Tensor) and m.requires_grad
        if ctx.m_provided:
            ctx.m_value = float(m)
            ctx.m_device = m.device if isinstance(m, torch.Tensor) else "cpu"
            ctx.m_dtype = m.dtype if isinstance(m, torch.Tensor) else torch.float32
        ctx.u_c_provided = u_c is not None
        ctx.u_c_needs_grad = isinstance(u_c, torch.Tensor) and u_c.requires_grad

        # return_u_s adds the DIVA SURFACE speed to the outputs.  Observations are of
        # surface velocity, and under DIVA that differs from the depth average by the
        # vertical shear -- the whole point of the scheme -- so an inversion against
        # surface data should compare against this rather than against u,v.  It is a
        # cell-centred SPEED; its direction is inherited from the depth-averaged flow,
        # so build a vector as u_s * (u,v)/|(u,v)| in the caller if one is needed.
        ctx.return_u_s = return_u_s
        if return_u_s and float(model.mg[level].rheology.stress_balance.value) < 0.5:
            raise ValueError(
                "return_u_s requires DIVA (mg.rheology.stress_balance = 1.0). Under SSA "
                "there is no vertical shear and the surface speed is just |ubar|, which "
                "the returned u,v already give.")

        model.set_top_level(level)

        model.mg.state.H_prev.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.state.H.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.geometry.bed.set(cp.asarray(bed.data),start_level=level)
        model.mg.sliding.beta.set(cp.asarray(beta.data),start_level=level)
        model.mg.forcing.smb.set(cp.asarray(smb.data),start_level=level)
        if ctx.m_provided:
            model.mg.sliding.m.set(float(m))
        if ctx.u_c_provided:
            model.mg.sliding.u_c.set(cp.asarray(u_c.data),start_level=level)

        model.forward(t,dt,update_geometry=False)

        u_torch = torch.tensor(model.mg[level].state.u.data)
        v_torch = torch.tensor(model.mg[level].state.v.data)
        H_torch = torch.tensor(model.mg[level].state.H.data)
        mask_torch = torch.tensor(model.mg[level].state.mask.data)
        phi_torch = torch.tensor(model.mg[level].state.phi.data)

        saved = [u_torch,v_torch,H_torch,mask_torch,phi_torch,H_prev,bed,beta,smb]
        if ctx.u_c_provided and isinstance(u_c, torch.Tensor):
            saved.append(u_c)
        ctx.save_for_backward(*saved)
        ctx.mark_non_differentiable(mask_torch)

        if return_u_s:
            u_s_torch = torch.tensor(model.mg[level].state.u_s.data)
            return u_torch, v_torch, H_torch, mask_torch, u_s_torch

        return u_torch, v_torch, H_torch, mask_torch

    @staticmethod
    def backward(ctx, *grads):
        if ctx.return_u_s:
            gu, gv, gH, gM, gu_s = grads
        else:
            (gu, gv, gH, gM), gu_s = grads, None

        t = ctx.t
        dt = ctx.dt
        model = ctx.model
        level = ctx.level
        saved = ctx.saved_tensors
        u_torch,v_torch,H_torch,mask_torch,phi_torch,H_prev,bed,beta,smb = saved[:9]
        u_c_saved = saved[9] if len(saved) > 9 else None

        model.mg.state.H_prev.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.geometry.bed.set(cp.asarray(bed.data),start_level=level)
        model.mg.sliding.beta.set(cp.asarray(beta.data),start_level=level)
        model.mg.forcing.smb.set(cp.asarray(smb.data),start_level=level)

        model.mg.state.u.set(cp.asarray(u_torch.data),start_level=level)
        model.mg.state.v.set(cp.asarray(v_torch.data),start_level=level)
        model.mg.state.H.set(cp.asarray(H_torch.data),start_level=level)
        model.mg.state.phi.set(cp.asarray(phi_torch.data),start_level=level)
        model.mg.state.mask.set(cp.asarray(mask_torch.data),start_level=level)
        if ctx.m_provided:
            model.mg.sliding.m.set(ctx.m_value)
        if u_c_saved is not None:
            model.mg.sliding.u_c.set(cp.asarray(u_c_saved.data),start_level=level)

        # The surface cotangent takes a different route into the adjoint than the state
        # cotangents: it is scattered to the velocity facets through the closure, and it
        # contributes an EXPLICIT parameter term that a depth-averaged objective has no
        # analogue for.  model.backward handles both; see notes/diva_adjoint_map.md.
        dJdu_s = None if gu_s is None else cp.asarray(gu_s)

        converged = model.backward(t,dt,dJdu=cp.asarray(gu),dJdv=cp.asarray(gv),
                                   dJdH=cp.asarray(gH),dJdu_s=dJdu_s,
                                   compute_m_grad=ctx.m_needs_grad,
                                   compute_u_c_grad=ctx.u_c_needs_grad)


        g_H_prev = torch.tensor(model.mg[level].state.H_prev.grad)
        g_bed = torch.tensor(model.mg[level].geometry.bed.grad)
        g_beta = torch.tensor(model.mg[level].sliding.beta.grad)
        g_smb = torch.tensor(model.mg[level].forcing.smb.grad)

        # A non-converged adjoint gives a wrong gradient, so it is suppressed rather than
        # returned -- but SILENTLY suppressing it is worse than either.  Downstream this
        # looks like a stalled optimizer, and if the objective carries a regularization
        # term (which contributes its gradient through torch, not through here) the total
        # gradient is still nonzero, so a caller checking `grad != 0` sees nothing wrong
        # while the optimizer quietly minimizes the regularizer alone.
        #
        # The usual cause is a tolerance below the float32 residual floor, which is grid
        # dependent -- about 7e-7 relative at 64^2 and 1.3e-6 at 128^2.  The solver then
        # spends every V-cycle and reports failure however well it actually did.
        model.last_backward_converged = bool(converged)
        if not converged:
            warnings.warn(
                "GLIDE adjoint solve did not converge; all gradients from this step are "
                "being zeroed. Check that adjoint_solver.fas_options tolerances are above "
                "the float32 residual floor (~1e-6 relative) and that maximum_vcycles is "
                "not the binding constraint. model.last_backward_converged is now False.",
                RuntimeWarning, stacklevel=2)
            g_H_prev[:,:] = 0.0
            g_bed[:,:] = 0.0
            g_beta[:,:] = 0.0
            g_smb[:,:] = 0.0

        g_m = None
        if ctx.m_needs_grad:
            val = float(model.mg[level].sliding.m.grad) if converged else 0.0
            g_m = torch.as_tensor(val, dtype=ctx.m_dtype, device=ctx.m_device)
        g_u_c = None
        if ctx.u_c_needs_grad:
            g_u_c = torch.tensor(model.mg[level].sliding.u_c.grad)
            if not converged:
                g_u_c[:,:] = 0.0

        # Return-count must match the number of arguments passed to apply().
        out = [None, None, None, None, g_H_prev, g_bed, g_beta, g_smb, g_m, g_u_c, None]
        return tuple(out[:ctx.n_inputs])


def glide_step(t,dt,model,level,H_prev,bed,beta,smb,m=None,u_c=None,return_u_s=False):
    """Keyword-friendly wrapper over ``GlideStep.apply``.

    ``torch.autograd.Function.apply`` takes positional arguments only, which makes the
    optional trailing arguments awkward to reach.  Behaviour is otherwise identical.

    With ``return_u_s=True`` (DIVA only) the step additionally returns the cell-centred
    surface speed, and gradients flowing back through it are routed via the closure's
    surface-velocity transpose.
    """
    return GlideStep.apply(t,dt,model,level,H_prev,bed,beta,smb,m,u_c,return_u_s)
