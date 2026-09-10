import torch
import cupy as cp

# Distinguishes "argument not supplied" from "supplied as None": autograd sizes the backward
# return tuple by the number of arguments actually passed to apply(), so the two differ.
_UNSET = object()


class GlideStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx,t,dt,model,level,H_prev,bed,beta,smb,return_u_s=_UNSET):
        # Trailing optional arg is positional, so the number supplied is what backward must
        # return gradients for.
        ctx.n_inputs = 8 + (1 if return_u_s is not _UNSET else 0)
        return_u_s = False if return_u_s is _UNSET else bool(return_u_s)

        ctx.t = t
        ctx.dt = dt
        ctx.model = model
        ctx.level = level
        ctx.ssa = model.mg.levels[level].ssa
        ctx.diva = model.mg.levels[level].diva

        # return_u_s adds the DIVA SURFACE speed to the outputs.  Observations are of surface
        # velocity, and under DIVA that differs from the depth average by the vertical shear, so
        # an inversion against surface data should compare against this rather than u,v.  It is a
        # cell-centred SPEED; build a vector as u_s*(u,v)/|(u,v)| in the caller if one is needed.
        ctx.return_u_s = return_u_s
        if return_u_s and not ctx.diva:
            raise ValueError(
                "return_u_s requires DIVA (stress_scheme='diva'). Under SSA/MOLHO there is no "
                "vertical shear and the surface speed is |ubar|, which the returned u,v give.")

        model.set_top_level(level)

        model.mg.state.H_prev.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.state.H.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.geometry.bed.set(cp.asarray(bed.data),start_level=level)
        model.mg.sliding.beta.set(cp.asarray(beta.data),start_level=level)
        model.mg.forcing.smb.set(cp.asarray(smb.data),start_level=level)

        model.forward(t,dt,update_geometry=False)

        u_torch = torch.tensor(model.mg[level].state.u.data)
        v_torch = torch.tensor(model.mg[level].state.v.data)
        H_torch = torch.tensor(model.mg[level].state.H.data)
        mask_torch = torch.tensor(model.mg[level].state.mask.data)
        phi_torch = torch.tensor(model.mg[level].state.phi.data)
        xi_torch = torch.tensor(model.mg[level].state.xi.data)

        if ctx.ssa:
            # SSA/DIVA mode: ud/vd are identically zero - return fresh zero
            # tensors so downstream code stays scheme-agnostic, but don't
            # copy or checkpoint the all-zero model state
            ud_torch = torch.zeros_like(u_torch)
            vd_torch = torch.zeros_like(v_torch)
            ctx.save_for_backward(u_torch,v_torch,H_torch,mask_torch,phi_torch,xi_torch,H_prev,bed,beta,smb)
        else:
            ud_torch = torch.tensor(model.mg[level].state.ud.data)
            vd_torch = torch.tensor(model.mg[level].state.vd.data)
            ctx.save_for_backward(u_torch,v_torch,ud_torch,vd_torch,H_torch,mask_torch,phi_torch,xi_torch,H_prev,bed,beta,smb)
        ctx.mark_non_differentiable(mask_torch)

        if return_u_s:
            u_s_torch = torch.tensor(model.mg[level].state.u_s.data)
            return u_torch, v_torch, ud_torch, vd_torch, H_torch, mask_torch, u_s_torch
        return u_torch, v_torch, ud_torch, vd_torch, H_torch, mask_torch

    @staticmethod
    def backward(ctx, *grads):
        if ctx.return_u_s:
            gu, gv, gud, gvd, gH, gM, gu_s = grads
        else:
            gu, gv, gud, gvd, gH, gM = grads; gu_s = None

        t = ctx.t
        dt = ctx.dt
        model = ctx.model
        level = ctx.level

        if ctx.ssa:
            u_torch,v_torch,H_torch,mask_torch,phi_torch,xi_torch,H_prev,bed,beta,smb = ctx.saved_tensors
        else:
            u_torch,v_torch,ud_torch,vd_torch,H_torch,mask_torch,phi_torch,xi_torch,H_prev,bed,beta,smb = ctx.saved_tensors

        model.mg.state.H_prev.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.geometry.bed.set(cp.asarray(bed.data),start_level=level)
        model.mg.sliding.beta.set(cp.asarray(beta.data),start_level=level)
        model.mg.forcing.smb.set(cp.asarray(smb.data),start_level=level)

        model.mg.state.u.set(cp.asarray(u_torch.data),start_level=level)
        model.mg.state.v.set(cp.asarray(v_torch.data),start_level=level)
        if not ctx.ssa:
            # in SSA/DIVA mode the model's ud/vd state is already (exactly) zero
            model.mg.state.ud.set(cp.asarray(ud_torch.data),start_level=level)
            model.mg.state.vd.set(cp.asarray(vd_torch.data),start_level=level)
        model.mg.state.H.set(cp.asarray(H_torch.data),start_level=level)
        model.mg.state.phi.set(cp.asarray(phi_torch.data),start_level=level)
        model.mg.state.xi.set(cp.asarray(xi_torch.data),start_level=level)
        model.mg.state.mask.set(cp.asarray(mask_torch.data),start_level=level)

        # The surface cotangent (DIVA) takes a different route into the adjoint than the state
        # cotangents: it is scattered to the velocity facets through the closure and contributes an
        # explicit parameter term.  model.backward handles both; see notes/diva_adjoint_map.md.
        dJdu_s = None if gu_s is None else cp.asarray(gu_s)

        # autograd passes None for outputs the objective never touched;
        # model.backward treats None as a zero adjoint forcing
        converged = model.backward(t,dt,
                dJdu=cp.asarray(gu) if gu is not None else None,
                dJdv=cp.asarray(gv) if gv is not None else None,
                dJdud=cp.asarray(gud) if gud is not None else None,
                dJdvd=cp.asarray(gvd) if gvd is not None else None,
                dJdH=cp.asarray(gH) if gH is not None else None,
                dJdu_s=dJdu_s)

        g_H_prev = torch.tensor(model.mg[level].state.H_prev.grad)
        g_bed = torch.tensor(model.mg[level].geometry.bed.grad)
        g_beta = torch.tensor(model.mg[level].sliding.beta.grad)
        g_smb = torch.tensor(model.mg[level].forcing.smb.grad)

        # Return-count must match the number of arguments passed to apply().
        out = [None, None, None, None, g_H_prev, g_bed, g_beta, g_smb, None]
        return tuple(out[:ctx.n_inputs])


def glide_step(t,dt,model,level,H_prev,bed,beta,smb,return_u_s=False):
    """Keyword-friendly wrapper over ``GlideStep.apply`` (which takes positional args only).

    With ``return_u_s=True`` (DIVA only) the step additionally returns the cell-centred surface
    speed as a 7th output, and gradients flowing back through it are routed via the closure's
    surface-velocity transpose (model.backward dJdu_s)."""
    return GlideStep.apply(t,dt,model,level,H_prev,bed,beta,smb,return_u_s)
