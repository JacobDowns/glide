import torch
import cupy as cp

class GlideStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx,t,dt,model,level,H_prev,bed,beta,smb,m=None,u_c=None):
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

        return u_torch, v_torch, H_torch, mask_torch

    @staticmethod
    def backward(ctx, gu, gv, gH, gM):
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

        converged = model.backward(t,dt,dJdu=cp.asarray(gu),dJdv=cp.asarray(gv),dJdH=cp.asarray(gH),
                                   compute_m_grad=ctx.m_needs_grad,
                                   compute_u_c_grad=ctx.u_c_needs_grad)


        g_H_prev = torch.tensor(model.mg[level].state.H_prev.grad)
        g_bed = torch.tensor(model.mg[level].geometry.bed.grad)
        g_beta = torch.tensor(model.mg[level].sliding.beta.grad)
        g_smb = torch.tensor(model.mg[level].forcing.smb.grad)

        if not converged:
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

        # Return-count must match the number of inputs passed to apply():
        # 10 if u_c was supplied, else 9 if m was, else the original 8.
        if ctx.u_c_provided:
            return None, None, None, None, g_H_prev, g_bed, g_beta, g_smb, g_m, g_u_c
        if ctx.m_provided:
            return None, None, None, None, g_H_prev, g_bed, g_beta, g_smb, g_m

        return None, None, None, None, g_H_prev, g_bed, g_beta, g_smb

