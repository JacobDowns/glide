import torch
import cupy as cp

class GlideStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx,t,dt,model,level,H_prev,bed,beta,smb,m=None):
        ctx.t = t
        ctx.dt = dt
        ctx.model = model
        ctx.level = level

        # Optional differentiable Weertman exponent m (a scalar). If omitted, the
        # model's existing m Constant is used and backward returns no grad for it
        # (byte-for-byte the previous behavior). If a tensor with requires_grad is
        # passed, backward returns dJ/dm via the exact adjoint.
        ctx.m_provided = m is not None
        ctx.m_needs_grad = isinstance(m, torch.Tensor) and m.requires_grad
        if ctx.m_provided:
            ctx.m_value = float(m)
            ctx.m_device = m.device if isinstance(m, torch.Tensor) else "cpu"
            ctx.m_dtype = m.dtype if isinstance(m, torch.Tensor) else torch.float32

        model.set_top_level(level)

        model.mg.state.H_prev.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.state.H.set(cp.asarray(H_prev.data),start_level=level)
        model.mg.geometry.bed.set(cp.asarray(bed.data),start_level=level)
        model.mg.sliding.beta.set(cp.asarray(beta.data),start_level=level)
        model.mg.forcing.smb.set(cp.asarray(smb.data),start_level=level)
        if ctx.m_provided:
            model.mg.sliding.m.set(float(m))

        model.forward(t,dt,update_geometry=False)

        u_torch = torch.tensor(model.mg[level].state.u.data)
        v_torch = torch.tensor(model.mg[level].state.v.data)
        H_torch = torch.tensor(model.mg[level].state.H.data)
        mask_torch = torch.tensor(model.mg[level].state.mask.data)
        phi_torch = torch.tensor(model.mg[level].state.phi.data)

        ctx.save_for_backward(u_torch,v_torch,H_torch,mask_torch,phi_torch,H_prev,bed,beta,smb)
        ctx.mark_non_differentiable(mask_torch)
  
        return u_torch, v_torch, H_torch, mask_torch

    @staticmethod
    def backward(ctx, gu, gv, gH, gM):
        t = ctx.t
        dt = ctx.dt
        model = ctx.model
        level = ctx.level
        u_torch,v_torch,H_torch,mask_torch,phi_torch,H_prev,bed,beta,smb = ctx.saved_tensors

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

        converged = model.backward(t,dt,dJdu=cp.asarray(gu),dJdv=cp.asarray(gv),dJdH=cp.asarray(gH),
                                   compute_m_grad=ctx.m_needs_grad)


        g_H_prev = torch.tensor(model.mg[level].state.H_prev.grad)
        g_bed = torch.tensor(model.mg[level].geometry.bed.grad)
        g_beta = torch.tensor(model.mg[level].sliding.beta.grad)
        g_smb = torch.tensor(model.mg[level].forcing.smb.grad)

        if not converged:
            g_H_prev[:,:] = 0.0
            g_bed[:,:] = 0.0
            g_beta[:,:] = 0.0
            g_smb[:,:] = 0.0

        if ctx.m_provided:
            g_m = None
            if ctx.m_needs_grad:
                val = float(model.mg[level].sliding.m.grad) if converged else 0.0
                g_m = torch.as_tensor(val, dtype=ctx.m_dtype, device=ctx.m_device)
            return None, None, None, None, g_H_prev, g_bed, g_beta, g_smb, g_m

        return None, None, None, None, g_H_prev, g_bed, g_beta, g_smb

