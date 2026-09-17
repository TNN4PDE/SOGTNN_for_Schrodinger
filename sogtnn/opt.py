import math
import logging
import torch
from typing import Optional, Callable, Tuple

def _flatten_grads(grads, params):
    views = []
    for g, p in zip(grads, params):
        if g is None:
            views.append(p.new_zeros(p.numel()))
        else:
            views.append(g.contiguous().view(-1))
    return torch.cat(views, dim=0)


def _detach_aux(lam_k, alpha_k, S_mat, H_mat):
    return (
        lam_k.detach() if torch.is_tensor(lam_k) else lam_k,
        alpha_k.detach() if torch.is_tensor(alpha_k) else alpha_k,
        S_mat.detach() if torch.is_tensor(S_mat) else S_mat,
        H_mat.detach() if torch.is_tensor(H_mat) else H_mat,
    )


def _zero_model_grads(models):
    for m in models:
        m.zero_grad(set_to_none=True)


class GPUHessianFreeNewtonCG:
    def __init__(self, models, criterion_fn, K, initial_trust_radius=0.1, max_trust_radius=100.0,
                 eta=0.12, gtol=1e-10, cg_max_iter=256):
        self.models = tuple(models)
        self.criterion_fn = criterion_fn
        self.K = K

        self.params = [p for m in self.models for p in m.parameters()]

        self.trust_radius = float(initial_trust_radius)
        self.max_trust_radius = float(max_trust_radius)
        self.eta = eta
        self.gtol = gtol
        self.cg_max_iter = int(cg_max_iter)

        if not all(p.dtype == torch.float64 for p in self.params):
            logging.warning("Optimizer Check: Parameters are not float64. This may cause numerical instability in HVP.")

    def _gather_flat_grad(self):
        views = []
        for p in self.params:
            if p.grad is None:
                views.append(p.new_zeros(p.numel()))
            else:
                views.append(p.grad.contiguous().view(-1))
        return torch.cat(views, dim=0)

    @torch.enable_grad()
    def _compute_hvp(self, gradient, p, retain_graph=True):
        grad_v_prod = torch.dot(gradient, p)
        hess_vp = torch.autograd.grad(grad_v_prod, self.params, retain_graph=retain_graph)
        flat_hvp = torch.cat([vp.contiguous().view(-1) for vp in hess_vp], dim=0)
        return flat_hvp

    @torch.no_grad()
    def _calc_boundaries(self, iterate, direction):
        a = torch.sum(direction ** 2)
        b = 2 * torch.sum(direction * iterate)
        c = torch.sum(iterate ** 2) - self.trust_radius ** 2
        sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)
        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        return [ta, tb] if ta.item() < tb.item() else [tb, ta]

    def _solve_subproblem_cg(self, loss_val, flat_grad):
        iterate = torch.zeros_like(flat_grad)
        residual = flat_grad.detach()
        direction = -residual

        jac_mag = torch.norm(flat_grad).item()
        tolerance = min(0.5, math.sqrt(jac_mag)) * jac_mag
        cg_iters = 0

        if jac_mag <= tolerance:
            return iterate, False, cg_iters

        for _ in range(self.cg_max_iter):
            cg_iters += 1

            hessian_vec_prod = self._compute_hvp(flat_grad, direction, retain_graph=True)
            hevp_dot_prod = torch.dot(hessian_vec_prod, direction)

            if hevp_dot_prod.item() <= 0:
                ta, tb = self._calc_boundaries(iterate, direction)
                pa = iterate + ta * direction
                pb = iterate + tb * direction

                with torch.no_grad():
                    hvp_pa = self._compute_hvp(flat_grad, pa, retain_graph=True)
                    val_a = loss_val + torch.dot(flat_grad, pa) + 0.5 * torch.dot(hvp_pa, pa)
                    del hvp_pa

                    hvp_pb = self._compute_hvp(flat_grad, pb, retain_graph=True)
                    val_b = loss_val + torch.dot(flat_grad, pb) + 0.5 * torch.dot(hvp_pb, pb)
                    del hvp_pb

                del hessian_vec_prod
                return (pa, True, cg_iters) if val_a.item() < val_b.item() else (pb, True, cg_iters)

            residual_sq_norm = torch.dot(residual, residual)
            cg_step_size = residual_sq_norm / hevp_dot_prod
            next_iterate = iterate + cg_step_size * direction

            if torch.norm(next_iterate).item() >= self.trust_radius:
                ta, tb = self._calc_boundaries(iterate, direction)
                del hessian_vec_prod
                return iterate + tb * direction, True, cg_iters

            next_residual = residual + cg_step_size * hessian_vec_prod
            del hessian_vec_prod

            if torch.norm(next_residual).item() < tolerance:
                return next_iterate, False, cg_iters

            beta = torch.dot(next_residual, next_residual) / residual_sq_norm
            direction = -next_residual + beta * direction
            iterate = next_iterate
            residual = next_residual

        return iterate, False, cg_iters

    @torch.no_grad()
    def _add_flat_step_(self, flat_step, alpha=1.0):
        start_idx = 0
        for param in self.params:
            num_els = param.numel()
            curr_upd = flat_step[start_idx:start_idx + num_els]
            param.add_(curr_upd.view_as(param), alpha=alpha)
            start_idx += num_els

    def step(self):
        _zero_model_grads(self.models)

        loss_tensor, lam_k, alpha_k, S_mat, H_mat = self.criterion_fn(*self.models, self.K)
        start_loss = loss_tensor.item()
        start_aux = _detach_aux(lam_k, alpha_k, S_mat, H_mat)

        grads = torch.autograd.grad(loss_tensor, self.params, create_graph=True)
        flat_grad = _flatten_grads(grads, self.params)
        del loss_tensor, lam_k, alpha_k, S_mat, H_mat

        grad_norm = torch.norm(flat_grad).item()
        if grad_norm <= self.gtol:
            del grads, flat_grad
            return start_loss, *start_aux, 0, self.trust_radius

        param_step, hit_boundary, cg_iters = self._solve_subproblem_cg(start_loss, flat_grad)

        hess_vp = self._compute_hvp(flat_grad, param_step, retain_graph=False).detach()
        with torch.no_grad():
            expected_improvement = -(
                torch.dot(flat_grad, param_step) + 0.5 * torch.dot(hess_vp, param_step)
            ).item()
        del hess_vp, grads, flat_grad

        self._add_flat_step_(param_step, alpha=1.0)

        with torch.no_grad():
            new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat = self.criterion_fn(*self.models, self.K)
            new_loss = new_loss_tensor.item()
            new_aux = _detach_aux(new_lam_k, new_alpha_k, new_S_mat, new_H_mat)
            del new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat

        actual_improvement = start_loss - new_loss
        ratio = actual_improvement / (expected_improvement + 1e-20)

        if ratio < 0.25:
            self.trust_radius *= 0.25
        elif ratio > 0.75 and hit_boundary:
            self.trust_radius = min(2.0 * self.trust_radius, self.max_trust_radius)

        if ratio <= self.eta:
            self._add_flat_step_(param_step, alpha=-1.0)
            del param_step
            return start_loss, *start_aux, cg_iters, self.trust_radius

        del param_step
        return new_loss, *new_aux, cg_iters, self.trust_radius


class GPUHessianFreeNewtonCG_LBFGS:

    def __init__(self, models, criterion_fn, K, alpha_param=None, initial_trust_radius=0.5, max_trust_radius=20.0,
                 eta=0.12, gtol=1e-10, cg_max_iter=256, lbfgs_history_size=24):
        self.models = tuple(models)
        self.criterion_fn = criterion_fn
        self.K = K
        self.alpha_param = alpha_param

        self.params = [p for m in self.models for p in m.parameters()]
        if self.alpha_param is not None:
            self.params.append(self.alpha_param)

        self.trust_radius = float(initial_trust_radius)
        self.max_trust_radius = float(max_trust_radius)
        self.eta = eta
        self.gtol = gtol
        self.cg_max_iter = int(cg_max_iter)

        self.history_size = lbfgs_history_size
        self.s_hist = []
        self.y_hist = []
        self.rho_hist = []
        self.flat_grad_old = None
        self.param_step_old = None

        if not all(p.dtype == torch.float64 for p in self.params):
            logging.warning("Optimizer Check: Parameters are not float64.")

    def _gather_flat_grad(self):
        views = []
        for p in self.params:
            if p.grad is None:
                views.append(p.new_zeros(p.numel()))
            else:
                views.append(p.grad.contiguous().view(-1))
        return torch.cat(views, dim=0)

    @torch.enable_grad()
    def _compute_hvp(self, gradient, p, retain_graph=True):
        grad_v_prod = torch.dot(gradient, p)
        hess_vp = torch.autograd.grad(grad_v_prod, self.params, retain_graph=retain_graph)
        flat_hvp = torch.cat([vp.contiguous().view(-1) for vp in hess_vp], dim=0)
        return flat_hvp

    @torch.no_grad()
    def _calc_boundaries(self, iterate, direction):
        a = torch.sum(direction ** 2)
        b = 2 * torch.sum(direction * iterate)
        c = torch.sum(iterate ** 2) - self.trust_radius ** 2
        sqrt_discriminant = torch.sqrt(torch.clamp(b * b - 4 * a * c, min=0.0))
        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        return [ta, tb] if ta.item() < tb.item() else [tb, ta]

    @torch.no_grad()
    def _apply_lbfgs_preconditioner(self, r):
        if len(self.s_hist) == 0:
            return r.clone()

        q = r.clone()
        alphas = []

        for s, y, rho in zip(reversed(self.s_hist), reversed(self.y_hist), reversed(self.rho_hist)):
            alpha = rho * torch.dot(s, q)
            alphas.append(alpha)
            q.add_(y, alpha=-alpha.item())

        s_latest = self.s_hist[-1]
        y_latest = self.y_hist[-1]
        gamma = torch.dot(s_latest, y_latest) / torch.dot(y_latest, y_latest)
        z = q.mul_(gamma)

        alphas.reverse()
        for s, y, rho, alpha in zip(self.s_hist, self.y_hist, self.rho_hist, alphas):
            beta = rho * torch.dot(y, z)
            z.add_(s, alpha=(alpha - beta).item())

        return z

    def _solve_subproblem_cg(self, loss_val, flat_grad):
        iterate = torch.zeros_like(flat_grad)
        residual = flat_grad.detach()

        jac_mag = torch.norm(residual).item()
        tolerance = min(0.5, math.sqrt(jac_mag)) * jac_mag
        tolerance = max(tolerance, self.gtol)
        cg_iters = 0

        if jac_mag <= tolerance:
            return iterate, False, cg_iters

        z = self._apply_lbfgs_preconditioner(residual)
        direction = -z
        rz_old = torch.dot(residual, z)
        q_model_val = 0.0

        for _ in range(self.cg_max_iter):
            cg_iters += 1

            hessian_vec_prod = self._compute_hvp(flat_grad, direction, retain_graph=True)
            hevp_dot_prod = torch.dot(hessian_vec_prod, direction)

            if hevp_dot_prod.item() <= 0:
                ta, tb = self._calc_boundaries(iterate, direction)
                pa = iterate + ta * direction
                pb = iterate + tb * direction

                with torch.no_grad():
                    hvp_pa = self._compute_hvp(flat_grad, pa, retain_graph=True)
                    val_a = loss_val + torch.dot(flat_grad, pa) + 0.5 * torch.dot(hvp_pa, pa)
                    del hvp_pa

                    hvp_pb = self._compute_hvp(flat_grad, pb, retain_graph=True)
                    val_b = loss_val + torch.dot(flat_grad, pb) + 0.5 * torch.dot(hvp_pb, pb)
                    del hvp_pb

                del hessian_vec_prod
                return (pa, True, cg_iters) if val_a.item() < val_b.item() else (pb, True, cg_iters)

            cg_step_size = rz_old / hevp_dot_prod
            next_iterate = iterate + cg_step_size * direction

            if torch.norm(next_iterate).item() >= self.trust_radius:
                ta, tb = self._calc_boundaries(iterate, direction)
                del hessian_vec_prod
                return iterate + tb * direction, True, cg_iters

            next_residual = residual + cg_step_size * hessian_vec_prod
            del hessian_vec_prod

            if torch.norm(next_residual).item() < tolerance:
                return next_iterate, False, cg_iters

            expected_reduction = 0.5 * cg_step_size.item() * rz_old.item()
            q_model_val += expected_reduction

            if expected_reduction < 1e-3 * q_model_val:
                return next_iterate, False, cg_iters

            next_z = self._apply_lbfgs_preconditioner(next_residual)
            rz_new = torch.dot(next_residual, next_z)

            beta = rz_new / rz_old
            direction = -next_z + beta * direction

            iterate = next_iterate
            residual = next_residual
            z = next_z
            rz_old = rz_new

        return iterate, False, cg_iters

    def _call_criterion(self):
        if self.alpha_param is not None:
            return self.criterion_fn(*self.models, self.alpha_param)
        return self.criterion_fn(*self.models, self.K)

    @torch.no_grad()
    def _add_flat_step_(self, flat_step, alpha=1.0):
        start_idx = 0
        for param in self.params:
            num_els = param.numel()
            curr_upd = flat_step[start_idx:start_idx + num_els]
            param.add_(curr_upd.view_as(param), alpha=alpha)
            start_idx += num_els

    def _zero_all_grads(self):
        _zero_model_grads(self.models)
        if self.alpha_param is not None:
            self.alpha_param.grad = None

    def step(self):
        self._zero_all_grads()

        loss_tensor, lam_k, alpha_k, S_mat, H_mat = self._call_criterion()
        start_loss = loss_tensor.item()
        start_aux = _detach_aux(lam_k, alpha_k, S_mat, H_mat)

        grads = torch.autograd.grad(loss_tensor, self.params, create_graph=True, allow_unused=True)
        flat_grad = _flatten_grads(grads, self.params)
        del loss_tensor, lam_k, alpha_k, S_mat, H_mat

        grad_norm = torch.norm(flat_grad).item()
        if grad_norm <= self.gtol:
            del grads, flat_grad
            return start_loss, *start_aux, 0, self.trust_radius

        if self.flat_grad_old is not None and self.param_step_old is not None:
            y = flat_grad.detach() - self.flat_grad_old
            s = self.param_step_old
            s_dot_y = torch.dot(s, y).item()
            if s_dot_y > 1e-14:
                rho = 1.0 / s_dot_y
                self.s_hist.append(s.detach().clone())
                self.y_hist.append(y.detach().clone())
                self.rho_hist.append(rho)
                if len(self.s_hist) > self.history_size:
                    self.s_hist.pop(0)
                    self.y_hist.pop(0)
                    self.rho_hist.pop(0)
            del y

        self.flat_grad_old = flat_grad.detach().clone()

        param_step, hit_boundary, cg_iters = self._solve_subproblem_cg(start_loss, flat_grad)

        hess_vp = self._compute_hvp(flat_grad, param_step, retain_graph=False).detach()
        with torch.no_grad():
            expected_improvement = -(
                torch.dot(flat_grad, param_step) + 0.5 * torch.dot(hess_vp, param_step)
            ).item()
        del hess_vp, grads, flat_grad

        self._add_flat_step_(param_step, alpha=1.0)

        with torch.no_grad():
            new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat = self._call_criterion()
            new_loss = new_loss_tensor.item()
            new_aux = _detach_aux(new_lam_k, new_alpha_k, new_S_mat, new_H_mat)
            del new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat

        actual_improvement = start_loss - new_loss
        ratio = actual_improvement / (expected_improvement + 1e-20)

        if ratio < 0.25:
            self.trust_radius *= 0.25
        elif ratio > 0.75 and hit_boundary:
            self.trust_radius = min(2.0 * self.trust_radius, self.max_trust_radius)

        if ratio <= self.eta:
            self._add_flat_step_(param_step, alpha=-1.0)
            self.param_step_old = None
            del param_step
            return start_loss, *start_aux, cg_iters, self.trust_radius

        self.param_step_old = param_step.detach().clone()
        del param_step
        return new_loss, *new_aux, cg_iters, self.trust_radius

def _canonical_hvp_mode(mode: str) -> str:
    mode = str(mode).lower()
    aliases = {
        "autograd": "exact",
        "exact_hvp": "exact",
        "fd": "fd_forward",
        "forward": "fd_forward",
        "fd_2point": "fd_forward",
        "2-point": "fd_forward",
        "2point": "fd_forward",
        "central": "fd_central",
        "fd_3point": "fd_central",
        "3-point": "fd_central",
        "3point": "fd_central",
    }
    mode = aliases.get(mode, mode)
    valid = {"exact", "fd_forward", "fd_central"}
    if mode not in valid:
        raise ValueError(f"Unsupported hvp_mode={mode!r}. Use one of {sorted(valid)}.")
    return mode


def _canonical_fd_step_rule(rule: str) -> str:
    rule = str(rule).lower()
    aliases = {
        "fixed": "scipy",
        "absolute": "scipy",
        "abs": "scipy",
        "2point_abs": "scipy",
        "scaled": "norm",
        "relative": "norm",
        "petsc": "petsc_ds",
        "ds": "petsc_ds",
        "dennis_schnabel": "petsc_ds",
        "dennis-schnabel": "petsc_ds",
    }
    rule = aliases.get(rule, rule)
    valid = {"norm", "scipy", "petsc_ds"}
    if rule not in valid:
        raise ValueError(f"Unsupported fd_step_rule={rule!r}. Use one of {sorted(valid)}.")
    return rule


class GPUFDHessianFreeNewtonCG_LBFGS:

    def __init__(
        self,
        models,
        criterion_fn,
        K,
        alpha_param=None,
        initial_trust_radius=0.5,
        max_trust_radius=20.0,
        eta=0.12,
        gtol=1e-10,
        cg_max_iter=256,
        lbfgs_history_size=24,
        hvp_mode="exact",
        fd_step_rule="norm",
        fd_rel_step: Optional[float] = None,
        fd_abs_step: Optional[float] = None,
        fd_min_step: Optional[float] = None,
        fd_max_step: Optional[float] = None,
        fd_umin: float = 1e-7,
        fd_use_abs_dot: bool = False,
    ):
        self.models = tuple(models)
        self.criterion_fn = criterion_fn
        self.K = K
        self.alpha_param = alpha_param

        self.params = [p for m in self.models for p in m.parameters()]
        if self.alpha_param is not None:
            self.params.append(self.alpha_param)

        self.trust_radius = float(initial_trust_radius)
        self.max_trust_radius = float(max_trust_radius)
        self.eta = eta
        self.gtol = gtol
        self.cg_max_iter = int(cg_max_iter)

        self.history_size = lbfgs_history_size
        self.s_hist = []
        self.y_hist = []
        self.rho_hist = []
        self.flat_grad_old = None
        self.param_step_old = None

        self.hvp_mode = _canonical_hvp_mode(hvp_mode)
        self.fd_step_rule = _canonical_fd_step_rule(fd_step_rule)
        self.fd_rel_step = fd_rel_step
        self.fd_abs_step = fd_abs_step
        self.fd_min_step = fd_min_step
        self.fd_max_step = fd_max_step
        self.fd_umin = float(fd_umin)
        self.fd_use_abs_dot = bool(fd_use_abs_dot)

        self.last_fd_step = None
        self.last_hvp_mode = self.hvp_mode
        self.hvp_calls = 0
        self.fd_grad_evals = 0

        if not all(p.dtype == torch.float64 for p in self.params):
            logging.warning(
                "Optimizer Check: Parameters are not float64. "
                "Finite-difference HVP is usually more stable in float64."
            )

    def _gather_flat_grad(self):
        views = []
        for p in self.params:
            if p.grad is None:
                views.append(p.new_zeros(p.numel()))
            else:
                views.append(p.grad.contiguous().view(-1))
        return torch.cat(views, dim=0)

    def _call_criterion(self):
        if self.alpha_param is not None:
            return self.criterion_fn(*self.models, self.alpha_param)
        return self.criterion_fn(*self.models, self.K)

    def _zero_all_grads(self):
        _zero_model_grads(self.models)
        if self.alpha_param is not None:
            self.alpha_param.grad = None

    @torch.no_grad()
    def _add_flat_step_(self, flat_step, alpha=1.0):
        start_idx = 0
        for param in self.params:
            num_els = param.numel()
            curr_upd = flat_step[start_idx:start_idx + num_els]
            param.add_(curr_upd.view_as(param), alpha=alpha)
            start_idx += num_els

    @torch.no_grad()
    def _flat_param_norm(self) -> torch.Tensor:
        acc = None
        for p in self.params:
            val = torch.dot(p.detach().reshape(-1), p.detach().reshape(-1))
            acc = val if acc is None else acc + val
        return torch.sqrt(acc) if acc is not None else torch.tensor(0.0)

    @torch.no_grad()
    def _flat_param_dot(self, v) -> torch.Tensor:
        start_idx = 0
        acc = None
        for p in self.params:
            num_els = p.numel()
            vv = v[start_idx:start_idx + num_els].view_as(p)
            val = torch.sum(p.detach() * vv.detach())
            acc = val if acc is None else acc + val
            start_idx += num_els
        return acc if acc is not None else v.new_tensor(0.0)

    def _default_fd_rel_step(self, mode: Optional[str] = None) -> float:
        mode = self.hvp_mode if mode is None else _canonical_hvp_mode(mode)
        finfo = torch.finfo(self.params[0].dtype)
        eps = float(finfo.eps)
        if mode == "fd_central":
            return eps ** (1.0 / 3.0)
        return math.sqrt(eps)

    def _compute_fd_step(self, direction) -> float:

        direction_detached = direction.detach()
        dtype = direction_detached.dtype
        device = direction_detached.device
        tiny = torch.finfo(dtype).tiny

        rel = self.fd_rel_step
        if rel is None:
            rel = self._default_fd_rel_step(self.hvp_mode)
        rel = float(rel)

        if self.fd_step_rule == "scipy":
            h = float(self.fd_abs_step) if self.fd_abs_step is not None else rel

        elif self.fd_step_rule == "norm":
            v_norm = torch.norm(direction_detached).item()
            theta_norm = self._flat_param_norm().to(device=device, dtype=dtype).item()
            h = rel * max(1.0, theta_norm) / max(v_norm, tiny)

        elif self.fd_step_rule == "petsc_ds":
            v_norm_sq = torch.dot(direction_detached, direction_detached).item()
            if v_norm_sq <= tiny:
                h = rel
            else:
                dot_uv = self._flat_param_dot(direction_detached).item()
                v_l1 = torch.sum(torch.abs(direction_detached)).item()

                signed_dot = dot_uv
                if self.fd_use_abs_dot:
                    signed_dot = abs(dot_uv)
                sign = 1.0 if signed_dot >= 0.0 else -1.0

                if abs(dot_uv) > self.fd_umin * v_l1:
                    h = rel * signed_dot / v_norm_sq
                else:
                    h = rel * self.fd_umin * sign * v_l1 / v_norm_sq
        else:
            raise RuntimeError(f"Unexpected fd_step_rule={self.fd_step_rule!r}")

        if h == 0.0 or not math.isfinite(h):
            h = rel

        if self.fd_min_step is not None:
            min_step = abs(float(self.fd_min_step))
            if abs(h) < min_step:
                h = math.copysign(min_step, h)

        if self.fd_max_step is not None:
            max_step = abs(float(self.fd_max_step))
            if abs(h) > max_step:
                h = math.copysign(max_step, h)

        self.last_fd_step = h
        return h

    @torch.enable_grad()
    def _compute_flat_grad_no_graph(self):
        self._zero_all_grads()
        loss_tensor, lam_k, alpha_k, S_mat, H_mat = self._call_criterion()
        grads = torch.autograd.grad(
            loss_tensor,
            self.params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        flat_grad = _flatten_grads(grads, self.params).detach()
        del loss_tensor, lam_k, alpha_k, S_mat, H_mat, grads
        self.fd_grad_evals += 1
        return flat_grad

    @torch.enable_grad()
    def _compute_hvp_exact(self, gradient, p, retain_graph=True):
        grad_v_prod = torch.dot(gradient, p)
        hess_vp = torch.autograd.grad(grad_v_prod, self.params, retain_graph=retain_graph)
        flat_hvp = torch.cat([vp.contiguous().view(-1) for vp in hess_vp], dim=0)
        return flat_hvp

    @torch.enable_grad()
    def _compute_hvp_fd_forward(self, base_gradient, p):
        h = self._compute_fd_step(p)
        shift = 0.0
        try:
            self._add_flat_step_(p, alpha=h)
            shift += h
            grad_plus = self._compute_flat_grad_no_graph()
            hvp = (grad_plus - base_gradient.detach()) / h
            del grad_plus
            return hvp.detach()
        finally:
            if shift != 0.0:
                self._add_flat_step_(p, alpha=-shift)

    @torch.enable_grad()
    def _compute_hvp_fd_central(self, p):
        h = self._compute_fd_step(p)
        shift = 0.0
        try:
            self._add_flat_step_(p, alpha=h)
            shift += h
            grad_plus = self._compute_flat_grad_no_graph()

            self._add_flat_step_(p, alpha=-2.0 * h)
            shift -= 2.0 * h
            grad_minus = self._compute_flat_grad_no_graph()

            hvp = (grad_plus - grad_minus) / (2.0 * h)
            del grad_plus, grad_minus
            return hvp.detach()
        finally:
            if shift != 0.0:
                self._add_flat_step_(p, alpha=-shift)

    @torch.enable_grad()
    def _compute_hvp(self, gradient, p, retain_graph=True):
        self.hvp_calls += 1
        self.last_hvp_mode = self.hvp_mode

        if self.hvp_mode == "exact":
            return self._compute_hvp_exact(gradient, p, retain_graph=retain_graph)
        if self.hvp_mode == "fd_forward":
            return self._compute_hvp_fd_forward(gradient, p)
        if self.hvp_mode == "fd_central":
            return self._compute_hvp_fd_central(p)
        raise RuntimeError(f"Unexpected hvp_mode={self.hvp_mode!r}")
    
    @torch.no_grad()
    def _calc_boundaries(self, iterate, direction):
        a = torch.sum(direction ** 2)
        b = 2 * torch.sum(direction * iterate)
        c = torch.sum(iterate ** 2) - self.trust_radius ** 2
        sqrt_discriminant = torch.sqrt(torch.clamp(b * b - 4 * a * c, min=0.0))
        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        return [ta, tb] if ta.item() < tb.item() else [tb, ta]

    @torch.no_grad()
    def _apply_lbfgs_preconditioner(self, r):
        if len(self.s_hist) == 0:
            return r.clone()

        q = r.clone()
        alphas = []

        for s, y, rho in zip(reversed(self.s_hist), reversed(self.y_hist), reversed(self.rho_hist)):
            alpha = rho * torch.dot(s, q)
            alphas.append(alpha)
            q.add_(y, alpha=-alpha.item())

        s_latest = self.s_hist[-1]
        y_latest = self.y_hist[-1]
        gamma = torch.dot(s_latest, y_latest) / torch.dot(y_latest, y_latest)
        z = q.mul_(gamma)

        alphas.reverse()
        for s, y, rho, alpha in zip(self.s_hist, self.y_hist, self.rho_hist, alphas):
            beta = rho * torch.dot(y, z)
            z.add_(s, alpha=(alpha - beta).item())

        return z

    def _solve_subproblem_cg(self, loss_val, flat_grad):
        iterate = torch.zeros_like(flat_grad)
        residual = flat_grad.detach()

        jac_mag = torch.norm(residual).item()
        tolerance = min(0.5, math.sqrt(jac_mag)) * jac_mag
        tolerance = max(tolerance, self.gtol)
        cg_iters = 0

        if jac_mag <= tolerance:
            return iterate, False, cg_iters

        z = self._apply_lbfgs_preconditioner(residual)
        direction = -z
        rz_old = torch.dot(residual, z)
        q_model_val = 0.0

        for _ in range(self.cg_max_iter):
            cg_iters += 1

            hessian_vec_prod = self._compute_hvp(flat_grad, direction, retain_graph=True)
            hevp_dot_prod = torch.dot(hessian_vec_prod, direction)

            if hevp_dot_prod.item() <= 0:
                ta, tb = self._calc_boundaries(iterate, direction)
                pa = iterate + ta * direction
                pb = iterate + tb * direction

                with torch.no_grad():
                    hvp_pa = self._compute_hvp(flat_grad, pa, retain_graph=True)
                    val_a = loss_val + torch.dot(flat_grad, pa) + 0.5 * torch.dot(hvp_pa, pa)
                    del hvp_pa

                    hvp_pb = self._compute_hvp(flat_grad, pb, retain_graph=True)
                    val_b = loss_val + torch.dot(flat_grad, pb) + 0.5 * torch.dot(hvp_pb, pb)
                    del hvp_pb

                del hessian_vec_prod
                return (pa, True, cg_iters) if val_a.item() < val_b.item() else (pb, True, cg_iters)

            cg_step_size = rz_old / hevp_dot_prod
            next_iterate = iterate + cg_step_size * direction

            if torch.norm(next_iterate).item() >= self.trust_radius:
                ta, tb = self._calc_boundaries(iterate, direction)
                del hessian_vec_prod
                return iterate + tb * direction, True, cg_iters

            next_residual = residual + cg_step_size * hessian_vec_prod
            del hessian_vec_prod

            if torch.norm(next_residual).item() < tolerance:
                return next_iterate, False, cg_iters

            expected_reduction = 0.5 * cg_step_size.item() * rz_old.item()
            q_model_val += expected_reduction

            if expected_reduction < 1e-3 * q_model_val:
                return next_iterate, False, cg_iters

            next_z = self._apply_lbfgs_preconditioner(next_residual)
            rz_new = torch.dot(next_residual, next_z)

            beta = rz_new / rz_old
            direction = -next_z + beta * direction

            iterate = next_iterate
            residual = next_residual
            z = next_z
            rz_old = rz_new

        return iterate, False, cg_iters


    def step(self):
        self._zero_all_grads()
        self.hvp_calls = 0
        self.fd_grad_evals = 0
        self.last_fd_step = None

        loss_tensor, lam_k, alpha_k, S_mat, H_mat = self._call_criterion()
        start_loss = loss_tensor.item()
        start_aux = _detach_aux(lam_k, alpha_k, S_mat, H_mat)

        create_graph_for_grad = (self.hvp_mode == "exact")
        grads = torch.autograd.grad(
            loss_tensor,
            self.params,
            create_graph=create_graph_for_grad,
            retain_graph=create_graph_for_grad,
            allow_unused=True,
        )
        flat_grad = _flatten_grads(grads, self.params)
        if self.hvp_mode != "exact":
            flat_grad = flat_grad.detach()
        del loss_tensor, lam_k, alpha_k, S_mat, H_mat

        grad_norm = torch.norm(flat_grad).item()
        if grad_norm <= self.gtol:
            del grads, flat_grad
            return start_loss, *start_aux, 0, self.trust_radius

        if self.flat_grad_old is not None and self.param_step_old is not None:
            y = flat_grad.detach() - self.flat_grad_old
            s = self.param_step_old
            s_dot_y = torch.dot(s, y).item()
            if s_dot_y > 1e-14:
                rho = 1.0 / s_dot_y
                self.s_hist.append(s.detach().clone())
                self.y_hist.append(y.detach().clone())
                self.rho_hist.append(rho)
                if len(self.s_hist) > self.history_size:
                    self.s_hist.pop(0)
                    self.y_hist.pop(0)
                    self.rho_hist.pop(0)
            del y

        self.flat_grad_old = flat_grad.detach().clone()

        param_step, hit_boundary, cg_iters = self._solve_subproblem_cg(start_loss, flat_grad)

        hess_vp = self._compute_hvp(flat_grad, param_step, retain_graph=False).detach()
        with torch.no_grad():
            expected_improvement = -(
                torch.dot(flat_grad.detach(), param_step) + 0.5 * torch.dot(hess_vp, param_step)
            ).item()
        del hess_vp, grads, flat_grad

        self._add_flat_step_(param_step, alpha=1.0)

        with torch.no_grad():
            new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat = self._call_criterion()
            new_loss = new_loss_tensor.item()
            new_aux = _detach_aux(new_lam_k, new_alpha_k, new_S_mat, new_H_mat)
            del new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat

        actual_improvement = start_loss - new_loss
        ratio = actual_improvement / (expected_improvement + 1e-20)

        if ratio < 0.25:
            self.trust_radius *= 0.25
        elif ratio > 0.75 and hit_boundary:
            self.trust_radius = min(2.0 * self.trust_radius, self.max_trust_radius)

        if ratio <= self.eta:
            self._add_flat_step_(param_step, alpha=-1.0)
            self.param_step_old = None
            del param_step
            return start_loss, *start_aux, cg_iters, self.trust_radius

        self.param_step_old = param_step.detach().clone()
        del param_step
        return new_loss, *new_aux, cg_iters, self.trust_radius


def _extract_loss(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError("criterion output must be a Tensor or a tuple/list whose first element is the loss Tensor.")


class GPUFDHessianFreeNewtonCG_LBFGS_fast:

    def __init__(
        self,
        models,
        criterion_fn: Callable,
        K,
        alpha_param=None,
        initial_trust_radius=0.5,
        max_trust_radius=20.0,
        eta=0.12,
        gtol=1e-10,
        cg_max_iter=256,
        lbfgs_history_size=24,
        hvp_mode="exact",
        fd_step_rule="norm",
        fd_rel_step: Optional[float] = None,
        fd_abs_step: Optional[float] = None,
        fd_min_step: Optional[float] = None,
        fd_max_step: Optional[float] = None,
        fd_umin: float = 1e-7,
        fd_use_abs_dot: bool = False,
        store_history_on_cpu: bool = False,
        empty_cache_each_step: bool = False,
        loss_only_criterion_fn: Optional[Callable] = None,
        use_cg_model_reduction_for_fd: bool = True,
        cache_fd_param_norm: bool = True,
    ):
        self.models = tuple(models)
        self.criterion_fn = criterion_fn
        self.loss_only_criterion_fn = loss_only_criterion_fn
        self.K = K
        self.alpha_param = alpha_param

        self.params = [p for m in self.models for p in m.parameters()]
        if self.alpha_param is not None:
            self.params.append(self.alpha_param)

        self._param_slices = []
        start = 0
        for p in self.params:
            end = start + p.numel()
            self._param_slices.append((p, start, end))
            start = end
        self.num_params = start

        self.trust_radius = float(initial_trust_radius)
        self.max_trust_radius = float(max_trust_radius)
        self.eta = eta
        self.gtol = gtol
        self.cg_max_iter = int(cg_max_iter)

        self.history_size = lbfgs_history_size
        self.s_hist = []
        self.y_hist = []
        self.rho_hist = []
        self.flat_grad_old = None
        self.param_step_old = None

        self.hvp_mode = _canonical_hvp_mode(hvp_mode)
        self.fd_step_rule = _canonical_fd_step_rule(fd_step_rule)
        self.fd_rel_step = fd_rel_step
        self.fd_abs_step = fd_abs_step
        self.fd_min_step = fd_min_step
        self.fd_max_step = fd_max_step
        self.fd_umin = float(fd_umin)
        self.fd_use_abs_dot = bool(fd_use_abs_dot)
        self.store_history_on_cpu = bool(store_history_on_cpu)
        self.empty_cache_each_step = bool(empty_cache_each_step)
        self.use_cg_model_reduction_for_fd = bool(use_cg_model_reduction_for_fd)
        self.cache_fd_param_norm = bool(cache_fd_param_norm)

        self.last_fd_step = None
        self.last_hvp_mode = self.hvp_mode
        self.hvp_calls = 0
        self.fd_grad_evals = 0
        self._fd_base_param_norm = None

        if not all(p.dtype == torch.float64 for p in self.params):
            logging.warning(
                "Optimizer Check: Parameters are not float64. "
                "Finite-difference HVP is usually more stable in float64."
            )

    def _call_criterion(self):
        if self.alpha_param is not None:
            return self.criterion_fn(*self.models, self.alpha_param)
        return self.criterion_fn(*self.models, self.K)

    def _call_loss_only(self):
        fn = self.loss_only_criterion_fn if self.loss_only_criterion_fn is not None else self.criterion_fn
        if self.alpha_param is not None:
            return _extract_loss(fn(*self.models, self.alpha_param))
        return _extract_loss(fn(*self.models, self.K))

    def _zero_all_grads(self):
        _zero_model_grads(self.models)
        if self.alpha_param is not None:
            self.alpha_param.grad = None

    @torch.no_grad()
    def _add_flat_step_(self, flat_step, alpha=1.0):
        for param, start, end in self._param_slices:
            param.add_(flat_step[start:end].view_as(param), alpha=alpha)

    @torch.no_grad()
    def _flat_param_norm_uncached(self) -> torch.Tensor:
        acc = None
        for p in self.params:
            v = p.detach().reshape(-1)
            val = torch.dot(v, v)
            acc = val if acc is None else acc + val
        return torch.sqrt(acc) if acc is not None else self.params[0].new_tensor(0.0)

    @torch.no_grad()
    def _flat_param_norm(self) -> torch.Tensor:
        if self._fd_base_param_norm is not None:
            return self._fd_base_param_norm
        return self._flat_param_norm_uncached()

    @torch.no_grad()
    def _flat_param_dot(self, v) -> torch.Tensor:
        acc = None
        for p, start, end in self._param_slices:
            vv = v[start:end].view_as(p)
            val = torch.sum(p.detach() * vv.detach())
            acc = val if acc is None else acc + val
        return acc if acc is not None else v.new_tensor(0.0)

    def _default_fd_rel_step(self, mode: Optional[str] = None) -> float:
        mode = self.hvp_mode if mode is None else _canonical_hvp_mode(mode)
        eps = float(torch.finfo(self.params[0].dtype).eps)
        if mode == "fd_central":
            return eps ** (1.0 / 3.0)
        return math.sqrt(eps)

    def _compute_fd_step(self, direction) -> float:
        direction_detached = direction.detach()
        dtype = direction_detached.dtype
        tiny = torch.finfo(dtype).tiny

        rel = self.fd_rel_step
        if rel is None:
            rel = self._default_fd_rel_step(self.hvp_mode)
        rel = float(rel)

        if self.fd_step_rule == "scipy":
            h = float(self.fd_abs_step) if self.fd_abs_step is not None else rel

        elif self.fd_step_rule == "norm":
            v_norm = torch.norm(direction_detached).item()
            theta_norm = self._flat_param_norm().item()
            h = rel * max(1.0, theta_norm) / max(v_norm, tiny)

        elif self.fd_step_rule == "petsc_ds":
            v_norm_sq = torch.dot(direction_detached, direction_detached).item()
            if v_norm_sq <= tiny:
                h = rel
            else:
                dot_uv = self._flat_param_dot(direction_detached).item()
                v_l1 = torch.sum(torch.abs(direction_detached)).item()
                signed_dot = abs(dot_uv) if self.fd_use_abs_dot else dot_uv
                sign = 1.0 if signed_dot >= 0.0 else -1.0
                if abs(dot_uv) > self.fd_umin * v_l1:
                    h = rel * signed_dot / v_norm_sq
                else:
                    h = rel * self.fd_umin * sign * v_l1 / v_norm_sq
        else:
            raise RuntimeError(f"Unexpected fd_step_rule={self.fd_step_rule!r}")

        if h == 0.0 or not math.isfinite(h):
            h = rel

        if self.fd_min_step is not None:
            m = abs(float(self.fd_min_step))
            if abs(h) < m:
                h = math.copysign(m, h)
        if self.fd_max_step is not None:
            m = abs(float(self.fd_max_step))
            if abs(h) > m:
                h = math.copysign(m, h)

        self.last_fd_step = h
        return h

    @torch.enable_grad()
    def _compute_flat_grad_no_graph(self):
        self._zero_all_grads()
        loss_tensor = self._call_loss_only()
        grads = torch.autograd.grad(
            loss_tensor,
            self.params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        flat_grad = _flatten_grads(grads, self.params).detach()
        del loss_tensor, grads
        self.fd_grad_evals += 1
        return flat_grad

    @torch.enable_grad()
    def _compute_hvp_exact(self, gradient, p, retain_graph=True):
        grad_v_prod = torch.dot(gradient, p)
        hess_vp = torch.autograd.grad(grad_v_prod, self.params, retain_graph=retain_graph)
        return torch.cat([vp.contiguous().view(-1) for vp in hess_vp], dim=0)

    @torch.enable_grad()
    def _compute_hvp_fd_forward(self, base_gradient, p):
        h = self._compute_fd_step(p)
        shift = 0.0
        try:
            self._add_flat_step_(p, alpha=h)
            shift += h
            grad_plus = self._compute_flat_grad_no_graph()
            grad_plus.sub_(base_gradient.detach()).div_(h)
            return grad_plus.detach()
        finally:
            if shift != 0.0:
                self._add_flat_step_(p, alpha=-shift)

    @torch.enable_grad()
    def _compute_hvp_fd_central(self, p):
        h = self._compute_fd_step(p)
        shift = 0.0
        try:
            self._add_flat_step_(p, alpha=h)
            shift += h
            grad_plus = self._compute_flat_grad_no_graph()

            self._add_flat_step_(p, alpha=-2.0 * h)
            shift -= 2.0 * h
            grad_minus = self._compute_flat_grad_no_graph()

            grad_plus.sub_(grad_minus).div_(2.0 * h)
            del grad_minus
            return grad_plus.detach()
        finally:
            if shift != 0.0:
                self._add_flat_step_(p, alpha=-shift)

    @torch.enable_grad()
    def _compute_hvp(self, gradient, p, retain_graph=True):
        self.hvp_calls += 1
        self.last_hvp_mode = self.hvp_mode
        if self.hvp_mode == "exact":
            return self._compute_hvp_exact(gradient, p, retain_graph=retain_graph)
        if self.hvp_mode == "fd_forward":
            return self._compute_hvp_fd_forward(gradient, p)
        if self.hvp_mode == "fd_central":
            return self._compute_hvp_fd_central(p)
        raise RuntimeError(f"Unexpected hvp_mode={self.hvp_mode!r}")

    @torch.no_grad()
    def _calc_boundaries(self, iterate, direction):
        a = torch.sum(direction ** 2)
        b = 2 * torch.sum(direction * iterate)
        c = torch.sum(iterate ** 2) - self.trust_radius ** 2
        sqrt_discriminant = torch.sqrt(torch.clamp(b * b - 4 * a * c, min=0.0))
        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        return [ta, tb] if ta.item() < tb.item() else [tb, ta]

    @torch.no_grad()
    def _history_vec_to_device(self, x, ref):
        if x.device == ref.device and x.dtype == ref.dtype:
            return x
        return x.to(device=ref.device, dtype=ref.dtype, non_blocking=True)

    @torch.no_grad()
    def _apply_lbfgs_preconditioner(self, r):
        if len(self.s_hist) == 0:
            return r.clone()

        q = r.clone()
        alphas = []
        for s, y, rho in zip(reversed(self.s_hist), reversed(self.y_hist), reversed(self.rho_hist)):
            s_dev = self._history_vec_to_device(s, q)
            y_dev = self._history_vec_to_device(y, q)
            alpha = rho * torch.dot(s_dev, q)
            alphas.append(alpha)
            q.add_(y_dev, alpha=-alpha.item())
            if self.store_history_on_cpu:
                del s_dev, y_dev

        s_latest = self._history_vec_to_device(self.s_hist[-1], q)
        y_latest = self._history_vec_to_device(self.y_hist[-1], q)
        gamma = torch.dot(s_latest, y_latest) / torch.dot(y_latest, y_latest)
        z = q.mul_(gamma)
        if self.store_history_on_cpu:
            del s_latest, y_latest

        alphas.reverse()
        for s, y, rho, alpha in zip(self.s_hist, self.y_hist, self.rho_hist, alphas):
            s_dev = self._history_vec_to_device(s, z)
            y_dev = self._history_vec_to_device(y, z)
            beta = rho * torch.dot(y_dev, z)
            z.add_(s_dev, alpha=(alpha - beta).item())
            if self.store_history_on_cpu:
                del s_dev, y_dev
        return z

    def _solve_subproblem_cg(self, loss_val, flat_grad):
        track_q = (self.hvp_mode != "exact" and self.use_cg_model_reduction_for_fd)

        iterate = torch.zeros_like(flat_grad)
        residual = flat_grad.detach()
        h_iterate = torch.zeros_like(flat_grad) if track_q else None

        jac_mag = torch.norm(residual).item()
        tolerance = min(0.5, math.sqrt(jac_mag)) * jac_mag
        tolerance = max(tolerance, self.gtol)
        cg_iters = 0

        if jac_mag <= tolerance:
            return iterate, False, cg_iters, 0.0

        z = self._apply_lbfgs_preconditioner(residual)
        direction = -z
        rz_old = torch.dot(residual, z)
        q_model_val = 0.0 

        for _ in range(self.cg_max_iter):
            cg_iters += 1

            hessian_vec_prod = self._compute_hvp(flat_grad, direction, retain_graph=True)
            hevp_dot_prod = torch.dot(hessian_vec_prod, direction)

            if hevp_dot_prod.item() <= 0:
                ta, tb = self._calc_boundaries(iterate, direction)
                pa = iterate + ta * direction
                pb = iterate + tb * direction

                hvp_pa = self._compute_hvp(flat_grad, pa, retain_graph=True)
                val_a = loss_val + torch.dot(flat_grad, pa) + 0.5 * torch.dot(hvp_pa, pa)
                del hvp_pa

                hvp_pb = self._compute_hvp(flat_grad, pb, retain_graph=True)
                val_b = loss_val + torch.dot(flat_grad, pb) + 0.5 * torch.dot(hvp_pb, pb)
                del hvp_pb

                del hessian_vec_prod
                if val_a.item() < val_b.item():
                    pred = max(loss_val - val_a.item(), 0.0)
                    return pa, True, cg_iters, pred
                pred = max(loss_val - val_b.item(), 0.0)
                return pb, True, cg_iters, pred

            cg_step_size = rz_old / hevp_dot_prod
            next_iterate = iterate + cg_step_size * direction

            if torch.norm(next_iterate).item() >= self.trust_radius:
                ta, tb = self._calc_boundaries(iterate, direction)
                final_step = iterate + tb * direction
                pred = None
                if track_q:
                    h_final = h_iterate + tb * hessian_vec_prod
                    q_val = loss_val + torch.dot(flat_grad, final_step) + 0.5 * torch.dot(h_final, final_step)
                    pred = max(loss_val - q_val.item(), 0.0)
                    del h_final
                del hessian_vec_prod
                return final_step, True, cg_iters, pred

            next_residual = residual + cg_step_size * hessian_vec_prod

            expected_reduction = 0.5 * cg_step_size.item() * rz_old.item()
            q_model_next = q_model_val + expected_reduction

            if track_q:
                next_h_iterate = h_iterate + cg_step_size * hessian_vec_prod
            else:
                next_h_iterate = None

            del hessian_vec_prod

            if torch.norm(next_residual).item() < tolerance:
                if track_q:
                    del h_iterate
                return next_iterate, False, cg_iters, max(q_model_next, 0.0)

            if expected_reduction < 1e-3 * q_model_next:
                if track_q:
                    del h_iterate, next_h_iterate
                return next_iterate, False, cg_iters, max(q_model_next, 0.0)

            next_z = self._apply_lbfgs_preconditioner(next_residual)
            rz_new = torch.dot(next_residual, next_z)

            beta = rz_new / rz_old
            direction = -next_z + beta * direction

            iterate = next_iterate
            residual = next_residual
            z = next_z
            rz_old = rz_new
            q_model_val = q_model_next
            if track_q:
                h_iterate = next_h_iterate

        if track_q:
            del h_iterate
        return iterate, False, cg_iters, max(q_model_val, 0.0)

    def step(self):
        self._zero_all_grads()
        self.hvp_calls = 0
        self.fd_grad_evals = 0
        self.last_fd_step = None
        self._fd_base_param_norm = None

        loss_tensor, lam_k, alpha_k, S_mat, H_mat = self._call_criterion()
        start_loss = loss_tensor.item()
        start_aux = _detach_aux(lam_k, alpha_k, S_mat, H_mat)

        create_graph_for_grad = (self.hvp_mode == "exact")
        grads = torch.autograd.grad(
            loss_tensor,
            self.params,
            create_graph=create_graph_for_grad,
            retain_graph=create_graph_for_grad,
            allow_unused=True,
        )
        flat_grad = _flatten_grads(grads, self.params)
        if self.hvp_mode != "exact":
            flat_grad = flat_grad.detach()
            del grads
            grads = None
            if self.cache_fd_param_norm and self.fd_step_rule == "norm":
                self._fd_base_param_norm = self._flat_param_norm_uncached().detach()
        del loss_tensor, lam_k, alpha_k, S_mat, H_mat

        grad_norm = torch.norm(flat_grad).item()
        if grad_norm <= self.gtol:
            del grads, flat_grad
            self._fd_base_param_norm = None
            return start_loss, *start_aux, 0, self.trust_radius

        # L-BFGS history
        if self.flat_grad_old is not None and self.param_step_old is not None:
            old_grad = self._history_vec_to_device(self.flat_grad_old, flat_grad)
            s = self._history_vec_to_device(self.param_step_old, flat_grad)
            y = flat_grad.detach() - old_grad
            s_dot_y = torch.dot(s, y).item()
            if s_dot_y > 1e-14:
                rho = 1.0 / s_dot_y
                s_new = s.detach().clone()
                y_new = y.detach().clone()
                if self.store_history_on_cpu:
                    s_new = s_new.cpu()
                    y_new = y_new.cpu()
                self.s_hist.append(s_new)
                self.y_hist.append(y_new)
                self.rho_hist.append(rho)
                if len(self.s_hist) > self.history_size:
                    self.s_hist.pop(0)
                    self.y_hist.pop(0)
                    self.rho_hist.pop(0)
            del y, old_grad, s

        self.flat_grad_old = flat_grad.detach().clone()
        if self.store_history_on_cpu:
            self.flat_grad_old = self.flat_grad_old.cpu()

        param_step, hit_boundary, cg_iters, predicted_from_cg = self._solve_subproblem_cg(start_loss, flat_grad)

        use_cg_pred = (
            self.hvp_mode != "exact"
            and self.use_cg_model_reduction_for_fd
            and predicted_from_cg is not None
            and math.isfinite(float(predicted_from_cg))
            and float(predicted_from_cg) > 0.0
        )

        if use_cg_pred:
            expected_improvement = float(predicted_from_cg)
            del flat_grad
            if grads is not None:
                del grads
        else:
            hess_vp = self._compute_hvp(flat_grad, param_step, retain_graph=False).detach()
            with torch.no_grad():
                expected_improvement = -(
                    torch.dot(flat_grad.detach(), param_step) + 0.5 * torch.dot(hess_vp, param_step)
                ).item()
            del hess_vp, flat_grad
            if grads is not None:
                del grads

        self._add_flat_step_(param_step, alpha=1.0)

        with torch.no_grad():
            new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat = self._call_criterion()
            new_loss = new_loss_tensor.item()
            new_aux = _detach_aux(new_lam_k, new_alpha_k, new_S_mat, new_H_mat)
            del new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat

        actual_improvement = start_loss - new_loss
        ratio = actual_improvement / (expected_improvement + 1e-20)

        if ratio < 0.25:
            self.trust_radius *= 0.25
        elif ratio > 0.75 and hit_boundary:
            self.trust_radius = min(2.0 * self.trust_radius, self.max_trust_radius)

        if ratio <= self.eta:
            self._add_flat_step_(param_step, alpha=-1.0)
            self.param_step_old = None
            del param_step
            self._fd_base_param_norm = None
            if self.empty_cache_each_step and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return start_loss, *start_aux, cg_iters, self.trust_radius

        self.param_step_old = param_step.detach().clone()
        if self.store_history_on_cpu:
            self.param_step_old = self.param_step_old.cpu()
        del param_step
        self._fd_base_param_norm = None
        if self.empty_cache_each_step and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return new_loss, *new_aux, cg_iters, self.trust_radius
