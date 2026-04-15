import math
import torch
import logging
from torch.nn.utils import parameters_to_vector

class GPUHessianFreeNewtonCG:
    """
    纯 GPU 驱动的 Hessian-Free 截断牛顿共轭梯度优化器 (信赖域法)
    专为高维度的非凸量子/物理积分神经网络损失面设计。
    """
    def __init__(self, models, criterion_fn, K, initial_trust_radius=0.1, max_trust_radius=100.0, 
                 eta=0.12, gtol=1e-10, cg_max_iter=256):
        self.models = tuple(models)
        self.criterion_fn = criterion_fn
        self.K = K
        
        self.params = [p for m in self.models for p in m.parameters()]
        
        # 信赖域超参数
        self.trust_radius = float(initial_trust_radius)
        self.max_trust_radius = float(max_trust_radius)
        self.eta = eta  # 接受更新的最小比率阈值
        self.gtol = gtol
        self.cg_max_iter = int(cg_max_iter)

        # 严谨性 Check
        if not all(p.dtype == torch.float64 for p in self.params):
            logging.warning("Optimizer Check: Parameters are not float64. This may cause numerical instability in HVP.")

    def _gather_flat_grad(self):
        """【优化 1】安全展平梯度，强制内存连续"""
        views = []
        for p in self.params:
            if p.grad is None:
                views.append(p.data.new(p.data.numel()).zero_())
            else:
                views.append(p.grad.contiguous().view(-1))
        return torch.cat(views, 0)

    @torch.enable_grad()
    def _compute_hvp(self, gradient, p):
        """【优化 2 & 1】极限性能的海森向量积 (HVP) 计算"""
        # 使用 torch.dot 将内存开销从 O(N) 降至 O(1)
        grad_v_prod = torch.dot(gradient, p)
        # retain_graph=True 允许在此次大循环内多次计算 HVP
        hess_vp = torch.autograd.grad(grad_v_prod, self.params, retain_graph=True)
        return torch.cat([vp.contiguous().view(-1) for vp in hess_vp], dim=-1)

    @torch.no_grad()
    def _calc_boundaries(self, iterate, direction):
        """计算当前搜索方向与信赖域边界的交点距离"""
        a = torch.sum(direction ** 2)
        b = 2 * torch.sum(direction * iterate)
        c = torch.sum(iterate ** 2) - self.trust_radius ** 2
        sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)
        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        return [ta, tb] if ta.item() < tb.item() else [tb, ta]

    def _solve_subproblem_cg(self, loss_val, flat_grad):
        """Steihaug-Toint CG 核心求解回路"""
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
            hessian_vec_prod = self._compute_hvp(flat_grad, direction)
            hevp_dot_prod = torch.dot(hessian_vec_prod, direction)

            # 遇到负曲率，直接走到边界
            if hevp_dot_prod.item() <= 0:
                ta, tb = self._calc_boundaries(iterate, direction)
                pa = iterate + ta * direction
                pb = iterate + tb * direction
                
                val_a = loss_val + torch.dot(flat_grad, pa) + 0.5 * torch.dot(self._compute_hvp(flat_grad, pa), pa)
                val_b = loss_val + torch.dot(flat_grad, pb) + 0.5 * torch.dot(self._compute_hvp(flat_grad, pb), pb)
                
                return (pa, True, cg_iters) if val_a.item() < val_b.item() else (pb, True, cg_iters)

            residual_sq_norm = torch.dot(residual, residual)
            cg_step_size = residual_sq_norm / hevp_dot_prod
            next_iterate = iterate + cg_step_size * direction

            # 如果越出信赖域，截断至边界
            if torch.norm(next_iterate).item() >= self.trust_radius:
                ta, tb = self._calc_boundaries(iterate, direction)
                return iterate + tb * direction, True, cg_iters

            next_residual = residual + cg_step_size * hessian_vec_prod
            if torch.norm(next_residual).item() < tolerance:
                return next_iterate, False, cg_iters

            beta = torch.dot(next_residual, next_residual) / residual_sq_norm
            direction = -next_residual + beta * direction
            iterate = next_iterate
            residual = next_residual

        return iterate, False, cg_iters

    def step(self):
        """
        向外暴露的标准步进接口，接管所有的前向、反向与信赖域评估。
        """
        # 【优化 3】强制清空显存指针，切断上一轮的循环引用
        for m in self.models: m.zero_grad(set_to_none=True)

        # 1. 评估当前状态
        loss_tensor, lam_k, alpha_k, S_mat, H_mat = self.criterion_fn(*self.models, self.K)
        start_loss = loss_tensor.item()

        # 2. 精确计算一阶梯度，必须建立图以便后续 HVP
        loss_tensor.backward(create_graph=True)
        flat_grad = self._gather_flat_grad()

        # 极小梯度直接退出
        if torch.norm(flat_grad).item() <= self.gtol:
            return start_loss, lam_k, alpha_k, S_mat, H_mat, 0, self.trust_radius

        # 3. 求解信赖域子问题 (CG)
        param_step, hit_boundary, cg_iters = self._solve_subproblem_cg(start_loss, flat_grad)

        # 4. 在更新参数前，计算二次模型的预测下降量 (q_model)
        with torch.no_grad():
            # 这里必须再算一次 HVP，因为参数马上要改变了
            hess_vp = self._compute_hvp(flat_grad, param_step).detach()
            expected_improvement = - (torch.dot(flat_grad, param_step) + 0.5 * torch.dot(hess_vp, param_step)).item()

        # 5. 原地更新参数以试探新的 Loss
        with torch.no_grad():
            start_idx = 0
            for param in self.params:
                num_els = param.numel()
                curr_upd = param_step[start_idx:start_idx + num_els]
                param.data.add_(curr_upd.view_as(param))
                start_idx += num_els

            # 评估新状态
            new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat = self.criterion_fn(*self.models, self.K)
            new_loss = new_loss_tensor.item()

        # 6. 信赖域控制器 (Trust-Region Levenberg-Marquardt Logic)
        actual_improvement = start_loss - new_loss
        ratio = actual_improvement / (expected_improvement + 1e-20)

        # 更新信赖域半径
        if ratio < 0.25:
            self.trust_radius *= 0.25
        elif ratio > 0.75 and hit_boundary:
            self.trust_radius = min(2.0 * self.trust_radius, self.max_trust_radius)

        # 7. 步进判定
        if ratio <= self.eta:
            # 拒绝更新，回滚参数
            with torch.no_grad():
                start_idx = 0
                for param in self.params:
                    num_els = param.numel()
                    curr_upd = param_step[start_idx:start_idx + num_els]
                    param.data.add_(-curr_upd.view_as(param)) # 减回去
                    start_idx += num_els
            
            # 返回原始状态，但带有新的信赖域半径
            return start_loss, lam_k, alpha_k, S_mat, H_mat, cg_iters, self.trust_radius
        else:
            # 接受更新，返回新状态
            return new_loss, new_lam_k, new_alpha_k, new_S_mat, new_H_mat, cg_iters, self.trust_radius
        
class GPUHessianFreeNewtonCG_LBFGS:
    """
    基线对比版 (Baseline)：全量 L-BFGS 预条件 Hessian-Free 优化器。
    - 不使用主副步交替，每一步都进行全量 CG 迭代和 HVP 计算。
    - 保留了最初的绝对容差判定和 q_model 早停条件。
    - 已应用极致的底层显存隔离与原地运算优化。
    """
    def __init__(self, models, criterion_fn, K, initial_trust_radius=0.5, max_trust_radius=20.0, 
                 eta=0.12, gtol=1e-10, cg_max_iter=256, lbfgs_history_size=24):
        self.models = tuple(models)
        self.criterion_fn = criterion_fn
        self.K = K
        
        self.params = [p for m in self.models for p in m.parameters()]
        
        # 信赖域与 CG 参数
        self.trust_radius = float(initial_trust_radius)
        self.max_trust_radius = float(max_trust_radius)
        self.eta = eta  
        self.gtol = gtol
        self.cg_max_iter = int(cg_max_iter)

        # L-BFGS 预条件子内存
        self.history_size = lbfgs_history_size
        self.s_hist = []   
        self.y_hist = []   
        self.rho_hist = [] 
        self.flat_grad_old = None
        self.param_step_old = None

        if not all(p.dtype == torch.float64 for p in self.params):
            logging.warning("Optimizer Check: Parameters are not float64.")

    def _gather_flat_grad(self):
        """安全展平梯度，强制内存连续"""
        views = []
        for p in self.params:
            if p.grad is None:
                views.append(p.data.new(p.data.numel()).zero_())
            else:
                views.append(p.grad.contiguous().view(-1))
        return torch.cat(views, 0)

    @torch.enable_grad()
    def _compute_hvp(self, gradient, p):
        """极限性能的海森向量积 (HVP) 计算"""
        grad_v_prod = torch.dot(gradient, p)
        hess_vp = torch.autograd.grad(grad_v_prod, self.params, retain_graph=True)
        return torch.cat([vp.contiguous().view(-1) for vp in hess_vp], dim=-1)

    @torch.no_grad()
    def _calc_boundaries(self, iterate, direction):
        """计算当前搜索方向与信赖域边界的交点距离"""
        a = torch.sum(direction ** 2)
        b = 2 * torch.sum(direction * iterate)
        c = torch.sum(iterate ** 2) - self.trust_radius ** 2
        sqrt_discriminant = torch.sqrt(max(b * b - 4 * a * c, 0.0))
        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        return [ta, tb] if ta.item() < tb.item() else [tb, ta]

    @torch.no_grad()
    def _apply_lbfgs_preconditioner(self, r):
        """计算 z = P^{-1} r (使用双循环递归)"""
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
        """预条件 Steihaug-Toint CG 核心求解回路"""
        iterate = torch.zeros_like(flat_grad)
        residual = flat_grad.detach()
        
        jac_mag = torch.norm(residual).item()
        
        # 恢复最初代码的容差判定逻辑
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
            hessian_vec_prod = self._compute_hvp(flat_grad, direction)
            hevp_dot_prod = torch.dot(hessian_vec_prod, direction)

            if hevp_dot_prod.item() <= 0:
                ta, tb = self._calc_boundaries(iterate, direction)
                pa = iterate + ta * direction
                pb = iterate + tb * direction
                val_a = loss_val + torch.dot(flat_grad, pa) + 0.5 * torch.dot(self._compute_hvp(flat_grad, pa), pa)
                val_b = loss_val + torch.dot(flat_grad, pb) + 0.5 * torch.dot(self._compute_hvp(flat_grad, pb), pb)
                return (pa, True, cg_iters) if val_a.item() < val_b.item() else (pb, True, cg_iters)

            cg_step_size = rz_old / hevp_dot_prod
            next_iterate = iterate + cg_step_size * direction

            if torch.norm(next_iterate).item() >= self.trust_radius:
                ta, tb = self._calc_boundaries(iterate, direction)
                return iterate + tb * direction, True, cg_iters

            next_residual = residual + cg_step_size * hessian_vec_prod
            
            if torch.norm(next_residual).item() < tolerance:
                return next_iterate, False, cg_iters

            expected_reduction = 0.5 * cg_step_size.item() * rz_old.item()
            q_model_val += expected_reduction

            # 恢复最初代码的早停逻辑
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
        """全量牛顿步进：无交替，强制每步计算二阶图"""
        # 优化点：切断循环引用
        for m in self.models: m.zero_grad(set_to_none=True)

        loss_tensor, lam_k, alpha_k, S_mat, H_mat = self.criterion_fn(*self.models, self.K)
        start_loss = loss_tensor.item()

        # 重点：Baseline 版本每一步都必须 create_graph=True，占用全量显存
        loss_tensor.backward(create_graph=True)
        flat_grad = self._gather_flat_grad()

        if torch.norm(flat_grad).item() <= self.gtol:
            return start_loss, lam_k, alpha_k, S_mat, H_mat, 0, self.trust_radius, "Converged"

        # 更新 L-BFGS 历史
        if self.flat_grad_old is not None and self.param_step_old is not None:
            y = flat_grad.detach() - self.flat_grad_old
            s = self.param_step_old
            s_dot_y = torch.dot(s, y).item()
            if s_dot_y > 1e-14:
                rho = 1.0 / s_dot_y
                # 优化点：使用 clone() 防止历史列表挂载多余的计算图
                self.s_hist.append(s.clone())
                self.y_hist.append(y.clone())
                self.rho_hist.append(rho)
                if len(self.s_hist) > self.history_size:
                    self.s_hist.pop(0)
                    self.y_hist.pop(0)
                    self.rho_hist.pop(0)

        self.flat_grad_old = flat_grad.detach().clone()

        # 执行全量 CG 求解
        param_step, hit_boundary, cg_iters = self._solve_subproblem_cg(start_loss, flat_grad)

        with torch.no_grad():
            hess_vp = self._compute_hvp(flat_grad, param_step).detach()
            expected_improvement = - (torch.dot(flat_grad, param_step) + 0.5 * torch.dot(hess_vp, param_step)).item()

            start_idx = 0
            for param in self.params:
                num_els = param.numel()
                curr_upd = param_step[start_idx:start_idx + num_els]
                param.data.add_(curr_upd.view_as(param))
                start_idx += num_els

            new_loss_tensor, new_lam_k, new_alpha_k, new_S_mat, new_H_mat = self.criterion_fn(*self.models, self.K)
            new_loss = new_loss_tensor.item()

        actual_improvement = start_loss - new_loss
        ratio = actual_improvement / (expected_improvement + 1e-20)

        # 信赖域半径更新
        if ratio < 0.25:
            self.trust_radius *= 0.25
        elif ratio > 0.75 and hit_boundary:
            self.trust_radius = min(2.0 * self.trust_radius, self.max_trust_radius)

        if ratio <= self.eta:
            # 拒绝更新，回滚
            with torch.no_grad():
                start_idx = 0
                for param in self.params:
                    num_els = param.numel()
                    curr_upd = param_step[start_idx:start_idx + num_els]
                    param.data.add_(-curr_upd.view_as(param))
                    start_idx += num_els
            
            self.param_step_old = None
            return start_loss, lam_k, alpha_k, S_mat, H_mat, cg_iters, self.trust_radius
        else:
            # 接受更新
            self.param_step_old = param_step.detach().clone()
            return new_loss, new_lam_k, new_alpha_k, new_S_mat, new_H_mat, cg_iters, self.trust_radius