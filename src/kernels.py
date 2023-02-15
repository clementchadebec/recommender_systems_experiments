import numpy as np
import torch
import torch.nn as nn


class HMC_our(nn.Module):
    def __init__(self, kwargs):
        super(HMC_our, self).__init__()
        self.device = kwargs.device
        self.gamma = nn.Parameter(torch.tensor(np.log(kwargs.gamma), device=self.device))
        self.N = kwargs.N  # num leapfrogs
        self.alpha_logit = nn.Parameter(
            torch.tensor(np.log(kwargs.alpha) - np.log(1. - kwargs.alpha), device=self.device), requires_grad=False)
        self.use_barker = kwargs.use_barker  # If false, we are using standard MH ration, otherwise Barker ratio
        self.device_zero = torch.tensor(0., dtype=kwargs.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)
        self.uniform = torch.distributions.Uniform(low=self.device_zero,
                                                   high=self.device_one)  # distribution for transition making
        self.std_normal = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)
        self.naf = None
        if kwargs.neutralizing_idea:
            self.naf = kwargs.naf

    def _forward_step(self, q_old, x=None, k=None, target=None, p_old=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        q_old - current position
        x - data object (optional)
        k - auxiliary variable
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        q_new - new position
        p_new - new momentum
        """
        #         gamma = torch.sigmoid(self.gamma_logit)  # to make eps positive
        #         pdb.set_trace()
        gamma = torch.exp(self.gamma)
        p_flipped = -p_old

        p_ = p_flipped + gamma / 2. * self.get_grad(q=q_old, target=target,
                                                    x=x)  # NOTE that we are using log-density, not energy!
        q_ = q_old
        for l in range(self.N):
            q_ = q_ + gamma * p_
            if (l != self.N - 1):
                p_ = p_ + gamma * self.get_grad(q=q_, target=target,
                                                x=x)  # NOTE that we are using log-density, not energy!
        p_ = p_ + gamma / 2. * self.get_grad(q=q_, target=target,
                                             x=x)  # NOTE that we are using log-density, not energy!
        return q_, p_

    def make_transition(self, q_old, p_old, target_distr, k=None, x=None, flows=None, args=None, get_prior=None,
                        prior_flow=None, scales=None):
        """
        Input:
        q_old - current position
        p_old - current momentum
        target_distr - target distribution
        x - data object (optional)
        k - vector for partial momentum refreshment
        args - dict of arguments
        scales - if we train scales for momentum or not
        Output:
        q_new - new position
        p_new - new momentum
        log_jac - log jacobians of transformations
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        a - decision variables (0 or +1)
        q_upd - proposal states
        """
        ### Partial momentum refresh
        alpha = torch.sigmoid(self.alpha_logit)
        p_ref = p_old * alpha + torch.sqrt(1. - alpha ** 2) * k
        log_jac = p_old.shape[1] * torch.log(alpha) * torch.ones(q_old.shape[0], device=self.device)
        ############ Then we compute new points and densities ############
        q_upd, p_upd = self._forward_step(q_old=q_old, p_old=p_ref, k=k, target=target_distr, x=x)

        if scales is None:
            scales = torch.ones_like(p_old[0, :][None])
        target_log_density_f = target_distr.get_logdensity(z=q_upd, x=x, prior=get_prior, args=args,
                                                           prior_flow=prior_flow) + self.std_normal.log_prob(
            p_upd / scales).sum(
            1)
        target_log_density_old = target_distr.get_logdensity(z=q_old, x=x, prior=get_prior, args=args,
                                                             prior_flow=prior_flow) + self.std_normal.log_prob(
            p_ref / scales).sum(1)

        log_t = target_log_density_f - target_log_density_old
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                             log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)
        if self.use_barker:
            current_log_alphas_pre = log_t - log_1_t
        else:
            current_log_alphas_pre = torch.min(self.device_zero, log_t)

        log_probs = torch.log(self.uniform.sample((q_upd.shape[0],)))
        a = torch.where(log_probs < current_log_alphas_pre, self.device_one, self.device_zero)

        if self.use_barker:
            current_log_alphas = torch.where((a == 0.), -log_1_t, current_log_alphas_pre)
        else:
            expression = 1. - torch.exp(log_t)
            expression = torch.where(expression <= self.device_one * 1e-8, self.device_one * 1e-8, expression)
            corr_expression = torch.log(expression)
            current_log_alphas = torch.where((a == 0), corr_expression, current_log_alphas_pre)

        q_new = torch.where((a == self.device_zero)[:, None], q_old, q_upd)
        p_new = torch.where((a == self.device_zero)[:, None], p_ref, p_upd)

        return q_new, p_new, log_jac, current_log_alphas, a, q_upd

    def get_grad(self, q, target, x=None):
        q_init = q.clone().detach().requires_grad_(True)
        if self.naf:
            sum_log_jac = torch.zeros(q_init.shape[0], device=self.device)
            q_prev = q_init
            for naf in self.naf:
                q = naf(q_prev)
                sum_log_jac = sum_log_jac + naf.log_abs_det_jacobian(q_prev, q)
                q_prev = q
            grad = torch.autograd.grad((target.get_logdensity(x=x, z=q) + sum_log_jac).sum(), q_init)[
                0]
        else:
            grad = torch.autograd.grad(target.get_logdensity(x=x, z=q_init).sum(), q_init)[
                0]
        return grad


class HMC_vanilla(nn.Module):
    def __init__(self, kwargs):
        super(HMC_vanilla, self).__init__()
        self.device = kwargs.device
        self.N = kwargs.N

        self.alpha_logit = torch.tensor(np.log(kwargs.alpha) - np.log(1. - kwargs.alpha), device=self.device)
        #         self.gamma_logit = torch.tensor(np.log(kwargs.gamma) - np.log(1. - kwargs.gamma), device=self.device)
        self.gamma = torch.tensor(np.log(kwargs.gamma), device=self.device)
        self.use_partialref = kwargs.use_partialref  # If false, we are using full momentum refresh
        self.use_barker = kwargs.use_barker  # If false, we are using standard MH ration, otherwise Barker ratio
        self.device_zero = torch.tensor(0., dtype=kwargs.torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)
        self.uniform = torch.distributions.Uniform(low=self.device_zero,
                                                   high=self.device_one)  # distribution for transition making
        self.std_normal = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)

    def _forward_step(self, q_old, x=None, k=None, target=None, p_old=None, flows=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        q_old - current position
        x - data object (optional)
        k - auxiliary variable
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        q_new - new position
        p_new - new momentum
        """
        #         gamma = torch.sigmoid(self.gamma_logit)
        gamma = torch.exp(self.gamma)
        p_flipped = -p_old
        q_old.requires_grad_(True)
        p_ = p_flipped + gamma / 2. * self.get_grad(q=q_old, target=target, x=x,
                                                    flows=flows)  # NOTE that we are using log-density, not energy!
        q_ = q_old
        for l in range(self.N):
            q_ = q_ + gamma * p_
            if (l != self.N - 1):
                p_ = p_ + gamma * self.get_grad(q=q_, target=target, x=x,
                                                flows=flows)  # NOTE that we are using log-density, not energy!
        p_ = p_ + gamma / 2. * self.get_grad(q=q_, target=target, x=x,
                                             flows=flows)  # NOTE that we are using log-density, not energy!

        p_ = p_.detach()
        q_ = q_.detach()
        q_old.requires_grad_(False)

        return q_, p_

    def make_transition(self, q_old, p_old, target_distr, k=None, x=None, flows=None, args=None, get_prior=None,
                        prior_flow=None):
        """
        Input:
        q_old - current position
        p_old - current momentum
        target_distr - target distribution
        x - data object (optional)
        k - vector for partial momentum refreshment
        args - dict of arguments
        scales - if we train scales for momentum or not
        Output:
        q_new - new position
        p_new - new momentum
        log_jac - log jacobians of transformations
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        a - decision variables (0 or +1)
        q_upd - proposal states
        """
        # pdb.set_trace()
        ### Partial momentum refresh
        alpha = torch.sigmoid(self.alpha_logit)
        if self.use_partialref:
            p_ref = p_old * alpha + torch.sqrt(1. - alpha ** 2) * self.std_normal.sample(p_old.shape)
        else:
            p_ref = self.std_normal.sample(p_old.shape)

        ############ Then we compute new points and densities ############
        q_upd, p_upd = self._forward_step(q_old=q_old, p_old=p_ref, k=k, target=target_distr, x=x, flows=flows)

        target_log_density_f = target_distr.get_logdensity(z=q_upd, x=x) + self.std_normal.log_prob(p_upd).sum(1)
        target_log_density_old = target_distr.get_logdensity(z=q_old, x=x) + self.std_normal.log_prob(p_ref).sum(1)

        log_t = target_log_density_f - target_log_density_old
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                             log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)
        if self.use_barker:
            current_log_alphas_pre = log_t - log_1_t
        else:
            current_log_alphas_pre = torch.min(self.device_zero, log_t)

        log_probs = torch.log(self.uniform.sample((q_upd.shape[0],)))
        a = torch.where(log_probs < current_log_alphas_pre, self.device_one, self.device_zero)

        q_new = torch.where((a == self.device_zero)[:, None], q_old, q_upd)
        p_new = torch.where((a == self.device_zero)[:, None], p_ref, p_upd)

        return q_new, p_new, None, None, a, q_upd

    def get_grad(self, q, target, x=None, flows=None):
        q_init = q.detach().requires_grad_(True)
        if flows:
            log_jacobian = 0.
            q_prev = q_init
            q_new = q_init
            for i in range(len(flows)):
                q_new = flows[i](q_prev)
                log_jacobian += flows[i].log_abs_det_jacobian(q_prev, q_new)
                q_prev = q_new
            s = target.get_logdensity(x=x, z=q_new) + log_jacobian
            grad = torch.autograd.grad(s.sum(), q_init)[0]
        else:
            s = target.get_logdensity(x=x, z=q_init)
            grad = torch.autograd.grad(s.sum(), q_init)[0]
        return grad


class Reverse_kernel(nn.Module):
    def __init__(self, kwargs):
        super(Reverse_kernel, self).__init__()
        self.device = kwargs.device
        self.device_one = torch.tensor(1., dtype=kwargs.torchType, device=self.device)
        self.z_dim = kwargs.z_dim
        self.K = kwargs.K
        self.linear_z = nn.Linear(in_features=self.z_dim, out_features=5 * self.K)
        self.linear_h = nn.Linear(in_features=self.z_dim, out_features=5 * self.K)
        self.linear_hidden = nn.Linear(in_features=10 * self.K, out_features=5 * self.K)
        self.linear_out = nn.Linear(in_features=5 * self.K, out_features=self.K)

    def forward(self, z_fin, h, a):
        z_ = torch.relu(self.linear_z(z_fin))
        h_ = torch.relu(self.linear_h(h))
        cat_z_h = torch.cat([z_, h_], dim=1)
        h1 = torch.relu(self.linear_hidden(cat_z_h))
        probs = torch.sigmoid(self.linear_out(h1))
        probs = torch.where(a == self.device_one, probs, self.device_one - probs)
        log_prob = torch.sum(torch.log(probs), dim=1)
        return log_prob
