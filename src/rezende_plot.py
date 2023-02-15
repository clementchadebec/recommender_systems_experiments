import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from kernels import HMC_vanilla, HMC_our


class RNVP(nn.Module):
    def __init__(self, args):
        super(RNVP, self).__init__()

        self.hidden_dim = args["hidden_dim"]
        self.z_dim = args["z_dim"]
        mask = np.array([[i % 2 for i in range(args['z_dim'])],
                         [(i + 1) % 2 for i in range(args['z_dim'])]]).astype(np.float32)
        self.masks = torch.tensor(mask, dtype=args.torchType, device=args.device)
        self.t = torch.nn.ModuleList([args.nett() for _ in range(len(self.masks))])
        self.s = torch.nn.ModuleList([args.nets() for _ in range(len(self.masks))])

    def _forward_step(self, z_old):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        Output:
        z_new - new position = T(z_old)
        log_jac - log_jacobian of the transformation
        """
        log_det_J = z_old.new_zeros(z_old.shape[0])

        for i in range(len(self.t)):
            z_old_ = z_old * self.masks[i]
            s = self.s[i](z_old_) * (1. - self.masks[i])
            t = self.t[i](z_old_) * (1. - self.masks[i])
            z_old = z_old_ + (1. - self.masks[i]) * (z_old * torch.exp(s) + t)
            if len(s.squeeze()) == 2:
                log_det_J += s.squeeze().sum(dim=1)
            else:
                log_det_J += s.sum(dim=1)
        z_new = z_old
        return z_new, log_det_J


class Target():
    def __init__(self, cur_dat, energy=False):
        self.data = cur_dat
        self.energy = energy

    def get_logdensity(self, z, x=None, prior=None, args=None, prior_flow=None):
        if self.energy:
            z = z['points']
        if self.data == 't1':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            z_norm = torch.norm(z, 2, 1)
            add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
            add2 = - torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + \
                               torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2) + 1e-9)
            if self.energy:
                return add1 + add2
            else:
                return -add1 - add2

        elif self.data == 't2':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            if self.energy:
                return 0.5 * ((z[:, 1] - w1) / 0.4) ** 2
            else:
                return -0.5 * ((z[:, 1] - w1) / 0.4) ** 2
        elif self.data == 't3':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            w2 = 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
            in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.35) ** 2)
            in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w2) / 0.35) ** 2)
            if self.energy:
                return -torch.log(in1 + in2 + 1e-9)
            else:
                return torch.log(in1 + in2 + 1e-9)
        elif self.data == 't4':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            w3 = 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)
            in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.4) ** 2)
            in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w3) / 0.35) ** 2)
            if self.energy:
                return -torch.log(in1 + in2 + 1e-9)
            else:
                return torch.log(in1 + in2 + 1e-9)
        else:
            raise RuntimeError

    def get_density(self, z, x=None):
        density = self.distr.log_prob(z).exp()
        return density

    def get_samples(self, n):
        raise NotImplementedError


def run_rezende(args):
    args.z_dim = 2
    args.data_dim = 2
    args.n_samples = 100000
    args.n_batches = 5000

    prior = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                       scale=torch.tensor(1., dtype=args.torchType, device=args.device))
    # # rnvp
    # all_samples = run_rezende_rnvp(args, prior)
    # for i, samples in enumerate(all_samples):
    #     np.savetxt('../rezende_data/rnvp_{}.txt'.format(i), samples)

    # # hoffman
    # all_samples = run_rezende_hoffman(args, prior)
    # for i, samples in enumerate(all_samples):
    #     np.savetxt('../rezende_data/hoffman_{}.txt'.format(i), samples)

    # methmc
    all_samples = run_rezende_methmc(args, prior)
    for i, samples in enumerate(all_samples):
        np.savetxt('../rezende_data/methmc_{}.txt'.format(i), samples)


def run_rezende_hoffman(args, prior):
    # hoffman
    all_samples = []
    for cur_dat in ['t1', 't2', 't3', 't4']:
        target = Target(cur_dat).get_logdensity
        ##### Minimize KL first
        mu_init_hoff = nn.Parameter(torch.zeros(args.data_dim, device=args.device, dtype=args.torchType))
        sigma_init_hoff = nn.Parameter(torch.ones(args.data_dim, device=args.device, dtype=args.torchType))
        optimizer = torch.optim.Adam(params=[mu_init_hoff, sigma_init_hoff])
        scheduler = MultiStepLR(optimizer, [200, 500, 750, 1000, 1500, 2000], gamma=0.3)
        for i in tqdm(range(args.n_batches)):
            u_init = prior.sample((500, 2))
            q_init = mu_init_hoff + nn.functional.softplus(sigma_init_hoff) * u_init

            current_kl = prior.log_prob(u_init).mean() - torch.mean(
                nn.functional.softplus(sigma_init_hoff).log()) - target(z=q_init).mean()
            (current_kl).backward()  ## minimize KL
            optimizer.step()
            optimizer.zero_grad()
            if i % 1000 == 0:
                print(current_kl.mean().cpu().detach().numpy())
            scheduler.step()
        mu_init_hoff.requires_grad_(False)
        sigma_init_hoff.requires_grad_(False)

        samples = prior.sample((args.n_samples, 2))

        q_new = mu_init_hoff + samples * nn.functional.softplus(
            sigma_init_hoff)
        p_new = prior.sample(q_new.shape)

        target = Target(cur_dat)
        vanilla_kernel = HMC_vanilla(args)

        for _ in tqdm(range(args.K)):
            q_new, p_new, _, _, _, _ = vanilla_kernel.make_transition(q_old=q_new, p_old=p_new, target_distr=target)
        samples_hoffman = q_new
        all_samples.append(samples_hoffman.cpu().detach().numpy())

    return all_samples


def run_rezende_rnvp(args, prior):
    # rnvp
    args['z_dim'] = 2
    args['hidden_dim'] = 4

    args['nets'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']), nn.Tanh())
    args['nett'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']))
    all_samples = []
    ################################################################################################
    for cur_dat in ['t1', 't2', 't3', 't4']:
        target = Target(cur_dat).get_logdensity
        transitions_rnvp = nn.ModuleList([RNVP(args=args).to(args.device) for _ in range(2)])
        optimizer_rnvp = torch.optim.Adam(params=transitions_rnvp.parameters())
        scheduler = MultiStepLR(optimizer_rnvp, [750, 1500, 2000], gamma=0.3)
        for current_b in tqdm(range(args.n_batches)):
            optimizer_rnvp.zero_grad()
            u = prior.sample((500, args.z_dim))
            sum_log_jacobian = torch.zeros(u.shape[0], dtype=args.torchType,
                                           device=args.device)  # for log_jacobian accumulation
            z = u
            for k in range(len(transitions_rnvp)):
                z_upd, log_jac = transitions_rnvp[k]._forward_step(z)
                sum_log_jacobian = sum_log_jacobian + log_jac  # refresh log jacobian
                z = z_upd
            log_p = target(z).mean()
            log_q = prior.log_prob(u).mean() - sum_log_jacobian.mean()
            elbo = log_p - log_q
            (-elbo).backward()
            optimizer_rnvp.step()
            optimizer_rnvp.zero_grad()
            if current_b % 1000 == 0:
                print('Current ELBO is ', elbo.cpu().detach().item())
            scheduler.step()

        samples = prior.sample((args.n_samples, 2))
        with torch.no_grad():
            for rnvp in transitions_rnvp:
                samples, _ = rnvp._forward_step(samples)
        all_samples.append(samples.cpu().detach().numpy())
    return all_samples


def run_rezende_methmc(args, prior):
    all_samples = []
    for cur_dat in ['t1', 't2', 't3', 't4']:
        target = Target(cur_dat)
        transitions = nn.ModuleList([HMC_our(kwargs=args).to(args.device) for _ in range(args['K'])])
        if not args.learntransitions:
            for p in transitions.parameters():
                p.requires_grad_(False)
        else:
            for k in range(len(transitions)):
                transitions[k].alpha_logit.requires_grad_(False)

        torch_log_2 = torch.tensor(np.log(2), device=args.device, dtype=args.torchType)
        momentum_scale = nn.Parameter(torch.zeros(args.z_dim, device=args.device, dtype=args.torchType)[None, :],
                                      requires_grad=args.learnscale)
        mu_init = nn.Parameter(torch.zeros(args.data_dim, device=args.device, dtype=args.torchType))
        sigma_init = nn.Parameter(torch.ones(args.data_dim, device=args.device, dtype=args.torchType))
        optimizer = torch.optim.Adam(list(transitions.parameters()) + [momentum_scale, mu_init, sigma_init])
        scheduler = MultiStepLR(optimizer, [200, 500, 750, 1000, 1500, 2500], gamma=0.3)
        for bnum in range(args.n_batches):
            u = prior.sample((500, 2))
            z = mu_init + nn.functional.softplus(sigma_init) * u

            sum_log_alpha = torch.zeros_like(z[:, 0])
            sum_log_jacobian = torch.zeros_like(z[:, 0])
            scales = torch.exp(momentum_scale)
            p_ = prior.sample(z.shape) * scales
            p_old = p_.clone()
            all_directions = torch.tensor([], device=args.device)
            for i in range(args.K):
                cond_vector = prior.sample(p_.shape) * scales
                z, p_, log_jac, current_log_alphas, directions, _ = transitions[i].make_transition(q_old=z,
                                                                                                   p_old=p_,
                                                                                                   k=cond_vector,
                                                                                                   target_distr=target,
                                                                                                   scales=scales)
                sum_log_alpha = sum_log_alpha + current_log_alphas
                sum_log_jacobian = sum_log_jacobian + log_jac
                all_directions = torch.cat([all_directions, directions.view(-1, 1)], dim=1)

            # loglikelihood part
            log_likelihood = target.get_logdensity(z).mean() + prior.log_prob(p_ / scales).mean()
            # compute objective
            log_r = -args.K * torch_log_2
            log_sigma = torch.log(nn.functional.softplus(sigma_init))
            log_q = prior.log_prob(u).mean() - log_sigma.mean() + prior.log_prob(
                p_old / scales).mean() - sum_log_jacobian.mean() + sum_log_alpha.mean()
            elbo_full = log_likelihood + log_r - log_q
            grad_elbo = elbo_full + elbo_full.detach() * torch.mean(sum_log_alpha)
            (-grad_elbo).backward()

            optimizer.step()
            optimizer.zero_grad()

            if bnum % 1000 == 0:
                if args.learnscale:
                    print('Min scale', torch.exp(momentum_scale.detach()).min().item(), 'Max scale',
                          torch.exp(momentum_scale.detach()).max().item())
                print(elbo_full.cpu().detach().numpy())
                for k in range(args.K):
                    print('k =', k)
                    print('0: {} and for +1: {}'.format((all_directions[:, k] == 0.).to(float).mean(),
                                                        (all_directions[:, k] == 1.).to(float).mean()))
                    print('autoreg:', torch.sigmoid(transitions[k].alpha_logit.detach()).item())
                    print('stepsize', torch.exp(transitions[k].gamma.detach()).item())
                    print('-' * 100)

            if np.isnan(elbo_full.cpu().detach().numpy()):
                break
            scheduler.step()

        momentum_scale.requires_grad_(False)
        mu_init.requires_grad_(False)
        sigma_init.requires_grad_(False)
        scales = torch.exp(momentum_scale)

        samples = mu_init + prior.sample((args.n_samples, 2)) * nn.functional.softplus(sigma_init)
        p_ = prior.sample(samples.shape) * scales
        for i in range(args.K):
            cond_vector = prior.sample(p_.shape) * scales
            samples, p_, _, _, _, _ = transitions[i].make_transition(q_old=samples,
                                                                     p_old=p_,
                                                                     k=cond_vector,
                                                                     target_distr=target,
                                                                     scales=scales)
        all_samples.append(samples.cpu().detach().numpy())
    return all_samples
