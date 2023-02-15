import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
import pdb
from kernels import HMC_our, Reverse_kernel
import copy

import pdb


def truncated_normal(size, std=1):
    values = truncnorm.rvs(-2. * std, 2. * std, size=size)
    return values


def make_linear_network(dims, encoder=False):
    layer_list = nn.ModuleList([])
    for i in range(len(dims) - 1):
        if i == len(dims) - 2 and encoder:
            layer_list.append(nn.Linear(dims[i], 2 * dims[i + 1]))
        else:
            layer_list.append(nn.Linear(dims[i], dims[i + 1]))
        layer_list[-1].weight = nn.init.xavier_uniform_(layer_list[-1].weight)
        layer_list[-1].bias = nn.Parameter(
            torch.tensor(truncated_normal(layer_list[-1].bias.shape, 0.001), dtype=torch.float32))
        if i != len(dims) - 2 and encoder:
            layer_list.append(nn.BatchNorm1d(dims[i + 1]))
        layer_list.append(nn.Tanh())
    layer_list = layer_list[:-1]
    model = nn.Sequential(*layer_list)
    return model

def logprob_normal(z, mu=torch.tensor(0.), logvar=torch.tensor(0.)):
    '''
    Similar to torch.distributions.Normal(mu, exp(logvar * 0.5)).log_prob(z)
    '''
    return -(z - mu) ** 2 / (2 * torch.exp(logvar)) - 0.5 * logvar - 0.919


class MultiVAE(nn.Module):
    '''
    Model described in the paper Liang, Dawen, et al. "Variational autoencoders for collaborative filtering." Proceedings of the 2018 World Wide Web Conference. 2018.
    '''

    def __init__(self, p_dims, q_dims=None, args=None):
        super(MultiVAE, self).__init__()

        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
            q_dims = self.q_dims
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims

        self.encoder = make_linear_network(q_dims, encoder=True)
        print(self.encoder)

        self.decoder = make_linear_network(p_dims)
        print(self.decoder)

        self.dropout = nn.Dropout()

        device_zero = torch.tensor(0., dtype=torch.float32, device=args.device)
        device_one = torch.tensor(1., dtype=torch.float32, device=args.device)
        self.std_normal = torch.distributions.Normal(loc=device_zero, scale=device_one)

    def forward(self, x_initial, is_training_ph=1.):
        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)

        enc_out = self.encoder(x)
        mu, logvar = enc_out[:, :self.q_dims[-1]], enc_out[:, self.q_dims[-1]:]
        std = torch.exp(0.5 * logvar)

        u = self.std_normal.sample(mu.shape)
        z = mu + is_training_ph * u * std

        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))
        logits = self.decoder(z)

        return logits, KL

class MultiIWAE(nn.Module):
    '''
    Model described in the paper Liang, Dawen, et al. "Variational autoencoders for collaborative filtering." Proceedings of the 2018 World Wide Web Conference. 2018.
    '''

    def __init__(self, p_dims, q_dims=None, args=None):
        super(MultiIWAE, self).__init__()

        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
            q_dims = self.q_dims
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.latent_dim = q_dims[-1]
        self.encoder = make_linear_network(q_dims, encoder=True)
        print(self.encoder)

        self.decoder = make_linear_network(p_dims)
        print(self.decoder)
        self.K = 5 if args.K is None else args.K 
        self.dropout = nn.Dropout()
        self.clamp_kl = 1e6
        device_zero = torch.tensor(0., dtype=torch.float32, device=args.device)
        device_one = torch.tensor(1., dtype=torch.float32, device=args.device)
        self.std_normal = torch.distributions.Normal(loc=device_zero, scale=device_one)

    def forward(self, x_initial, is_training_ph=1.):
        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)

        enc_out = self.encoder(x)
        mu, logvar = enc_out[:, :self.q_dims[-1]], enc_out[:, self.q_dims[-1]:]
        std = torch.exp(0.5 * logvar)
        
        dims_ = (self.K,) + mu.shape
        u = self.std_normal.sample(dims_)
        z = mu + is_training_ph * u * std

        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))
        logits = self.decoder(z)
        
        return logits, KL, z, mu, logvar
    
    def loss_function(self, x, anneal, is_training_ph=1.):
        logits, KL, z, mu, logvar = self.forward(x, is_training_ph)
        # loglikelihood part
        log_Q = logprob_normal(z,mu,logvar).view((self.K, -1, self.latent_dim)).sum(-1)
        log_Pr = (-0.5 * z ** 2).view((self.K, -1, self.latent_dim)).sum(-1)
        
        log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
        BCE = - torch.mean(torch.sum(log_softmax_var * x[None,...], dim=2))
        KL_eq = torch.clamp(log_Q - log_Pr, -self.clamp_kl, self.clamp_kl)
        
        log_weight = - BCE - KL_eq
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()
        loss = torch.mean(torch.sum(weight * (BCE + anneal * KL_eq), 0))

        return loss, torch.sum(BCE * weight, dim=0).mean()

        



class Target(nn.Module):
    def __init__(self, dec, device='cpu'):
        super(Target, self).__init__()
        self.decoder = dec
        self.prior = torch.distributions.Normal(loc=torch.tensor(0., device=device, dtype=torch.float32),
                                                scale=torch.tensor(1., device=device, dtype=torch.float32))

    def get_logdensity(self, x, z, prior=None, args=None, prior_flow=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x, z)
        """
        logits = self.decoder(z)
        log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
        log_density = torch.sum(log_softmax_var * x, dim=1) + self.prior.log_prob(z).sum(1)
        return log_density


class Multi_our_VAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, args=None):
        super(Multi_our_VAE, self).__init__()

        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
            q_dims = self.q_dims
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims

        ## Define encoder
        self.encoder = make_linear_network(q_dims, encoder=True)
        print(self.encoder)

        ## Define target(decoder)
        decoder = make_linear_network(p_dims)
        print(decoder)
        self.target = Target(dec=decoder, device=args.device)

        ## Define transitions
        self.K = args.K
        self.transitions = nn.ModuleList([HMC_our(kwargs=args).to(args.device) for _ in range(args['K'])])

        ## Define reverse kernel (if it is needed)
        self.learnable_reverse = args.learnable_reverse
        if args.learnable_reverse:
            self.reverse_kernel = Reverse_kernel(kwargs=args).to(args.device)

        self.dropout = nn.Dropout()

        device_zero = torch.tensor(0., dtype=torch.float32, device=args.device)
        device_one = torch.tensor(1., dtype=torch.float32, device=args.device)
        self.std_normal = torch.distributions.Normal(loc=device_zero, scale=device_one)
        self.torch_log_2 = torch.tensor(np.log(2), device=args.device, dtype=args.torchType)
        self.annealing = args.annealing
        self.momentum_scale = nn.Parameter(torch.zeros(args.z_dim, device=args.device, dtype=args.torchType)[None, :],
                                           requires_grad=args.learnscale)  ## Comment this line for this case validating ml20m models with option annealing = False

    def forward(self, x_initial, is_training_ph=1.):
        # self.momentum_scale = nn.Parameter(torch.zeros(self.q_dims[-1], device=x_initial.device, dtype=torch.float32)[None, :],
        #                                    requires_grad=False) Uncomment this line for this case validating ml20m models with option annealing = False
        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)

        enc_out = self.encoder(x)
        mu, logvar = enc_out[:, :self.q_dims[-1]], enc_out[:, self.q_dims[-1]:]
        std = torch.exp(0.5 * logvar)
        sum_log_alpha = torch.zeros_like(mu[:, 0])
        sum_log_jacobian = torch.zeros_like(mu[:, 0])

        u = self.std_normal.sample(mu.shape)
        z = mu + is_training_ph * u * std

        scales = torch.exp(self.momentum_scale)
        p_ = self.std_normal.sample(z.shape) * scales
        p_old = p_.clone()

        all_directions = torch.tensor([], device=x.device)

        for i in range(self.K):
            cond_vector = self.std_normal.sample(p_.shape) * scales
            z, p_, log_jac, current_log_alphas, directions, _ = self.transitions[i].make_transition(q_old=z, x=x,
                                                                                                    p_old=p_,
                                                                                                    k=cond_vector,
                                                                                                    target_distr=self.target,
                                                                                                    scales=scales)
            sum_log_alpha = sum_log_alpha + current_log_alphas
            sum_log_jacobian = sum_log_jacobian + log_jac
            all_directions = torch.cat([all_directions, directions.view(-1, 1)], dim=1)

        ## logdensity of Variational family
        log_sigma = torch.log(std)
        log_q = self.std_normal.log_prob(u) + self.std_normal.log_prob(p_old / scales) - log_sigma
        log_aux = sum_log_alpha - sum_log_jacobian

        ## logdensity of prior
        log_priors = self.std_normal.log_prob(z) + self.std_normal.log_prob(p_ / scales)

        ## logits
        logits = self.target.decoder(z)

        ## logdensity of reverse (if needed)
        if self.learnable_reverse:
            log_r = self.reverse_kernel(z_fin=z.detach(), h=mu.detach(), a=all_directions)
        else:
            log_r = -self.K * self.torch_log_2

        return logits, log_q, log_aux, log_priors, log_r, sum_log_alpha, all_directions


    
class SimCLR_reco_model(nn.Module):
    
    def __init__(self, args,  p_dims,  q_dims=None, layers_proj = None):
        super(SimCLR_reco_model, self).__init__()

        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
            q_dims = self.q_dims
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.args = args
        self.args.n_views = 2 
        self.n_views = 2
        self.encoder = make_linear_network(q_dims, encoder=False)
        print(self.encoder)

        self.decoder = make_linear_network(p_dims)
        print(self.decoder)
        if args.dropout_rate is None:
            self.dropout = nn.Dropout()
        else:
            self.dropout = nn.Dropout(p=args.dropout_rate)
        device_zero = torch.tensor(0., dtype=torch.float32, device=args.device)
        device_one = torch.tensor(1., dtype=torch.float32, device=args.device)

        self.args = args
        if args.criterion_dec is None:
            self.criterion_dec = nn.MSELoss().to(self.args.device)
        else:
            self.criterion_dec = args.criterion_dec.to(self.args.device)
            
        self.criterion_enc = nn.CrossEntropyLoss().to(self.args.device)
        
        if layers_proj is None:
            self.projection_head = nn.Identity
            self.proj = False
        else:
            self.projection_head = make_linear_network(layers_proj, encoder = True)
            self.proj = True
        
    def forward(self, x):
        return self.encoder(x)
    
    
    def simclr_info_nce_loss(self, x_initial): 
        #pdb.set_trace()
        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x1 = self.dropout(x_normed)
        x2 = self.dropout(x_normed)
        batch_size = x1.shape[0]
        
        x = torch.cat([x1, x2], dim=0)
        features = self.encoder(x)
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0) ##Vector of size 2*bs, because we have two psoitive exampkes here x1 and x2
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    
    def step_encoder(self, batch):
        logits, labels = self.simclr_info_nce_loss(batch)
        loss = self.criterion_enc(logits, labels)
        return loss
    
    
    def step_decoder(self, x):
        #pdb.set_trace()
        z_ = self.encoder(x)
        pred_logits = self.decoder(z_)
        log_softmax_var = nn.LogSoftmax(dim=-1)(pred_logits)
        neg_ll = -torch.mean(torch.sum(log_softmax_var * x, dim=1))

        pred = nn.Softmax(dim=-1)(pred_logits)
        
        return self.criterion_dec(pred, x), pred
        
        
class BYOL_reco_model(nn.Module):
    def __init__(self, args, p_dims, q_dims= None, layers_pred = None):#online_network, target_network, predictor, optimizer, device, **params):
        super(BYOL_reco_model, self).__init__()
        #self.online_network = online_network
        #self.target_network = target_network
        #self.optimizer = optimizer
        #self.device = device
        self.m = .996 if (args['m'] is None) else args.m
        print('m ', self.m)
        #self.batch_size = params['batch_size']
        #self.num_workers = params['num_workers']
        #self.checkpoint_interval = params['checkpoint_interval']
        #_create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])
        
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
            q_dims = self.q_dims
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        dim_feat = self.p_dims[0]
        if layers_pred is None:
            layers_pred = [dim_feat, 2*dim_feat, dim_feat]
        self.args = args
        self.online_network = make_linear_network(q_dims, encoder=False)
        self.target_network = make_linear_network(q_dims, encoder=False)
        print(self.target_network)
        self.predictor = make_linear_network(layers_pred, encoder = False)
        
        self.decoder = make_linear_network(p_dims)
        print(self.decoder)
        if args.dropout_rate is None:
            self.dropout = nn.Dropout()
        else:
            self.dropout = nn.Dropout(p=args.dropout_rate)
        device_zero = torch.tensor(0., dtype=torch.float32, device=args.device)
        device_one = torch.tensor(1., dtype=torch.float32, device=args.device)

        self.args = args
        if args.criterion_dec is None:
            self.criterion_dec = nn.MSELoss().to(self.args.device)
        else:
            self.criterion_dec = args.criterion_dec.to(self.args.device)
            
        #self.criterion_enc = nn.CrossEntropyLoss().to(self.args.device)
        
        
    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    def update(self, x):
        l2 = torch.sum(x ** 2, 1)[..., None]
        x_normed = x / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x1 = self.dropout(x_normed)
        x2 = self.dropout(x_normed)
        batch_size = x1.shape[0]
        
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(x1))
        predictions_from_view_2 = self.predictor(self.online_network(x2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(x1)
            targets_to_view_1 = self.target_network(x2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean() 
    
    
    
    
    def step_decoder(self, x):
        #pdb.set_trace()
        z_ = self.online_network(x)
        pred_logits = self.decoder(z_)
        log_softmax_var = nn.LogSoftmax(dim=-1)(pred_logits)
        neg_ll = -torch.mean(torch.sum(log_softmax_var * x, dim=1))

        pred = nn.Softmax(dim=-1)(pred_logits)
        
        return self.criterion_dec(pred, x), pred
        
        
   