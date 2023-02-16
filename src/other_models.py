import torch
import torch.nn as nn

from pythae.models import VAMP, VAMPConfig, VAE, VAEConfig, VAE_LinNF, VAE_LinNF_Config, VAE_IAF, VAE_IAF_Config, CIWAE, CIWAEConfig
from pythae.models.base.base_utils import ModelOutput


class MultiVAEPythae(VAE):
    def __init__(self, model_config: VAEConfig, encoder=None, decoder=None) -> None:
        super().__init__(model_config, encoder, decoder)
        self.dropout = nn.Dropout()

    def forward(self, x_initial, is_training_ph=1., **kwargs):

        anneal = kwargs.pop("anneal", 1)

        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)

        enc_out = self.encoder(x)

        mu = enc_out.embedding
        logvar = enc_out.log_covariance

        std = torch.exp(0.5 * logvar)

        u = torch.randn_like(std)
        z = mu + is_training_ph * u * std

        logits = self.decoder(z).reconstruction

        loss, elbo = self.loss_function(logits=logits, x=x_initial, mu=mu, logvar=logvar, anneal=anneal)

        return ModelOutput(
            loss=loss,
            elbo=elbo,
            logits=logits
        )

    def loss_function(self, logits, x, mu, logvar, anneal):

        # loglikelihood part
        log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
        neg_ll = -torch.mean(torch.sum(log_softmax_var * x, dim=1))
        
        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))

        # compute objective
        loss = neg_ll + anneal * KL

        return loss, neg_ll + KL



class MultiVAMP(VAMP):
    def __init__(self, model_config: VAMPConfig, encoder=None, decoder=None) -> None:
        super().__init__(model_config, encoder, decoder)
        self.dropout = nn.Dropout()

    def forward(self, x_initial, is_training_ph=1., **kwargs):

        anneal = kwargs.pop("anneal", 1)

        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)

        enc_out = self.encoder(x)

        mu = enc_out.embedding
        logvar = enc_out.log_covariance

        std = torch.exp(0.5 * logvar)

        u = torch.randn_like(std)
        z = mu + is_training_ph * u * std

        logits = self.decoder(z).reconstruction

        loss, elbo = self.loss_function(logits=logits, x=x_initial, mu=mu, logvar=logvar, z=z, anneal=anneal)

        return ModelOutput(
            loss=loss,
            elbo=elbo,
            logits=logits
        )


    def loss_function(self, logits, x, mu, logvar, z, anneal):

        log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
        BCE = - torch.mean(torch.sum(log_softmax_var * x[None,...], dim=2))

        # prior
        log_p_z = self._log_p_z(z)

        log_q_z = (-0.5 * (logvar + torch.pow(z - mu, 2) / logvar.exp())).sum(dim=1)
        KL = -(log_p_z - log_q_z).mean()


        return BCE + anneal * KL, BCE + KL


class MultiVAELinNF(VAE_LinNF):
    def __init__(self, model_config: VAE_LinNF_Config, encoder=None, decoder=None) -> None:
        super().__init__(model_config, encoder, decoder)
        self.dropout = nn.Dropout()


    def forward(self, x_initial, is_training_ph=1., **kwargs):

        anneal = kwargs.pop("anneal", 1)

        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)

        enc_out = self.encoder(x)

        mu = enc_out.embedding
        logvar = enc_out.log_covariance

        std = torch.exp(0.5 * logvar)

        u = torch.randn_like(std)
        z = mu + is_training_ph * u * std

        z0 = z

        log_abs_det_jac = torch.zeros((z0.shape[0],)).to(z.device)

        for layer in self.net:
            layer_output = layer(z)
            z = layer_output.out
            log_abs_det_jac += layer_output.log_abs_det_jac

        logits = self.decoder(z).reconstruction

        loss, elbo = self.loss_function(
            logits=logits,
            x=x_initial,
            mu=mu,
            logvar=logvar,
            z0=z0,
            zk=z,
            log_abs_det_jac=log_abs_det_jac,
            anneal=anneal
        )

        return ModelOutput(
            loss=loss,
            elbo=elbo,
            logits=logits
        )

    def loss_function(self, logits, x, mu, logvar, z0, zk, log_abs_det_jac, anneal):

        log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
        BCE = - torch.mean(torch.sum(log_softmax_var * x[None,...], dim=2))

         # starting gaussian log-density
        log_prob_z0 = (
            -0.5 * (logvar + torch.pow(z0 - mu, 2) / torch.exp(logvar))
        ).sum(dim=1)

        # prior log-density
        log_prob_zk = (-0.5 * torch.pow(zk, 2)).sum(dim=1)

        KL = (log_prob_z0 - log_prob_zk - log_abs_det_jac).mean()

        return BCE + anneal * KL, BCE + KL 
    

class MultiVAEIAF(VAE_IAF):
    def __init__(self, model_config: VAE_IAF_Config, encoder=None, decoder=None) -> None:
        super().__init__(model_config, encoder, decoder)
        self.dropout = nn.Dropout()


    def forward(self, x_initial, is_training_ph=1., **kwargs):

        anneal = kwargs.pop("anneal", 1)

        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)

        enc_out = self.encoder(x)

        mu = enc_out.embedding
        logvar = enc_out.log_covariance

        std = torch.exp(0.5 * logvar)

        u = torch.randn_like(std)
        z = mu + is_training_ph * u * std

        z0 = z

        # Pass it through the Normalizing flows
        flow_output = self.iaf_flow.inverse(z)  # sampling

        z = flow_output.out
        log_abs_det_jac = flow_output.log_abs_det_jac

        logits = self.decoder(z).reconstruction

        loss, elbo = self.loss_function(
            logits=logits,
            x=x_initial,
            mu=mu,
            logvar=logvar,
            z0=z0,
            zk=z,
            log_abs_det_jac=log_abs_det_jac,
            anneal=anneal
        )

        return ModelOutput(
            loss=loss,
            elbo=elbo,
            logits=logits
        )
    

    def loss_function(self, logits, x, mu, logvar, z0, zk, log_abs_det_jac, anneal):

        log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
        BCE = - torch.mean(torch.sum(log_softmax_var * x[None,...], dim=2))

         # starting gaussian log-density
        log_prob_z0 = (
            -0.5 * (logvar + torch.pow(z0 - mu, 2) / torch.exp(logvar))
        ).sum(dim=1)

        # prior log-density
        log_prob_zk = (-0.5 * torch.pow(zk, 2)).sum(dim=1)

        KL = (log_prob_z0 - log_prob_zk - log_abs_det_jac).mean()

        return BCE + anneal * KL, BCE + KL
    

class MultiCIWAE(CIWAE):
    def __init__(self, model_config: VAE_IAF_Config, encoder=None, decoder=None) -> None:
        super().__init__(model_config, encoder, decoder)
        self.dropout = nn.Dropout()


    def forward(self, x_initial, is_training_ph=1., **kwargs):

        anneal = kwargs.pop("anneal", 1)

        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)

        enc_out = self.encoder(x)

        mu = enc_out.embedding
        logvar = enc_out.log_covariance

        std = torch.exp(0.5 * logvar)

        dims_ = (self.n_samples,) + mu.shape
        u = torch.distributions.Normal(
            loc=torch.tensor(0., device=mu.device),
            scale=torch.tensor(1., device=mu.device)
        ).sample(dims_)
        z = mu + is_training_ph * u * std

        logits = self.decoder(z).reconstruction

        loss, elbo = self.loss_function(
            logits=logits,
            x=x_initial,
            mu=mu,
            logvar=logvar,
            z=z,
            anneal=anneal
        )

        return ModelOutput(
            loss=loss,
            elbo=elbo,
            logits=logits.mean(0)
        )

    def loss_function(self, logits, x, mu, logvar, z, anneal):
        log_Q = (
            -(z - mu) ** 2 / (2 * torch.exp(logvar)) - 0.5 * logvar
        ).view((self.n_samples, -1, self.latent_dim)).sum(-1)
        log_Pr = -0.5 * (z ** 2).view((self.n_samples, -1, self.latent_dim)).sum(-1)

        log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
        BCE = - torch.mean(torch.sum(log_softmax_var * x[None,...], dim=2))
        KL = log_Q - log_Pr

        log_weight = - BCE - KL
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()

        iwae_elbo = torch.mean(torch.sum(weight * (BCE + KL), 0))
        vae_elbo = torch.mean(BCE + KL)

        return self.beta * vae_elbo + (1 - self.beta) * iwae_elbo, self.beta * vae_elbo + (1 - self.beta) * iwae_elbo