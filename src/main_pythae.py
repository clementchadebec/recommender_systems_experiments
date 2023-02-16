import argparse
import random

import wandb

import numpy as np
import torch
from training import train_pythae_model
from nn import Encoder, Decoder

from args import get_args
from data import Dataset

parser = argparse.ArgumentParser(
    description='VAE for CF')

parser.add_argument('-data', type=str, choices=['ml20m', 'foursquare', 'gowalla', 'Rezende'],
                    help='Specify, which data to use', required=True)
parser.add_argument('-model', type=str,
                    choices=['MultiVAEIAF', 'MultiVAELinNF', 'MultiVAE', 'MultiVAMP', 'MultiCIWAE'],
                    help='Specify, which model to use', required=True)
parser.add_argument('-learnable_reverse', type=str, choices=['True', 'False'],
                    help='If we use learnable reverse or not', default='False')
parser.add_argument('-annealing', type=str, choices=['True', 'False'],
                    help='If we use annealing or not', default='True')
parser.add_argument('-learntransitions', type=str, choices=['True', 'False'],
                    help='If we train transitions or not', default='False')

parser.add_argument('-beta_ciwae', type=float, default=.5, help="beta factor in CIWAE")
parser.add_argument('-n_samples', type=int, default=5, help="num importance samples in IWAE based methods")
parser.add_argument('-n_components', type=int, default=100, help="num components in VAMP")
parser.add_argument('-n_made_blocks', type=int, default=3, help="num of IAF flows in VAEIAF")


parser.add_argument('-learnscale', type=str, choices=['True', 'False'],
                    help='If we train diagonal matrix for momentum rescale or not', default='False')

parser.add_argument('-gpu', type=int, help='If >=0 - id of device, -1 means cpu', default=-1)

parser.add_argument('-lrdec', type=float, help='Learning rate for decoder', default=1e-3)
parser.add_argument('-lrenc', type=float, help='Learning rate for inference part', default=None)

parser.add_argument('-n_epoches', type=int, help='Number of epoches', default=200)
parser.add_argument('-train_batch_size', type=int, help='Batch size', default=500)
parser.add_argument('-n_val_samples', type=int, help='How many samples to use on evaluation', default=1)

parser.add_argument('-anneal_cap', type=float, help='Maximal annealing coeff', default=1.)

args = parser.parse_args()


def set_seeds(rand_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def main(args):
    set_seeds(322)
    args = get_args(args)

    if args.data == 'gowalla':
        input_dim = (21208,)

    elif args.data == 'foursquare':
        input_dim = (26150,)

    dataset = Dataset(args)
    # pdb.set_trace()
    layers = [200, 600, dataset.n_items]
    args.z_dim = layers[0]
    args.l2_coeff = 0.
    # with torch.autograd.detect_anomaly():
    if args.model == "MultiVAE":
        from other_models import MultiVAEPythae
        from pythae.models import VAEConfig

        config = VAEConfig(
            latent_dim=args.z_dim,
            input_dim=input_dim
        )

        model = MultiVAEPythae(
            model_config=config,
            encoder=Encoder(dims=layers[::-1]),
            decoder=Decoder(dims=layers)
        ).to(args.device)

    elif args.model == "MultiVAMP":
        from other_models import MultiVAMP
        from pythae.models import VAMPConfig

        config = VAMPConfig(
            latent_dim=args.z_dim,
            input_dim=input_dim,
            n_components=args.n_components
        )

        model = MultiVAMP(model_config=config,
            encoder=Encoder(dims=layers[::-1]),
            decoder=Decoder(dims=layers)
        ).to(args.device)

    elif args.model == "MultiVAELinNF":
        from other_models import MultiVAELinNF
        from pythae.models import VAE_LinNF_Config

        config = VAE_LinNF_Config(
            latent_dim=args.z_dim,
            input_dim=input_dim,
            flows=['Planar', 'Radial', 'Planar', 'Radial', 'Planar']
        )

        model = MultiVAELinNF(model_config=config,
            encoder=Encoder(dims=layers[::-1]),
            decoder=Decoder(dims=layers)
        ).to(args.device)

    elif args.model == "MultiVAEIAF":
        from other_models import MultiVAEIAF
        from pythae.models import VAE_IAF_Config

        config = VAE_IAF_Config(
            latent_dim=args.z_dim,
            input_dim=input_dim,
            n_made_blocks=args.n_made_blocks
        )

        model = MultiVAEIAF(model_config=config,
            encoder=Encoder(dims=layers[::-1]),
            decoder=Decoder(dims=layers)
        ).to(args.device)

    elif args.model == "MultiCIWAE":
        from other_models import MultiCIWAE
        from pythae.models import CIWAEConfig

        config = CIWAEConfig(
            latent_dim=args.z_dim,
            input_dim=input_dim,
            number_samples=args.n_samples,
            beta=args.beta
        )

        model = MultiCIWAE(model_config=config,
            encoder=Encoder(dims=layers[::-1]),
            decoder=Decoder(dims=layers)
        ).to(args.device)

    print(model)


    wandb.init(
        project="recommenders",
        entity="clementchadebec"
    )

    wandb.config.update(
        {
            "model_config": config.to_dict(),
            "args": args
        }
    )

    metric_values = train_pythae_model(model, dataset, args)

    np.savetxt(
        "../logs/pythae_logs/metrics_{}_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}_learnscale_{}.txt".format(
            args.data, args.model, args.K, args.N,
            args.learnable_reverse, args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma,
            args.learnscale),
        np.array(metric_values))

    with open("../logs/log.txt", "a") as myfile:
        myfile.write("!!Success!! \n \n \n \n".format(args))
    print('Success!')


if __name__ == "__main__":
    main(args)
