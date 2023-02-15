This the companion repository for the paper: **Metropolized Variational Autoencoders with Applications to Collaborative Filtering**

------

All the experiments which appear in the paper can be run via the `runs.sh` script, located in `./src/` folder. You can run them from the folder, using the folliwing bash command:

```bash
sh runs.sh
```

Inside the bash file, you can see the following (for example):

```bash
python3 main.py -data ml20m -model Multi_our_VAE -K 2 -N 1 -annealing False -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learnscale True -gpu 0 -train_batch_size 500 -n_epoches 300
```

Some of the fields are self-explaining. Nevertheless, some clarifications are required:

- **K** -- a number of Metropolis-Hastings transitions to use.
- **N** -- a number of leapfrog iterations within the HMC algorithm.
- **n_val_samples** -- how many samples (per one data object) from latent space to use to compute validation score.
- **learnscale** -- whether train scale coefficients for momentum or not.
- **batch_size_train** -- batch size for training.
- **gpu** -- define an index of your device (-1 for CPU).

Some extra arguments could be found in `/src/args.py`.

---

To download data and the best models we received, run `download.sh`, using the following command:

```bash
sh download.sh
```

---



RNVP-implementation we used:
https://github.com/senya-ashukha/real-nvp-pytorch