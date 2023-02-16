import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import pdb

def train_byol_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    update_count = 0
    num_epochs = args['n_epoches']
    num_epochs_dec = num_epochs if args['n_epoches_dec'] is None else args['n_epoches_dec']
    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc
    lrdec = args.lrdec
    optimizer_byol = torch.optim.Adam([{'params':model.online_network.parameters()}, {'params':model.predictor.parameters()}], lr=lrenc)
    optimizer_dec = torch.optim.Adam(params=model.decoder.parameters(), lr=lrdec)
    losses = []
    model.initializes_target_network()

    for epoch_counter in range(num_epochs):

        for b_num, batch in enumerate(dataset.next_train_batch()):

            #batch_view_1 = batch_view_1.to(self.device)
            #batch_view_2 = batch_view_2.to(self.device)

            loss = model.update(batch)

            optimizer_byol.zero_grad()
            loss.backward()
            optimizer_byol.step()

            model._update_target_network_parameters()  # update the key encoder
            update_count += 1
        print('Epoch ', epoch_counter, 'Loss ', loss)
            
    for epoch in tqdm(range(num_epochs_dec)):
        model.train()
        for b_num, batch in enumerate(dataset.next_train_batch()):
            loss, _ = model.step_decoder(batch)
            loss.backward()
            optimizer_dec.step()
            optimizer_dec.zero_grad()
        
        model.eval()
        with torch.no_grad():
            metric_dist = []
            for bnum, batch_val in enumerate(dataset.next_val_batch()):
                reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
                is_training_ph = int(args.n_val_samples > 1)
                _, pred_val = model.step_decoder(reshaped_batch)
                pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
                X = batch_val[0].cpu().detach().numpy()
                pred_val = pred_val.cpu().detach().numpy()
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                metric_dist.append(args.metric(pred_val, batch_val[1]))

            metric_dist = np.concatenate(metric_dist)
            current_metric = metric_dist.mean()
            metric_vad.append(current_metric)
            
            if current_metric > best_metric:
                best_metric = current_metric
            # update the best model (if necessary)
            if epoch % args.print_info_ == 0:
                print('Best NDCG:', best_metric)
                print('Current NDCG:', current_metric)
        
    return metric_vad, model
    # save checkpoints


def train_simclr_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    update_count = 0
    num_epochs = args['n_epoches']
    num_epochs_dec = num_epochs if args['n_epoches_dec'] is None else args['n_epoches_dec']
    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc
    lrdec = args.lrdec
    optimizer_enc = torch.optim.Adam(params=model.encoder.parameters(), lr=lrenc)
    optimizer_dec = torch.optim.Adam(params=model.decoder.parameters(), lr=lrdec)

    for epoch in tqdm(range(num_epochs)):
        for b_num, batch in enumerate(dataset.next_train_batch()):
            loss = model.step_encoder(batch)
            loss.backward()
            optimizer_enc.step()
            optimizer_enc.zero_grad()
    
    for epoch in tqdm(range(num_epochs_dec)):
        model.train()
        for b_num, batch in enumerate(dataset.next_train_batch()):
            loss, _ = model.step_decoder(batch)
            loss.backward()
            optimizer_dec.step()
            optimizer_dec.zero_grad()
        
        model.eval()
        with torch.no_grad():
            metric_dist = []
            for bnum, batch_val in enumerate(dataset.next_val_batch()):
                reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
                is_training_ph = int(args.n_val_samples > 1)
                _, pred_val = model.step_decoder(reshaped_batch)
                pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
                X = batch_val[0].cpu().detach().numpy()
                pred_val = pred_val.cpu().detach().numpy()
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                metric_dist.append(args.metric(pred_val, batch_val[1]))

            metric_dist = np.concatenate(metric_dist)
            current_metric = metric_dist.mean()
            metric_vad.append(current_metric)
            if current_metric > best_metric:
                best_metric = current_metric
            # update the best model (if necessary)
            if epoch % args.print_info_ == 0:
                print('Best NDCG:', best_metric)
                print('Current NDCG:', current_metric)
    return metric_vad, model

            
    

def train_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    print_info_ = args.print_info_
    update_count = 0.0

    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc

    if args.model == 'MultiDAE':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrdec, weight_decay=args.l2_coeff)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.decoder.parameters(), 'lr': args.lrdec},
            {'params': model.encoder.parameters()}
        ],
            lr=lrenc, weight_decay=args.l2_coeff)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            logits, KL = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            neg_ll = -torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))

            # compute objective
            neg_ELBO = neg_ll + anneal * KL
            neg_ELBO.backward()

            optimizer.step()
            optimizer.zero_grad()

            if (bnum % 100 == 0) and (epoch % print_info_ == 0):
                print(neg_ELBO.cpu().detach().numpy())

            update_count += 1

        # compute validation NDCG
        model.eval()
        with torch.no_grad():
            metric_dist = []
            for bnum, batch_val in enumerate(dataset.next_val_batch()):
                reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
                is_training_ph = int(args.n_val_samples > 1)
                pred_val, _ = model(reshaped_batch, is_training_ph=is_training_ph)
                pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
                X = batch_val[0].cpu().detach().numpy()
                pred_val = pred_val.cpu().detach().numpy()
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                metric_dist.append(args.metric(pred_val, batch_val[1]))

            metric_dist = np.concatenate(metric_dist)
            current_metric = metric_dist.mean()
            metric_vad.append(current_metric)

            # update the best model (if necessary)
            if current_metric > best_metric:
                torch.save(model,
                           '../models/best_model_{}_data_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}.pt'.format(
                               args.model, args.data, args.K,
                               args.N,
                               args.learnable_reverse,
                               args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma))
                best_metric = current_metric
            if epoch % print_info_ == 0:
                print('Best NDCG:', best_metric)
                print('Current NDCG:', current_metric)
    return metric_vad


def train_iw_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    print_info_ = args.print_info_
    update_count = 0.0

    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc

    if args.model == 'MultiDAE':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrdec, weight_decay=args.l2_coeff)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.decoder.parameters(), 'lr': args.lrdec},
            {'params': model.encoder.parameters()}
        ],
            lr=lrenc, weight_decay=args.l2_coeff)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            loss, elbo = model.loss_function(batch_train, anneal)

            # loglikelihood part
            #log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            #neg_ll = -torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))

            # compute objective
            #neg_ELBO = neg_ll + anneal * KL
            #neg_ELBO.backward()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (bnum % 100 == 0) and (epoch % print_info_ == 0):
                print(elbo.cpu().detach().numpy())

            update_count += 1

        # compute validation NDCG
        model.eval()
        with torch.no_grad():
            metric_dist = []
            for bnum, batch_val in enumerate(dataset.next_val_batch()):
                #pdb.set_trace()
                reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
                is_training_ph = int(args.n_val_samples > 1)
                pred_val, _, _, _, _ = model(reshaped_batch, is_training_ph=is_training_ph)
                pred_val = pred_val.mean(0)
                pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
                X = batch_val[0].cpu().detach().numpy()
                pred_val = pred_val.cpu().detach().numpy()
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                metric_dist.append(args.metric(pred_val, batch_val[1]))

            metric_dist = np.concatenate(metric_dist)
            current_metric = metric_dist.mean()
            metric_vad.append(current_metric)

            # update the best model (if necessary)
            if current_metric > best_metric:
                torch.save(model,
                           '../models/best_model_{}_data_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}.pt'.format(
                               args.model, args.data, args.K,
                               args.N,
                               args.learnable_reverse,
                               args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma))
                best_metric = current_metric
            if epoch % print_info_ == 0:
                print('Best NDCG:', best_metric)
                print('Current NDCG:', current_metric)
    return metric_vad


def train_met_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    print_info_ = args.print_info_
    update_count = 0.0

    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc

    if not args.learntransitions:
        for p in model.transitions.parameters():
            p.requires_grad_(False)
    else:
        for k in range(len(model.transitions)):
            model.transitions[k].alpha_logit.requires_grad_(False)

    if args.learnable_reverse:
        optimizer = torch.optim.Adam([
            {'params': model.target.decoder.parameters(), 'lr': args.lrdec},
            {'params': model.encoder.parameters()},
            {'params': model.transitions.parameters()},
            {'params': model.reverse_kernel.parameters()},
            {'params': model.momentum_scale},
        ],
            lr=lrenc, weight_decay=args.l2_coeff)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.target.decoder.parameters(), 'lr': args.lrdec},
            {'params': model.encoder.parameters()},
            {'params': model.transitions.parameters()},
            {'params': model.momentum_scale},
        ],
            lr=lrenc, weight_decay=args.l2_coeff)
    scheduler = MultiStepLR(optimizer, [20, 50, 75, 100, 150, 200], gamma=0.3)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            logits, log_q, log_aux, log_priors, log_r, sum_log_alpha, directions = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            log_likelihood = torch.sum(log_softmax_var * batch_train, 1).mean()
            # compute objective
            KLD = log_q.mean() + log_aux.mean() - log_r.mean() - log_priors.mean()
            elbo_full = log_likelihood - anneal * KLD

            grad_elbo = elbo_full + elbo_full.detach() * torch.mean(sum_log_alpha)
            (-grad_elbo).backward()

            optimizer.step()
            optimizer.zero_grad()

            if (bnum % 200 == 0) and (epoch % print_info_ == 0):
                print('Current anneal coeff:', anneal)
                if args.learnscale:
                    print('Min scale', torch.exp(model.momentum_scale.detach()).min().item(), 'Max scale',
                          torch.exp(model.momentum_scale.detach()).max().item())
                print(elbo_full.cpu().detach().mean().numpy())
                for k in range(args.K):
                    print('k =', k)
                    print('0: {} and for +1: {}'.format((directions[:, k] == 0.).to(float).mean(),
                                                        (directions[:, k] == 1.).to(float).mean()))
                    print('autoreg:', torch.sigmoid(model.transitions[k].alpha_logit.detach()).item())
                    print('stepsize', torch.exp(model.transitions[k].gamma.detach()).item())
                    print('-' * 100)

            update_count += 1
        if np.isnan(elbo_full.cpu().detach().mean().numpy()):
            break

        if (args.data in ['ml20m', 'gowalla', 'foursquare']):  # and not args.annealing:
            scheduler.step()

        if epoch % print_info_ == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

        # compute validation NDCG
        model.eval()
        metric_dist = []
        for bnum, batch_val in enumerate(dataset.next_val_batch()):
            reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
            is_training_ph = int(args.n_val_samples > 1)
            pred_val, _, _, _, _, _, _ = model(reshaped_batch, is_training_ph=is_training_ph)
            pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
            X = batch_val[0].cpu().detach().numpy()
            pred_val = pred_val.cpu().detach().numpy()
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            metric_dist.append(args.metric(pred_val, batch_val[1]))

        metric_dist = np.concatenate(metric_dist)
        current_metric = metric_dist.mean()
        metric_vad.append(current_metric)

        # update the best model (if necessary)
        if current_metric > best_metric:
            torch.save(model,
                       '../models/best_model_{}_data_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}_learnscale_{}.pt'.format(
                           args.model, args.data, args.K,
                           args.N,
                           args.learnable_reverse,
                           args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma, args.learnscale))
            best_metric = current_metric
        if epoch % print_info_ == 0:
            print('Best NDCG:', best_metric)
            print('Current NDCG:', current_metric)
    return metric_vad



def train_pythae_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    print_info_ = args.print_info_
    update_count = 0.0

    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': args.lrdec},
        {'params': model.encoder.parameters()}
    ],
        lr=lrenc, weight_decay=args.l2_coeff)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap


            loss, elbo = model.loss_function(batch_train, anneal)

            # loglikelihood part
            #log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            #neg_ll = -torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))

            # compute objective
            #neg_ELBO = neg_ll + anneal * KL
            #neg_ELBO.backward()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (bnum % 100 == 0) and (epoch % print_info_ == 0):
                print(elbo.cpu().detach().numpy())

            update_count += 1

        # compute validation NDCG
        model.eval()
        with torch.no_grad():
            metric_dist = []
            for bnum, batch_val in enumerate(dataset.next_val_batch()):
                #pdb.set_trace()
                reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
                is_training_ph = int(args.n_val_samples > 1)
                pred_val, _, _, _, _ = model(reshaped_batch, is_training_ph=is_training_ph)
                pred_val = pred_val.mean(0)
                pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
                X = batch_val[0].cpu().detach().numpy()
                pred_val = pred_val.cpu().detach().numpy()
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                metric_dist.append(args.metric(pred_val, batch_val[1]))

            metric_dist = np.concatenate(metric_dist)
            current_metric = metric_dist.mean()
            metric_vad.append(current_metric)

            # update the best model (if necessary)
            if current_metric > best_metric:
                torch.save(model,
                           '../models/best_model_{}_data_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}.pt'.format(
                               args.model, args.data, args.K,
                               args.N,
                               args.learnable_reverse,
                               args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma))
                best_metric = current_metric
            if epoch % print_info_ == 0:
                print('Best NDCG:', best_metric)
                print('Current NDCG:', current_metric)
    return metric_vad