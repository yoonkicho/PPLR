from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time

from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from pplr import datasets
from pplr.models import resnet50part
from pplr.trainers import PPLRTrainer
from pplr.evaluators import Evaluator, extract_all_features
from pplr.utils.data import IterLoader
from pplr.utils.data import transforms as T
from pplr.utils.data.sampler import RandomMultipleGallerySampler
from pplr.utils.data.preprocessor import Preprocessor
from pplr.utils.logging import Logger
from pplr.utils.faiss_rerank import compute_ranked_list, compute_jaccard_distance

best_mAP = 0


def get_data(name, data_dir):
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                           batch_size=batch_size, num_workers=workers, sampler=sampler,
                           shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def compute_pseudo_labels(features, cluster, k1):
    mat_dist = compute_jaccard_distance(features, k1=k1, k2=6)
    ids = cluster.fit_predict(mat_dist)
    num_ids = len(set(ids)) - (1 if -1 in ids else 0)

    labels = []
    outliers = 0
    for i, id in enumerate(ids):
        if id != -1:
            labels.append(id)
        else:
            labels.append(num_ids + outliers)
            outliers += 1

    return torch.Tensor(labels).long().detach(), num_ids


def compute_cross_agreement(features_g, features_p, k, search_option=0):
    print("Compute cross agreement score...")
    N, D, P = features_p.size()
    score = torch.FloatTensor()
    end = time.time()
    ranked_list_g = compute_ranked_list(features_g, k=k, search_option=search_option, verbose=False)

    for i in range(P):
        ranked_list_p_i = compute_ranked_list(features_p[:, :, i], k=k, search_option=search_option, verbose=False)
        intersect_i = torch.FloatTensor(
            [len(np.intersect1d(ranked_list_g[j], ranked_list_p_i[j])) for j in range(N)])
        union_i = torch.FloatTensor(
            [len(np.union1d(ranked_list_g[j], ranked_list_p_i[j])) for j in range(N)])
        score_i = intersect_i / union_i
        score = torch.cat([score, score_i.unsqueeze(1)], dim=1)

    print("Cross agreement score time cost: {}".format(time.time() - end))
    return score


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    main_worker(args)


def main_worker(args):
    global best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # dataset
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers,
                                     testset=sorted(dataset.train))

    # model
    num_part = args.part
    model = resnet50part(num_parts=args.part, num_classes=3000)
    model.cuda()
    model = nn.DataParallel(model)

    # evaluator
    evaluator = Evaluator(model)

    # optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    score_log = torch.FloatTensor([])
    for epoch in range(args.epochs):
        features_g, features_p, _ = extract_all_features(model, cluster_loader)
        features_g = torch.cat([features_g[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        features_p = torch.cat([features_p[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

        if epoch == 0:
            cluster = DBSCAN(eps=args.eps, min_samples=4, metric='precomputed', n_jobs=8)

        # assign pseudo-labels
        pseudo_labels, num_class = compute_pseudo_labels(features_g, cluster, args.k1)

        # Compute the cross-agreement
        score = compute_cross_agreement(features_g, features_p, k=args.k)
        score_log = torch.cat([score_log, score.unsqueeze(0)], dim=0)

        # generate new dataset with pseudo-labels
        num_outliers = 0
        new_dataset = []

        idxs, pids = [], []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            pid = label.item()
            if pid >= num_class:  # append data except outliers
                num_outliers += 1
            else:
                new_dataset.append((fname, pid, cid))
                idxs.append(i)
                pids.append(pid)

        train_loader = get_train_loader(dataset, args.height, args.width, args.batch_size,
                                        args.workers, args.num_instances, args.iters, trainset=new_dataset)

        # statistics of clusters and un-clustered instances
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'.format(epoch, num_class,
                                                                                           num_outliers))

        # reindex
        idxs, pids = np.asarray(idxs), np.asarray(pids)
        features_g = features_g[idxs, :]
        features_p = features_p[idxs, :, :]
        score = score[idxs, :]

        # compute cluster centroids
        centroids_g, centroids_p = [], []
        for pid in sorted(np.unique(pids)):  # loop all pids
            idxs_p = np.where(pids == pid)[0]
            centroids_g.append(features_g[idxs_p].mean(0))
            centroids_p.append(features_p[idxs_p].mean(0))

        centroids_g = F.normalize(torch.stack(centroids_g), p=2, dim=1)
        model.module.classifier.weight.data[:num_class].copy_(centroids_g)
        for i in range(num_part):
            centroids_p_i = torch.stack(centroids_p)[:, :, i]
            centroids_p_i = F.normalize(centroids_p_i, p=2, dim=1)
            classifier_p_i = getattr(model.module, 'classifier' + str(i))
            classifier_p_i.weight.data[:num_class].copy_(centroids_p_i)

        # training
        trainer = PPLRTrainer(model, score, num_class=num_class, num_part=num_part, beta=args.beta,
                              aals_epoch=args.aals_epoch)

        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader))
        lr_scheduler.step()

        # evaluation
        if ((epoch+1) % args.eval_step == 0) or (epoch == args.epochs-1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), osp.join(args.logs_dir, 'best.pth'))
            print('\n* Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}\n'.format(epoch, mAP, best_mAP))

    torch.save(model.state_dict(), osp.join(args.logs_dir, 'last.pth'))
    np.save(osp.join(args.logs_dir, 'scores.npy'), score_log.numpy())

    # results
    model.load_state_dict(torch.load(osp.join(args.logs_dir, 'best.pth')))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Part-based Pseudo Label Refinement")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-n', '--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/test'))

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)

    # PPLR
    parser.add_argument('--part', type=int, default=3, help="number of part")
    parser.add_argument('--k', type=int, default=20,
                        help="hyperparameter for cross agreement score")
    parser.add_argument('--beta', type=float, default=0.5,
                        help="weighting parameter for part-guided label refinement")
    parser.add_argument('--aals-epoch', type=int, default=5,
                        help="starting epoch for agreement-aware label smoothing")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)

    # cluster
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--eps', type=float, default=0.5,
                        help="distance threshold for DBSCAN")

    main()
