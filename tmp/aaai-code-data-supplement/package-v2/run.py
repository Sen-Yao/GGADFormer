"""Single-seed training and fixed-endpoint evaluation for VecGAD."""

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import _LRScheduler

from controls import CONTROL_NAMES
from data import DATASETS, load_dataset
from tokenization import incremental_tokenization
from vecgad import VecGAD


class PolynomialDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_updates, total_updates, peak_lr, end_lr,
                 power=1.0, last_epoch=-1):
        self.warmup_updates = warmup_updates
        self.total_updates = total_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            learning_rate = self.peak_lr * self._step_count / float(self.warmup_updates)
        elif self._step_count >= self.total_updates:
            learning_rate = self.end_lr
        else:
            remaining = 1.0 - (
                (self._step_count - self.warmup_updates)
                / (self.total_updates - self.warmup_updates)
            )
            learning_rate = (
                (self.peak_lr - self.end_lr) * remaining ** self.power
                + self.end_lr
            )
        return [learning_rate for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        raise NotImplementedError


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_index):
    if torch.cuda.is_available() and device_index >= 0:
        return torch.device("cuda:{}".format(device_index))
    return torch.device("cpu")


def evaluate(model, loader, labels, test_indices, device):
    model.eval()
    batches = []
    with torch.no_grad():
        for (tokens,) in loader:
            batches.append(model.score(tokens.to(device)).cpu())
    scores = torch.cat(batches).numpy()
    targets = labels[test_indices].numpy()
    return (
        roc_auc_score(targets, scores),
        average_precision_score(targets, scores, average="macro", pos_label=1),
    )


def train(args):
    set_seed(args.seed)
    device = select_device(args.device)
    dataset = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        train_rate=args.train_rate,
        val_rate=0.1,
        data_split_seed=args.data_split_seed,
    )
    tokens = incremental_tokenization(
        dataset["features"],
        dataset["adjacency"],
        args.pp_k,
        args.progregate_alpha,
    )
    model = VecGAD(tokens.shape[-1], args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay
    )
    scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        total_updates=args.num_epoch,
        peak_lr=args.peak_lr,
        end_lr=args.end_lr,
    )
    binary_loss = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor([1.0], device=device)
    )

    n_nodes = tokens.shape[0]
    normal_for_train = dataset["normal_for_train"]
    if not normal_for_train or len(normal_for_train) == n_nodes:
        raise ValueError("the training split must contain labeled normals and an unlabeled pool")
    all_indices = torch.arange(n_nodes)
    train_dataset = torch_data.TensorDataset(tokens, all_indices)
    unknown = sorted(set(range(n_nodes)) - set(normal_for_train))
    weights = torch.zeros(n_nodes)
    weights[normal_for_train] = 1.0 / len(normal_for_train)
    weights[unknown] = 1.0 / len(unknown)
    sampler = torch_data.WeightedRandomSampler(weights, num_samples=n_nodes, replacement=True)
    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )
    normal_global = torch.tensor(normal_for_train, dtype=torch.long, device=device)

    for _epoch in range(args.num_epoch + 1):
        model.train()
        for batch_tokens, batch_global in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_global = batch_global.to(device)
            local_normals = torch.nonzero(
                torch.isin(batch_global, normal_global), as_tuple=False
            ).squeeze(-1)
            optimizer.zero_grad()
            logits, reconstruction_loss, hsc_loss, source_count = model.training_objectives(
                batch_tokens, local_normals
            )
            targets = torch.cat(
                (
                    torch.zeros(local_normals.numel(), device=device),
                    torch.ones(source_count, device=device),
                )
            ).view(1, -1, 1)
            classification_loss = binary_loss(logits, targets).mean()
            loss = (
                args.bce_loss_weight * classification_loss
                + args.rec_loss_weight * reconstruction_loss
                + args.ring_loss_weight * hsc_loss
            )
            loss.backward()
            optimizer.step()
        scheduler.step()

    test_tokens = torch_data.TensorDataset(tokens[dataset["idx_test"]])
    test_loader = torch_data.DataLoader(
        test_tokens, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    auroc, auprc = evaluate(
        model, test_loader, dataset["labels"], dataset["idx_test"], device
    )
    return auroc, auprc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate one fixed-seed VecGAD configuration."
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_split_seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train_rate", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_epoch", type=int, required=True)
    parser.add_argument("--peak_lr", type=float, required=True)
    parser.add_argument("--end_lr", type=float, required=True)
    parser.add_argument("--warmup_updates", type=int, required=True)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pp_k", type=int, required=True)
    parser.add_argument("--progregate_alpha", type=float, required=True)
    parser.add_argument("--sample_rate", type=float, default=0.15)
    parser.add_argument("--outlier_beta", type=float, default=0.3)
    parser.add_argument("--ring_R_min", type=float, default=0.3)
    parser.add_argument("--ring_R_max", type=float, default=1.0)
    parser.add_argument("--lambda_rec_tok", type=float, default=1.0)
    parser.add_argument("--lambda_rec_emb", type=float, default=0.1)
    parser.add_argument("--bce_loss_weight", type=float, default=1.0)
    parser.add_argument("--rec_loss_weight", type=float, default=1.0)
    parser.add_argument("--ring_loss_weight", type=float, default=1.0)
    parser.add_argument("--control", choices=CONTROL_NAMES, default="full")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--attention_dropout", type=float, default=0.4)
    return parser


def main():
    args = build_parser().parse_args()
    if args.dataset in ("reddit", "photo"):
        args.noise_mean = 0.02
        args.noise_std = 0.01
    else:
        args.noise_mean = 0.0
        args.noise_std = 0.0
    start = time.perf_counter()
    auroc, auprc = train(args)
    runtime = time.perf_counter() - start
    print(
        "final_epoch={} AUROC={:.6f} AUPRC={:.6f} runtime_seconds={:.2f}".format(
            args.num_epoch, auroc, auprc, runtime
        )
    )


if __name__ == "__main__":
    main()
