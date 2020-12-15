import os
import torch
import random
import numpy as np
from loguru import logger
from itertools import combinations
from transformers import AdamW, get_linear_schedule_with_warmup


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, output_path, patience=7, verbose=True, delta=0, save_knn=False, compare_loss=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.val_loss_min = float('inf')
        self.delta = delta
        self.output_path = output_path
        self.save_knn = save_knn
        self.compare_loss = compare_loss

    def __call__(self, value, model):
        score = value

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(value, model)
        elif ((not self.compare_loss) and score <= self.best_score + self.delta) or (
                self.compare_loss and score >= self.best_score + self.delta):
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(value, model)
            self.counter = 0

    def save_checkpoint(self, value, model):
        # def save_checkpoint(self, val_acc, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            message = f'Validation loss decreased ({self.val_loss_min:.6f} --> {value:.6f}).  Saving model ...' \
                if self.compare_loss else \
                f'Validation accuracy increased ({self.val_acc_max:.6f} --> {value:.6f}).  Saving model ...'
            logger.info(message)
        torch.save(model.state_dict(), self.output_path)

        if self.save_knn:
            model.knn_store.save_best()

        if self.compare_loss:
            self.val_loss_min = value
        else:
            self.val_acc_max = value


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=False):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + \
                      vectors.pow(2).sum(dim=1).view(1, -1) + \
                      vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def SemihardNegativeTripletSelector(margin, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=margin, negative_selection_fn=lambda x: semihard_negative(x, margin), cpu=cpu)


def check_dir(args):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.dstore_dir):
        os.mkdir(args.dstore_dir)
    if not os.path.exists(args.dstore_dir + 'finetune/'):
        os.mkdir(args.dstore_dir + 'finetune')
    if not os.path.exists(args.faiss_dir):
        os.mkdir(args.faiss_dir)
    if not os.path.exists(args.tokenized_data_dir):
        os.mkdir(args.tokenized_data_dir)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_optimizer(args, model, train_dataloader):
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.1)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.06 * total_steps, num_training_steps=total_steps)

    return optimizer, scheduler
