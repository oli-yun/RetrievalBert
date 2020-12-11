import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class SST2Dataset(Dataset):
    def __init__(self, data, label, tokenizer, return_idx=False):
        encoded_inputs = tokenizer(data, padding='max_length', truncation=True, max_length=66,
                                   return_attention_mask=True, return_tensors='pt')
        self.input_ids = encoded_inputs['input_ids']
        self.masks = encoded_inputs['attention_mask']
        self.labels = torch.tensor(label)
        self.len = len(label)
        self.return_idx = return_idx

    def __getitem__(self, item):
        if self.return_idx:
            return (torch.tensor(item), self.input_ids[item], self.masks[item]), self.labels[item]
        else:
            return (self.input_ids[item], self.masks[item]), self.labels[item]

    def __len__(self):
        return self.len


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def load_sst2(data_path):
    all_data = []
    all_label = []
    with open(data_path, 'r') as f:
        for record in f.readlines()[1:]:
            records = record.strip().split('\t')
            all_data.append(records[0])
            all_label.append(int(records[1]))
    return all_data, all_label


def save_tokenized_dataset(save_path, obj):
    with open(save_path, 'wb') as f:
        torch.save(obj, f)


def load_tokenized_dataset(save_path):
    with open(save_path, 'rb') as f:
        dataset = torch.load(f)

    return dataset


def save_pkl(obj, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)
