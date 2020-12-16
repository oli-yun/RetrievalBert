import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification


class PreTrainModel(nn.Module):
    def __init__(self, pretrain_model_name, num_labels, only_return_hidden_states=False, only_return_logits=False):
        super(PreTrainModel, self).__init__()
        self.only_return_hidden_states = only_return_hidden_states
        self.only_return_logits = only_return_logits
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_name, num_labels=num_labels)
        self.training = True
        if self.only_return_hidden_states:
            for param in self.model.classifier.parameters():
                param.requires_grad = False

    def forward(self, x, x_mask):
        outputs = self.model(x, x_mask, return_dict=True, output_hidden_states=True)
        if self.only_return_hidden_states:
            return outputs.hidden_states[-1][:, 0, :]
        elif self.only_return_logits:
            return outputs.logits
        else:
            return outputs.logits, outputs.hidden_states[-1][:, 0, :]

    def start_train(self):
        self.training = True

    def start_test(self):
        self.training = False


class NoArgKNN(nn.Module):
    """
    Base model.
    """

    def __init__(self, pretrain_model, knn_store):
        super(NoArgKNN, self).__init__()
        self.embedding_model = pretrain_model
        self.knn_store = knn_store
        self.training = True

    def get_knn_prob(self, queries, k, temperature):
        dists, knns = self.knn_store.get_knns(queries, k, True)
        knn_prob = self.knn_store.get_knn_prob(dists, knns, temperature)
        return knn_prob

    def get_model_prob(self, x, x_mask):
        self.embedding_model.eval()
        with torch.no_grad():
            model_prob, queries = self.embedding_model(x, x_mask)
            model_prob = torch.nn.functional.softmax(model_prob, dim=-1)
        return model_prob, queries

    def forward(self, x, x_mask, k, temperature, another_hyper):
        raise NotImplementedError

    def start_train(self):
        self.training = True

    def start_test(self):
        self.training = False


class OnlyKNN(NoArgKNN):
    """
    Fine tune pretrain model for the sequence classification task.
    Use the [CLS] token as embedding for kNN.
    """
    def __init__(self, pretrain_model, knn_store):
        super().__init__(pretrain_model, knn_store)

    def forward(self, x, x_mask, k, temperature, another_hyper=None):
        _, queries = self.get_model_prob(x, x_mask)
        knn_prob = self.get_knn_prob(queries, k, temperature)
        return knn_prob


class KNNBackoff(NoArgKNN):
    """
    The same as FineTuneKNN
    But use KNN predictions as backoff.
    """
    def __init__(self, pretrain_model, knn_store):
        super().__init__(pretrain_model, knn_store)

    def forward(self, x, x_mask, k, temperature, threshold):
        model_prob, queries = self.get_model_prob(x, x_mask)
        model_pred = torch.max(model_prob, dim=-1)[0]
        knn_prob = self.get_knn_prob(queries, k, temperature)
        final_prob = torch.where(model_pred.unsqueeze(dim=-1)
                                 .repeat(1, model_prob.size()[-1]).cpu() > threshold,
                                 model_prob.cpu(), knn_prob.cpu())
        return final_prob


class KNNStaticConcat(NoArgKNN):
    """
    The same as FineTuneKNN
    But concat KNN predictions with pretrain model predictions.
    """
    def __init__(self, pretrain_model, knn_store):
        super().__init__(pretrain_model, knn_store)

    def forward(self, x, x_mask, k, temperature, knn_weight):
        model_prob, queries = self.get_model_prob(x, x_mask)
        knn_prob = self.get_knn_prob(queries, k, temperature)
        final_prob = knn_prob.mul(knn_weight).cpu() + model_prob.mul(1 - knn_weight).cpu()
        return final_prob


class UpdateKNNAdaptiveConcat(nn.Module):
    """
    update knn representation per epoch and use adaptive weight.
    when training, retrieval use the representation in datastore, the weight computation use the updated representation.
    """

    def __init__(self, pretrain_model, knn_store, num_labels, k, temperature, train_datasets,
                 knn_embedding_model=None, fixed_pretrain=False, fixed_knn=False, only_knn=False):
        super().__init__()
        self.pretrain_model = pretrain_model
        if fixed_pretrain:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        self.training = True

        self.knn_store = knn_store
        self.k = k
        self.temperature = temperature
        self.fixed_knn = fixed_knn
        self.only_knn = only_knn

        self.train_datasets = train_datasets
        self.knn_embedding_model = knn_embedding_model

        self.text_dense = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.Dropout(p=0.1, inplace=False)
        )
        weight_init(self.text_dense[0])
        self.neighbor_dense = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.Dropout(p=0.1, inplace=False)
        )
        weight_init(self.neighbor_dense[0])
        self.classifier = nn.Linear(768, num_labels, bias=True)
        weight_init(self.classifier)
        self.get_weight = nn.Linear(768 * 2, 1, bias=True)
        weight_init(self.get_weight)

    def forward(self, x_idx, x, x_mask):
        text_rep = self.pretrain_model(x, x_mask)
        neighbors = None

        if self.training or self.fixed_knn:
            if self.training:
                dists, knns = self.knn_store.get_knns(self.knn_store.keys[x_idx.cpu().numpy()],
                                                      self.k + 1, change_type=False)
                dists, knns = dists[:, 1:], knns[:, 1:]
            else:
                dists, knns = self.knn_store.get_knns(self.knn_embedding_model(x, x_mask), self.k, change_type=True)
            knn_prob = self.knn_store.get_knn_prob(dists, knns, self.temperature)
            if not self.only_knn:
                neighbors = []
                for i in range(self.k):
                    train_neighbors, _ = self.train_datasets[knns[:, i]]
                    neighbor = self.pretrain_model(train_neighbors[0].type_as(x), train_neighbors[1].type_as(x_mask))
                    neighbors.append(neighbor.unsqueeze(dim=1))
                neighbors = torch.cat(neighbors, dim=1)
        else:
            dists, knns = self.knn_store.get_knns(text_rep, self.k, change_type=True)
            knn_prob = self.knn_store.get_knn_prob(dists, knns, self.temperature)
            if not self.only_knn:
                neighbors = torch.from_numpy(self.knn_store.keys[knns]).type_as(text_rep)

        if self.only_knn:
            return torch.log(knn_prob)
        else:
            dists = torch.from_numpy(-1 * dists)
            neighbor_probs = torch.nn.functional.softmax(dists / self.temperature, dim=-1).type_as(text_rep)
            neighbor_probs = neighbor_probs.unsqueeze(dim=-1).repeat(1, 1, 768)
            neighbor_rep = torch.sum(torch.mul(neighbor_probs, neighbors), dim=1)
            neighbor_rep = self.neighbor_dense(neighbor_rep)

            text_rep = self.text_dense(text_rep)
            model_prob = nn.functional.softmax(self.classifier(text_rep), dim=-1)

            p_knn = torch.sigmoid(self.get_weight(torch.cat([text_rep, neighbor_rep], dim=-1)))
            # print(p_knn.data)
            final_prob = torch.log(p_knn * knn_prob.type_as(model_prob) + (1 - p_knn) * model_prob)
            return final_prob
            # return model_prob
            # return knn_prob.type_as(model_prob)

    def start_train(self):
        self.training = True

    def start_test(self):
        self.training = False


def weight_init(module):
    module.weight.data.normal_(mean=0.0, std=0.02)
    module.bias.data.zero_()

