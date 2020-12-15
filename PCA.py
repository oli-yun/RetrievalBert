from parameters import parse
import torch
from models import PreTrainModel, UpdateKNNAdaptiveConcat
from datasets import load_tokenized_dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from knn import KNNDstore

args = parse()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# modify
# train_dataset = load_tokenized_dataset(args.tokenized_data_dir + 'train')
train_dataset = load_tokenized_dataset(args.tokenized_data_dir + 'test')
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

pretrain_model = PreTrainModel(args.pretrain_model_name, num_labels=args.num_labels, only_return_hidden_states=True).to(
    device)
knn_embedding_model = PreTrainModel(args.pretrain_model_name, num_labels=args.num_labels, only_return_hidden_states=True)\
    .to(device)
knn_embedding_model.load_state_dict(torch.load(args.model_dir + f"{args.pretrain_model_name}.pt"))
knn_store = KNNDstore(args)
knn_store.read_dstore('./dstore/finetune/keys.npy', './dstore/finetune/vals.npy')
knn_store.read_index('./faiss/finetune/index')
model = UpdateKNNAdaptiveConcat(pretrain_model, knn_store, args.num_labels, args.k, args.temperature,
                                train_dataset, knn_embedding_model=knn_embedding_model)
model_path = args.model_dir + 'fine_tune_with_knn(fixed_knn).pt'
model.load_state_dict(torch.load(model_path))
# modify

keys = np.zeros((1821, 768), dtype=np.float32)  # 67349/1821
vals = np.zeros((1821, 1), dtype=np.int)  # 67349/1821

for i, (data, target) in enumerate(tqdm(train_dataloader)):
    b_data = tuple(d.to(device) for d in data)
    b_labels = target.unsqueeze(dim=1)

    with torch.no_grad():
        sentence_embedding = model.pretrain_model(*b_data)
    keys[i * args.batch_size: min(args.dstore_size, (i + 1) * args.batch_size)] = \
        sentence_embedding.cpu().numpy()
    vals[i * args.batch_size: min(args.dstore_size, (i + 1) * args.batch_size)] = b_labels.numpy()

ipca = IncrementalPCA(n_components=3, batch_size=10)
X_ipca = ipca.fit_transform(keys)

fig = plt.figure(1, figsize=(12, 9))
plt.clf()
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_ipca[:, 0], X_ipca[:, 1], X_ipca[:, 2], c=vals[:, 0],
           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()
