import faiss
import torch
import numpy as np
from loguru import logger


class KNNDstore(object):
    def __init__(self, args):
        np.random.seed(args.seed)
        self.dimension = 768
        self.dstore_size = args.dstore_size
        self.classes_num = args.num_labels
        self.nprobe = args.nprobe
        self.clusters_num = args.clusters_num

        self.faiss_path = args.faiss_dir + f'index'
        self.keys_path = args.dstore_dir + f'keys.npy'
        self.vals_path = args.dstore_dir + f'vals.npy'

        self.keys = np.memmap(self.keys_path, dtype=np.float32, mode='w+', shape=(self.dstore_size, 768))
        self.vals = np.memmap(self.vals_path, dtype=np.int, mode='w+', shape=(self.dstore_size, 1))
        self.valid_dstore = True

        self.index = self.setup_faiss()

    def setup_faiss(self):
        logger.info('Initialize faiss index')
        # index = faiss.IndexFlatL2(768)
        quantizer = faiss.IndexFlatL2(768)
        index = faiss.IndexIVFPQ(quantizer, 768, self.clusters_num, 24, 8)
        index = faiss.IndexIDMap2(index)
        return index

    def write_dstore(self, embedding, labels, start_idx, data_size):
        self.keys[start_idx: min(self.dstore_size, start_idx + data_size)] = embedding.cpu().numpy()
        self.vals[start_idx: min(self.dstore_size, start_idx + data_size)] = labels.cpu().numpy()
        if start_idx + data_size >= self.dstore_size:
            self.valid_dstore = False

    def read_dstore(self, keys_path, vals_path):
        self.keys = np.memmap(keys_path, dtype=np.float32, mode='r', shape=(self.dstore_size, 768))
        self.vals = np.memmap(vals_path, dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.valid_dstore = False

    def add_index(self):
        if self.valid_dstore:
            message = 'Cannot build the index without the data.'
            logger.error(message)
            raise ValueError(message)

        index_size = self.index.ntotal
        if index_size > 0:
            logger.info('Clear faiss index.')
            self.index.remove_ids(np.arange(index_size, dtype='int64'))

        logger.info('Train keys for faiss.')
        random_sample = np.random.choice(np.arange(self.keys.shape[0]), size=self.keys.shape[0], replace=False)
        self.index.train(self.keys[random_sample].astype(np.float32))

        logger.info('Add keys.')
        self.index.add_with_ids(self.keys.astype(np.float32), np.arange(self.dstore_size, dtype='int64'))
        self.index.nprobe = self.nprobe
        faiss.write_index(self.index, self.faiss_path)

    def read_index(self, faiss_path):
        self.index = faiss.read_index(faiss_path, faiss.IO_FLAG_ONDISK_SAME_DIR)

    def get_knns(self, queries, k, change_type=False):
        if change_type:
            queries = queries.detach().cpu().float().numpy()
        dists, knns = self.index.search(queries, k)
        return dists, knns

    def get_knn_prob(self, dists, knns, temperature):
        probs = torch.nn.functional.softmax(torch.from_numpy(-1 * dists) / temperature, dim=-1)
        probs = probs.unsqueeze(dim=-1).repeat(1, 1, self.classes_num)

        vals = torch.from_numpy(self.vals[knns]).squeeze(dim=-1)
        vals = torch.nn.functional.one_hot(vals, num_classes=self.classes_num)

        knn_prob = probs.mul(vals)
        knn_prob = torch.sum(knn_prob, dim=1)
        return knn_prob

    def save_best(self):
        keys = np.memmap(self.keys_path + '.best', dtype=np.float32, mode='w+', shape=(self.dstore_size, 768))
        vals = np.memmap(self.vals_path + '.best', dtype=np.int, mode='w+', shape=(self.dstore_size, 1))
        keys[:] = self.keys[:]
        vals[:] = self.vals[:]
        faiss.write_index(self.index, self.faiss_path + '.best')

    def load_best(self):
        self.keys = np.memmap(self.keys_path + '.best', dtype=np.float32, mode='r',
                              shape=(self.dstore_size, 768))
        self.vals = np.memmap(self.vals_path + '.best', dtype=np.int, mode='r',
                              shape=(self.dstore_size, 1))
        self.valid_dstore = False
        self.index = faiss.read_index(self.faiss_path + '.best', faiss.IO_FLAG_ONDISK_SAME_DIR)
