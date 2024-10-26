from torch.utils.data import Dataset
import torch
import os
class IMDBDataset(Dataset):
    def __init__(self, file_path, vocab: list, tokenizer, max_length = 512):
        self.file_path = file_path
        self.reviews = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = vocab
        self._load_data()


    def _load_data(self):
        pos_dir = os.path.join(self.file_path, 'pos')
        for filename in os.listdir(pos_dir):
            if filename.endswith('txt'):
                text = open(os.path.join(pos_dir, filename), 'r').read().strip()
                self.reviews.append(text)
                self.labels.append(1)

        neg_dir = os.path.join(self.file_path, 'neg')
        for filename in os.listdir(neg_dir):
            if filename.endswith('txt'):
                text = open(os.path.join(neg_dir, filename), 'r').read().strip()
                self.reviews.append(text)
                self.labels.append(0)

    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        text, label = self.reviews[idx], self.labels[idx]
        tokens = self.tokenizer(text)
        vocab = self.vocab.tolist()
        indices = [vocab.index(token) if token in self.vocab else vocab.index('<unk>') for token in tokens]
        # length = min(len(indices), self.max_length)
        #check length:
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            # pad with <pad>
            diff = self.max_length - len(indices)
            indices =  indices + ([vocab.index('<pad>')] * diff)


        return (torch.tensor(indices), torch.tensor(label, dtype=int))