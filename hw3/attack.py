import json
import torch
from tqdm import tqdm
import numpy as np
class SubstitutionAttack():
    def __init__(self,vocab: np.ndarray, embedding_matrix, precomputed_bounds = None):
        self.json_file = "counterfitted_neighbors.json"
        self.populate_neighbors()
        self.vocab = vocab
        self.embedding_matrix = torch.tensor(embedding_matrix)
        if precomputed_bounds:
            self.bounds_dict = torch.load(precomputed_bounds)
        else:
            self.bounds_dict = self._precompute_bounds()

    def populate_neighbors(self):
        with open(self.json_file, 'r') as f:
            self.neighbors = json.load(f)

    def _precompute_bounds(self):
        """
        Precompute bounds for all words that have neighbors
        """
        bounds_dict = {}
        embedding_dim = self.embedding_matrix.shape[1]
        
        for word_idx, word in tqdm(enumerate(self.vocab)):
            word = str(word)
            if word in self.neighbors:
                embeddings = []
                for neighbor in self.neighbors[word]:
                    if neighbor in self.vocab:
                        idx = np.argwhere(neighbor == self.vocab).squeeze()
                        embeddings.append(torch.tensor(self.embedding_matrix[idx]))
                
                if embeddings:
                    embeddings = torch.stack(embeddings)
                    bounds_dict[word_idx] = {
                        'lower': torch.min(embeddings, dim=0)[0],
                        'upper': torch.max(embeddings, dim=0)[0]
                    }
        
        return bounds_dict

    def get_substitution_list(self, w):
        w = str(w)
        return self.neighbors[w] if w in self.neighbors else []

    def get_bounds(self, sentence, epsilon):
        batch_size, max_length = sentence.shape
        device = sentence.device
        self.embedding_matrix = self.embedding_matrix.to(device)

        lower_bounds = self.embedding_matrix[sentence]
        upper_bounds = self.embedding_matrix[sentence]

        seq_lengths = (sentence != 0).sum(dim=1)
        perturb_limits = (epsilon * seq_lengths).long()

        for batch_idx in range(batch_size):
            limit = perturb_limits[batch_idx]
            
            # Only process up to the perturbation limit
            for pos in range(min(limit, max_length)):
                word_idx = sentence[batch_idx, pos].item()
                
                # Skip padding tokens
                if word_idx == 0:
                    continue
                    
                # If word has precomputed bounds, use them
                if word_idx in self.bounds_dict:
                    bounds = self.bounds_dict[word_idx]
                    lower_bounds[batch_idx, pos] = bounds['lower'].to(device)
                    upper_bounds[batch_idx, pos] = bounds['upper'].to(device)
                else:
                    # If no neighbors, use original embedding
                    lower_bounds[batch_idx, pos] = self.embedding_matrix[word_idx]
                    upper_bounds[batch_idx, pos] = self.embedding_matrix[word_idx]
        return torch.stack([lower_bounds, upper_bounds])


