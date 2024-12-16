import torch
import torch.nn.functional as F
from model import MembershipInferenceAttacker
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

class PGD():
    def __init__(self) -> None:
        pass

    def fgsm(self, model, x, label, eps):
        #TODO: implement this as an intermediate step of PGD
        # Notes: put the model in eval() mode for this function
        model.eval()
        # x.requires_grad_()
        
        output = model(x)
        loss = F.cross_entropy(output, label)
        grad = torch.autograd.grad(
                    loss, x, retain_graph=False, create_graph=False
                )[0]

        x_adv = x.detach() + eps * torch.sign(grad)
        return x_adv



    def pgd_untargeted(self, model, x, y, k, eps, eps_step):
        #TODO: implement this 
        # Notes: put the model in eval() mode for this function
        
        model.eval()
        adv_images = x.clone().detach()
        for _ in range(k):
            adv_images.requires_grad = True
            x_adv = self.fgsm(model, adv_images, y, eps_step)
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            adv_images = torch.clamp(x + delta, min = 0, max = 1)
        
        return adv_images

# # class LSTMPGD():
#     def __init__(self, model, epsilon=0.1, alpha=0.01, num_steps=5, 
#                  random_start=False, embedding_layer=None):
#         """
#         Initialize PGD Attack for text classification
        
#         Args:
#             model (nn.Module): Target model to be attacked
#             epsilon (float): Maximum perturbation magnitude
#             alpha (float): Step size for each iteration
#             num_steps (int): Number of attack iterations
#             random_start (bool): Whether to start with random initialization
#             embedding_layer (nn.Embedding): Model's embedding layer 
#         """
#         self.model = model
#         self.epsilon = epsilon
#         self.alpha = alpha
#         self.num_steps = num_steps
#         self.random_start = random_start
#         self.embedding_layer = embedding_layer

#     def perturb_embedding(self, embedding, delta):
#         """
#         Apply perturbation to embedding while maintaining constraints
        
#         Args:
#             embedding (torch.Tensor): Original embedding
#             delta (torch.Tensor): Proposed perturbation
        
#         Returns:
#             Perturbed embedding tensor
#         """
#         # Project perturbation to be within epsilon ball
#         perturbed = embedding + delta
#         perturbed = torch.clamp(perturbed, 
#                                 min=embedding - self.epsilon, 
#                                 max=embedding + self.epsilon)
#         return perturbed

#     def find_nearest_token(self, original_embedding, perturbed_embedding):
#         """
#         Find the nearest token to the perturbed embedding
        
#         Args:
#             original_embedding (torch.Tensor): Original token embedding
#             perturbed_embedding (torch.Tensor): Perturbed embedding
        
#         Returns:
#             Nearest token ID
#         """
#         # Compute distances between perturbed embedding and all embeddings
#         token_embeddings = self.embedding_layer.weight.data
#         for i, token in enumerate(perturbed_embedding):
#             print(token, token_embeddings)
#             distances = torch.cdist(token, token_embeddings).squeeze()
#             perturbed_embedding[i] = token_embeddings[distances.argmin()]
#         # Find the token with minimum distance
#         return perturbed_embedding

#     def attack(self, input_ids, labels):
#         """
#         Perform PGD attack on input text
        
#         Args:
#             input_ids (torch.Tensor): Input token IDs
#             labels (torch.Tensor): Ground truth labels
        
#         Returns:
#             Adversarial example
#         """
#         # Ensure model is in eval mode during attack
#         self.model.eval()
        
#         # Get original embeddings
#         batch_embeddings = self.embedding_layer(input_ids)
        
#         # Initialize perturbation
#         if self.random_start:
#             delta = torch.rand_like(batch_embeddings) *  self.epsilon
#         else:
#             delta = torch.zeros_like(batch_embeddings)
        
#         delta =  Variable(delta, requires_grad=True)
        
#         # PGD iteration
#         for _ in range(self.num_steps):
#             # Zero gradients
#             if delta.grad is not None:
#                 delta.grad.zero_()
            
#             # Forward pass with current perturbation
#             adv_embeddings = self.perturb_embedding(batch_embeddings, delta)
#             # adv_embeddings = self.find_nearest_token(batch_embeddings, adv_embeddings)

#             # Compute loss
#             outputs = self.model.lstm(adv_embeddings)[0][:, -1, :]
#             outputs = self.model.fc(outputs)
#             loss = F.cross_entropy(outputs, labels)
            
#             # Compute gradients
            
#             # Update delta using gradient sign method
#             delta.data = delta.data + self.alpha * delta.grad.sign()
            
#             # Project back to epsilon ball
#             delta.data = torch.clamp(delta.data, 
#                                      min=-self.epsilon, 
#                                      max=self.epsilon)
        
#         # Final adversarial example
#         # adv_embeddings = self.perturb_embedding(batch_embeddings, delta)
#         return adv_embeddings

def pgd_text(self, model, x, y, k, eps=0.2, eps_step=0.02):
    #TODO: implement this 
    # Notes: put the model in eval() mode for this function
    
    model.eval()
    adv_images = x.clone().detach()
    for _ in range(k):
        adv_images.requires_grad = True
        x_adv = perturb_data(model, adv_images, y, eps_step)
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        adv_images = torch.clamp(x + delta, min = 0, max = 1)
    
    return adv_images

def perturb_data(model, X_test, label, optimizer= None, epsilon=0.2):

    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
    else:
        X_test = X_test.clone().detach().requires_grad_(True)

    predictions = model.lstm(X_test)[0][:, -1, :]
    predictions = torch.sigmoid(model.fc(predictions))
    loss = F.binary_cross_entropy(predictions.squeeze(), label.float()).to('cuda')

    loss.backward()
    grad = X_test.grad
    # # Apply perturbation
    with torch.no_grad():
                perturbed_embeddings = X_test + epsilon * torch.sign(grad)
    return perturbed_embeddings.detach()


def membership_inference_attack(target_model, train_data, test_data, device):
    """
    Perform membership inference attack
    """
    # Prepare feature extraction
    def extract_features(model, data):
        model.eval()
        features = []
        with torch.no_grad():
            for x, _ in data:
                x = x.to(device).flatten()
                # Extract intermediate layer features
                intermediate = model.network[:-1](x)
                features.append(intermediate)
        return torch.cat(features)

    # Extract features
    train_features = extract_features(target_model, train_data)
    test_features = extract_features(target_model, test_data)

    # Prepare attack dataset
    train_labels = torch.ones(train_features.size(0), 1)
    test_labels = torch.zeros(test_features.size(0), 1)

    attack_features = torch.cat([train_features, test_features])
    attack_labels = torch.cat([train_labels, test_labels])

    # Split attack data
    X_train, X_test, y_train, y_test = train_test_split(
        attack_features.numpy(),
        attack_labels.numpy(),
        test_size=0.2
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    print(X_train.size())
    # Membership inference attacker
    attacker = MembershipInferenceAttacker(
        input_dim=X_train.size(-1),
        hidden_dim=64
    )

    # Train attacker
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attacker.parameters(), lr=0.001)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = attacker(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate attack performance
    with torch.no_grad():
        test_outputs = attacker(X_test)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()

    return accuracy.item()

