import torch, pdb
import torch.nn as nn
import torch.nn.functional as F

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, use_norm=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.use_norm = use_norm

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.normal_()

    def forward(self, z):
        if self.use_norm:
            z = l2norm(z)
        z_flattened = z.view(-1, self.e_dim)
        if self.use_norm:
            embedd_norm = l2norm(self.embedding.weight)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.use_norm:
            d = 2 - 2 * torch.einsum('bc, nc -> bn', z_flattened, embedd_norm)
        else:
            d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)

        encoding_indices = torch.argmin(d, dim=1).view(*z.shape[:-1])
        z_q = self.embedding(encoding_indices)
        if self.use_norm:
            z_q = l2norm(z_q)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q-z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices

    def decode_from_indice(self, indices):
        z_q = self.embedding(indices)
        if self.use_norm:
            z_q = l2norm(z_q)
        
        return z_q