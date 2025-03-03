
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "1.0.0"

class AutoEncoder(nn.Module):
    def __init__(self, act_size: int, num_features: int, l1_coeff: float) -> None:
        super().__init__()

        self.l1_coeff = l1_coeff
        self.num_features = num_features

        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(act_size, num_features)))
        self.b_enc = nn.Parameter(torch.zeros(num_features))

        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(num_features, act_size)))
        self.b_dec = nn.Parameter(torch.zeros(act_size))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec
        return F.relu(x_cent @ self.W_enc + self.b_enc) # calcul des activations des concepts
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_dec + self.b_dec # calcul de la reconstruction

    def reconstruct_loss(self, x: torch.Tensor, acts: torch.Tensor, x_reconstruct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0) # loss de reconstruction MSE
        l1_loss = self.l1_coeff * (acts.float().abs().sum()) # penalité L1 sur les activations des concepts
        return l1_loss, l2_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] :
        """
        Args :
            x : input (B, S, act_size = d_model) ou (B*S, act_size = d_model)
        """
        hidden_acts = self.encode(x)
        x_reconstruct = self.decode(hidden_acts)

        l1_loss, l2_loss = self.reconstruct_loss(x, hidden_acts, x_reconstruct)
        loss = l2_loss + l1_loss # loss total

        return loss, x_reconstruct, hidden_acts, l2_loss, l1_loss
    
    # permet de stabiliser l'entraînement
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed


def load_sae(filename: str, act_size: int, num_features: int, device="cpu", verbose: bool = True) -> torch.Tensor:
    """
    Load weigths from a file and instanciate a SAE object
    """
    sae = AutoEncoder(act_size=act_size, num_features=num_features, l1_coeff=3e-4)
    sae.load_state_dict(
        torch.load(filename, map_location=torch.device('cpu'), weights_only=True)
        )
    if verbose:
        print(sae)
    return sae.to(device)