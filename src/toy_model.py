import einops
import torch

#----------------------------------------------------------------------------
# Denoiser model for learning 2D toy distributions.

class ToyModel(torch.nn.Module):
    def __init__(self,
        in_dim      = 2,      # Input dimensionality.
        num_layers  = 4,      # Number of hidden layers.
        hidden_dim  = 64,     # Number of hidden features.
        sigma_data  = 0.5,    # Expected standard deviation of the training data.
        new         = False,  # Whether to use the old version of the code or not
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.layers = torch.nn.Sequential()
        self.layers.append(torch.nn.Linear(in_dim + 2, hidden_dim))
        for _layer_idx in range(num_layers):
            self.layers.append(torch.nn.SiLU())
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.SiLU())

        self.layer_F =   torch.nn.Linear(hidden_dim, 2)
        self.gain_F  = torch.nn.Parameter(torch.zeros([]))

        if new:
            self.layer_G = torch.nn.Linear(hidden_dim, 3)
            self.gain_G= torch.nn.Parameter(torch.zeros([])).requires_grad_(False)
        self.new=new

    def forward(self, x, sigma=0):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1]).unsqueeze(-1)
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

        y = self.layers(torch.cat([c_in*x, sigma.log() / 4, torch.ones_like(sigma)], dim=-1))
        F = self.layer_F(y) * self.gain_F
        if not self.new: return F, None

        G = self.layer_G(y) * self.gain_G
        G = torch.clamp(G, min=-20, max=20) 
        
        return F, G



    def loss(self, x_0, sigma, Sigma):
        x_0 = x_0.detach()
        sigma = sigma.detach()
        Sigma = Sigma.detach()

        epsilon = torch.randn_like(x_0)

        x = x_0 + epsilon*sigma.unsqueeze(-1)
        F, G = self(x, sigma)

        sigma = sigma.unsqueeze(-1)
        target = (sigma*x_0 - self.sigma_data**2*epsilon)/(self.sigma_data*(sigma**2+self.sigma_data**2)**.5)
        error = F - target
        if not self.new: return (error**2).sum(-1).mean()
    
        logdet, S = transform_G(G)
        error = einops.einsum(S,error,'... i j, ... j -> ... j').pow(2).sum(dim=-1)

        c_out = self.sigma_data**2 / (sigma**2 * self.sigma_data**2)
        error = c_out*sigma**2*error
    
        Sigma_trace = einops.einsum(S,S,Sigma, '... i j, ... k i, ... j k -> ...')
        sigma_trace = sigma**2*einops.einsum(S,S,'... i j, ... j i -> ...')
    
        # This ensures the loss is minimized correctly instead of diverging.
        loss = .5*(error + sigma_trace + Sigma_trace - logdet)
        print(error.mean().cpu().detach().item(), sigma_trace.mean().cpu().detach().item(), Sigma_trace.mean().cpu().detach().item(), logdet.mean().cpu().detach().item(),)

        return loss.mean()

    def logp(self, x, sigma=0):
        F, G = self(x, sigma) 
        
        logdet, S = transform_G(G)
        
        # Ensure sigma has the correct shape for broadcasting
        sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1]).unsqueeze(-1) 
        
        # This is the target for F_theta when x = x̃, as we derived
        target = x * sigma_t / (self.sigma_data * (sigma_t**2 + self.sigma_data**2)**0.5)
        
        # The squared norm || F - target ||^2
        # For vectors, this would be a sum over the feature dimension
        error = (F - target)

        if not self.new:
            error = error**2
            coeff = self.sigma_data**2/(sigma_t**2+self.sigma_data**2)
            return -.5*(error*coeff).sum(dim=-1)
            

        # Assemble the log-probability according to the formula
        # log q = -0.5*G - 0.5 * exp(-G) * coeff * ||...||^2
        error = einops.einsum(S,error,'... i j, ... j -> ... j').pow(2).sum(dim=-1)
        c_out = self.sigma_data**2 / (sigma**2 * self.sigma_data**2)
        error = c_out*sigma**2*error

        log_prob = .5*(-error - logdet)
        
        # Sum the log-probabilities over the feature dimension
        return log_prob

    def pdf(self, x, sigma=0):
        logp = self.logp(x, sigma=sigma)
        pdf = (logp - logp.max()).exp()
        return pdf

    def score(self, x, sigma=0, graph=False):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            logp = self.logp(x, sigma=sigma)
            score = torch.autograd.grad(outputs=[logp.sum()], inputs=[x], create_graph=graph)[0]
        return score

        
        
    
def transform_G(G):
    S = torch.zeros([G.shape[0],2,2], device = G.device, dtype = G.dtype)
    S[:,0,0]=G[:,0].exp()
    S[:,1,1]=G[:,1].exp()
    S[:,0,1]=G[:,2].sinh()

    logdet = 2*(G[:,0] + G[:,1])

    return logdet, S