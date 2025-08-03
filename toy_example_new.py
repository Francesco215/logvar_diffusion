# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Dependency-free version of the original 2D toy example from the paper
"Guiding a Diffusion Model with a Bad Version of Itself".
"""

import os
import copy
import warnings
import functools
import numpy as np
import torch
import matplotlib.pyplot as plt
import click
import tqdm


warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
if torch.cuda.is_available():
    default_device = 'cuda'
elif torch.backends.mps.is_available():
    default_device = 'mps'
else:
    default_device = 'cpu' 
torch.autograd.set_detect_anomaly(True)
#----------------------------------------------------------------------------
# Multivariate mixture of Gaussians. Allows efficient evaluation of the
# probability density function (PDF) and score vector, as well as sampling,
# using the GPU. The distribution can be optionally smoothed by applying heat
# diffusion (sigma >= 0) on a per-sample basis.
class GaussianMixture(torch.nn.Module):
    def __init__(self,
        phi,                    # Per-component weight: [comp]
        mu,                     # Per-component mean: [comp, dim]
        Sigma,                  # Per-component covariance matrix: [comp, dim, dim]
        sample_lut_size = 64<<10,   # Lookup table size for efficient sampling.
    ):
        super().__init__()
        self.register_buffer('phi', torch.tensor(np.asarray(phi) / np.sum(phi), dtype=torch.float32))
        self.register_buffer('mu', torch.tensor(np.asarray(mu), dtype=torch.float32))
        self.register_buffer('Sigma', torch.tensor(np.asarray(Sigma), dtype=torch.float32))

        # Precompute eigendecompositions of Sigma for efficient heat diffusion.
        L, Q = torch.linalg.eigh(self.Sigma) # Sigma = Q @ L @ Q
        self.register_buffer('_L', L) # L: [comp, dim, dim]
        self.register_buffer('_Q', Q) # Q: [comp, dim, dim]

        # Precompute lookup table for efficient sampling.
        self.register_buffer('_sample_lut', torch.zeros(sample_lut_size, dtype=torch.int64))
        phi_ranges = (torch.cat([torch.zeros_like(self.phi[:1]), self.phi.cumsum(0)]) * sample_lut_size + 0.5).to(torch.int32)
        for idx, (begin, end) in enumerate(zip(phi_ranges[:-1], phi_ranges[1:])):
            self._sample_lut[begin : end] = idx


    # Evaluate the terms needed for calculating PDF and score.
    def _eval(self, x, sigma=0):                                                    # x: [..., dim], sigma: [...]
        L = self._L + sigma[..., None, None] ** 2                                  # L' = L + sigma * I: [..., dim]
        d = L.prod(-1)                                                            # d = det(Sigma') = det(Q @ L' @ Q) = det(L'): [...]
        y = self.mu - x[..., None, :]                                             # y = mu - x: [..., comp, dim]
        z = torch.einsum('...ij,...j,...kj,...k->...i', self._Q, 1 / L, self._Q, y) # z = inv(Sigma') @ (mu - x): [..., comp, dim]
        c = self.phi / (((2 * np.pi) ** x.shape[-1]) * d).sqrt()                   # normalization factor of N(x; mu, Sigma')
        w = c * (-1/2 * torch.einsum('...i,...i->...', y, z)).exp()                # w = N(x; mu, Sigma'): [..., comp]
        return z, w

    # Calculate p(x; sigma) for the given sample points, processing at most the given number of samples at a time.
    def pdf(self, x, sigma=0, max_batch_size=1<<14):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1])
        x_batches = x.flatten(0, -2).split(max_batch_size)
        sigma_batches = sigma.flatten().split(max_batch_size)
        pdf_batches = [self._eval(xx, ss)[1].sum(-1) for xx, ss in zip(x_batches, sigma_batches)]
        return torch.cat(pdf_batches).reshape(x.shape[:-1]) # x.shape[:-1]

    # Calculate log(p(x; sigma)) for the given sample points, processing at most the given number of samples at a time.
    def logp(self, x, sigma=0, max_batch_size=1<<14):
        return self.pdf(x, sigma, max_batch_size).log()

    # Calculate \nabla_x log(p(x; sigma)) for the given sample points.
    def score(self, x, sigma=0):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1])
        z, w = self._eval(x, sigma)
        w = w[..., None]
        return (w * z).sum(-2) / w.sum(-2) # x.shape


    def sample(self, shape, sigma=0, generator=None, for_training=False):
        """
        Draws random samples. 
        If for_training=True, also returns the target mean and covariance for the loss function.
        """
        # Ensure sigma is a tensor with the correct shape
        sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=self.mu.device).broadcast_to(shape)
        
        # 1. Pick a component 'i' for each sample
        comp_indices = self._sample_lut[torch.randint(len(self._sample_lut), size=shape, device=sigma_t.device, generator=generator)]
        
        # 2. Get the parameters for the chosen components
        mu_i = self.mu[comp_indices]          # Mean of the clean component: [B, 2]
        L_i = self._L[comp_indices]           # Eigenvalues of the clean component cov: [B, 2]
        Q_i = self._Q[comp_indices]           # Eigenvectors of the clean component cov: [B, 2, 2]

        # 3. Calculate the total covariance of the NOISY sample: Sigma_total = Sigma_i + sigma^2 * I
        # This is done by adding sigma^2 to the eigenvalues before reconstructing.
        L_total = L_i + sigma_t[..., None] ** 2  # Shape: [B, 2]
        
        # The full covariance matrix used to generate the sample
        Sigma_total = torch.einsum('...ij,...jk,...lk->...il', Q_i, torch.diag_embed(L_total), Q_i)

        # 4. Generate the noisy sample x_t ~ N(mu_i, Sigma_total)
        rand_norm = torch.randn(L_total.shape, device=sigma_t.device, generator=generator) # Shape: [B, 2]
        # Transform standard normal noise by sqrt(Sigma_total)
        y = torch.einsum('...ij,...j,...kj,...k->...i', Q_i, L_total.sqrt(), Q_i, rand_norm)
        noisy_sample = mu_i + y
        
        # If not for training, just return the sample for compatibility with plotting etc.
        if not for_training:
            return noisy_sample
            
        # For training, return the full tuple needed by the new loss function
        # The target for the model is to predict mu_i from noisy_sample, knowing the noise
        # had a total covariance of Sigma_total.
        target_mean = mu_i
        target_covariance = Sigma_total
        
        return noisy_sample, target_mean, target_covariance


#----------------------------------------------------------------------------
# Construct a ground truth 2D distribution for the given set of classes
# ('A', 'B', or 'AB').

@functools.lru_cache(None)
def gt(classes='A', device=torch.device('cpu'), seed=2, origin=np.array([0.0030, 0.0325]), scale=np.array([1.3136, 1.3844])):
    rnd = np.random.RandomState(seed)
    comps = []

    # Recursive function to generate a given branch of the distribution.
    def recurse(cls, depth, pos, angle):
        if depth >= 7:
            return

        # Choose parameters for the current branch.
        dir = np.array([np.cos(angle), np.sin(angle)])
        dist = 0.292 * (0.8 ** depth) * (rnd.randn() * 0.2 + 1)
        thick = 0.2 * (0.8 ** depth) / dist
        size = scale * dist * 0.06

        # Represent the current branch as a sequence of Gaussian components.
        for t in np.linspace(0.07, 0.93, num=8):
            # MODIFIED: Replaced dnnlib.EasyDict with a standard Python dict
            c = dict()
            c['cls'] = cls
            c['phi'] = dist * (0.5 ** depth)
            c['mu'] = (pos + dir * dist * t) * scale
            c['Sigma'] = (np.outer(dir, dir) + (np.eye(2) - np.outer(dir, dir)) * (thick ** 2)) * np.outer(size, size)
            comps.append(c)

        # Generate each child branch.
        for sign in [1, -1]:
            recurse(cls=cls, depth=(depth + 1), pos=(pos + dir * dist), angle=(angle + sign * (0.7 ** depth) * (rnd.randn() * 0.2 + 1)))

    # Generate each class.
    recurse(cls='A', depth=0, pos=origin, angle=(np.pi * 0.25))
    recurse(cls='B', depth=0, pos=origin, angle=(np.pi * 1.25))

    # Construct a GaussianMixture object for the selected classes.
    sel = [c for c in comps if c['cls'] in classes]
    distrib = GaussianMixture([c['phi'] for c in sel], [c['mu'] for c in sel], [c['Sigma'] for c in sel])

    print("Initialized GaussianMixture")
    return distrib.to(device)


#----------------------------------------------------------------------------
# Denoiser model for learning 2D toy distributions.

# REMOVED: @persistence.persistent_class decorator
class ToyModel(torch.nn.Module):
    # Replace the __init__ and forward methods in your `ToyModel` class
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

        self.layer_mean = torch.nn.Linear(hidden_dim, 2)
        self.gain_mean  = torch.nn.Parameter(torch.zeros([]))

        if new:
            # G now needs to output 3 values for the Cholesky factor (a, b, c)
            self.layer_cholesky = torch.nn.Linear(hidden_dim, 3) 
            self.gain_cholesky = torch.nn.Parameter(torch.zeros([]))
        self.new = new

    def forward(self, x, sigma=0):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1]).unsqueeze(-1)
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

        y = self.layers(torch.cat([c_in*x, sigma.log() / 4, torch.ones_like(sigma)], dim=-1))
        F = self.layer_mean(y) * self.gain_mean
        
        if not self.new: return F, None

        # G is now a tensor of shape [B, 3]
        G = self.layer_cholesky(y) * self.gain_cholesky
        G = G.clamp(min=-20, max=20)
        
        return F, G


    # You can replace your existing loss function with this more efficient version.

    def loss(self, noisy_x, target_mean, target_Sigma, sigma):
        """
        Computes the loss using the precision matrix (inverse covariance) for efficiency.
        
        The network now learns the Cholesky factor of P_phi = Sigma_phi^-1.
        """
        # Get raw network outputs F (for mean) and G (for the precision matrix's Cholesky factor)
        F_theta, G_raw = self(noisy_x, sigma)
        
        # --- Original Isotropic Loss (for backward compatibility) ---
        if not self.new:
            # This part remains the same as it doesn't use the non-isotropic path.
            clean_x = target_mean # In the old logic, target_mean is clean_x
            s = sigma.unsqueeze(-1)
            epsilon = (noisy_x - clean_x) / s
            target = (s * clean_x - self.sigma_data**2 * epsilon) / (self.sigma_data * (s**2 + self.sigma_data**2)**0.5)
            error = F_theta - target
            return (error**2).sum(-1).mean()

        # --- New Non-Isotropic Loss Calculation (using Precision Matrix) ---
        
        # 1. Construct the predicted mean mu_theta
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        mu_theta = c_skip.unsqueeze(-1) * noisy_x + c_out.unsqueeze(-1) * F_theta

        # 2. Construct the predicted precision matrix P_phi from its Cholesky factor
        a, b_raw, c = G_raw[..., 0], G_raw[..., 1], G_raw[..., 2]
        b = torch.sinh(b_raw)

        L_P_prime = torch.zeros(noisy_x.shape[0], 2, 2, device=noisy_x.device)
        L_P_prime[:, 0, 0] = torch.exp(a)
        L_P_prime[:, 1, 0] = b
        L_P_prime[:, 1, 1] = torch.exp(c)
        
        # Preconditioning scales the precision matrix by 1/c_var^2
        c_var = sigma * (1 + self.sigma_data**2 / (sigma**2 + self.sigma_data**2)).sqrt()
        P_phi = (L_P_prime @ L_P_prime.transpose(-1, -2)) / c_var.unsqueeze(-1).unsqueeze(-1)**2

        # 3. Compute the KL Divergence Loss: L = 0.5 * [ tr(P_phi @ S) + err^T @ P_phi @ err - logdet(P_phi) ]
        # This avoids all matrix inversions/solves!
        
        # Trace term: tr(P_phi @ target_Sigma)
        trace_term = torch.vmap(torch.trace)(P_phi @ target_Sigma)
        
        # Error term: (mu_theta - target_mean)^T @ P_phi @ (mu_theta - target_mean)
        error_vec = mu_theta - target_mean
        # This is calculated with a matrix-vector product, then a dot product.
        quad_form_vec = torch.einsum('bij,bj->bi', P_phi, error_vec)
        mahalanobis_term = torch.einsum('bi,bi->b', error_vec, quad_form_vec)

        # Log-determinant term: -log(det(P_phi))
        # log(det(P_phi)) is easily computed from its Cholesky factor
        log_det_P_phi = 2 * (a + c) - 2 * torch.log(c_var)
        
        # Combine terms. We add back the constant logdet(target_Sigma) which was part of the original KL divergence.
        # KL(p||q) = 0.5 * [ trace + error - logdet(P_phi) - logdet(target_Sigma) - k]
        loss_per_sample = 0.5 * (trace_term + mahalanobis_term - log_det_P_phi)

        return loss_per_sample.mean()


    def logp(self, x, sigma=0):
        """
        Calculates log q(y | x, sigma) evaluated at y=x, using the precision matrix.
        This version correctly handles both batched and grid inputs.
        """
        # --- Reshape Input for Consistent Batch Handling ---
        original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, original_shape[-1])

        # Get raw network outputs for mean (F) and precision matrix (G).
        F_theta, G_raw = self(x, sigma)

        # --- Case 1: Original Isotropic Model (for backward compatibility) ---
        if not self.new:
            sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1]).unsqueeze(-1)
            target = x * sigma_t / (self.sigma_data * (sigma_t**2 + self.sigma_data**2)**0.5)
            error_sq = (F_theta - target)**2

            if G_raw is None:
                coeff = self.sigma_data**2 / (sigma_t**2 + self.sigma_data**2)
                logp_flat = -0.5 * (error_sq * coeff).sum(dim=-1)
            else:
                coeff = self.sigma_data**2 / (sigma_t**2 + 2 * self.sigma_data**2)
                log_prob_per_dim = -0.5 * G_raw - 0.5 * torch.exp(-G_raw) * coeff * error_sq
                logp_flat = log_prob_per_dim.sum(dim=-1)
            
            # Reshape output back to original grid shape if necessary
            if len(original_shape) > 2:
                return logp_flat.reshape(original_shape[:-1])
            return logp_flat

        # --- Case 2: New Non-Isotropic Model (using Precision Matrix) ---
        sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1])

        c_skip = self.sigma_data**2 / (sigma_t**2 + self.sigma_data**2)
        c_out = sigma_t * self.sigma_data / (sigma_t**2 + self.sigma_data**2).sqrt()
        mu_theta = c_skip.unsqueeze(-1) * x + c_out.unsqueeze(-1) * F_theta

        a, b_raw, c = G_raw[..., 0], G_raw[..., 1], G_raw[..., 2]
        b = torch.sinh(b_raw)

        L_P_prime = torch.zeros(x.shape[0], 2, 2, device=x.device)
        L_P_prime[:, 0, 0] = torch.exp(a)
        L_P_prime[:, 1, 0] = b
        L_P_prime[:, 1, 1] = torch.exp(c)
        
        c_var = sigma_t * (1 + self.sigma_data**2 / (sigma_t**2 + self.sigma_data**2)).sqrt()
        
        P_phi = (L_P_prime @ L_P_prime.transpose(-1, -2)) / c_var.unsqueeze(-1).unsqueeze(-1)**2

        k = x.shape[-1]
        error_vec = x - mu_theta

        quad_form_vec = torch.einsum('bij,bj->bi', P_phi, error_vec)
        quadratic_term = torch.einsum('bi,bi->b', error_vec, quad_form_vec)

        log_det_P_phi = 2 * (a + c) - 2 * torch.log(c_var)
        
        log_prob_flat = 0.5 * (log_det_P_phi - quadratic_term - k * np.log(2 * np.pi))
        
        # Reshape the final output to match the original input's grid shape
        if len(original_shape) > 2:
            return log_prob_flat.reshape(original_shape[:-1])
        return log_prob_flat

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

#----------------------------------------------------------------------------
# Train a 2D toy model with the given parameters.

def do_train(new=False, score_matching=True,
    classes='A', num_layers=4, hidden_dim=64, batch_size=4<<10, total_iter=4<<10, seed=0,
    P_mean=-2.3, P_std=1.5, sigma_data=0.5, lr_ref=1e-2, lr_iter=512, ema_decay=0.99,
    pkl_pattern=None, pkl_iter=256, viz_iter=32,
    device=torch.device(default_device),
):
    # The ground truth score function is only valid for isotropic noise.
    # Therefore, score_matching=True is not supported for the new method.
    if new and score_matching:
        warnings.warn("Score matching with ground truth is not supported for the new non-isotropic model. Training with NLL loss instead.")
        score_matching = False

    torch.manual_seed(seed)

    # Initialize model.
    net = ToyModel(num_layers=num_layers, hidden_dim=hidden_dim, sigma_data=sigma_data, new=new).to(device).train().requires_grad_(True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), betas=(0.9, 0.99))
    print(f"Training with new method: {new}, Score matching: {score_matching}")

    # Initialize plot.
    if viz_iter is not None:
        plt.ion()
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['figure.subplot.left'] = plt.rcParams['figure.subplot.bottom'] = 0
        plt.rcParams['figure.subplot.right'] = plt.rcParams['figure.subplot.top'] = 1
        plt.figure(figsize=[12, 12], dpi=75)
        do_plot(ema, elems={'gt_uncond', 'gt_outline'}, device=device)
        plt.gcf().canvas.flush_events()

    # Training loop.
    pbar = tqdm.tqdm(range(total_iter))
    for iter_idx in pbar:

        # Visualize current sample distribution.
        if viz_iter is not None and iter_idx > 0 and iter_idx % viz_iter == 0:
            for x in plt.gca().lines: x.remove()
            do_plot(ema, elems={'samples'}, device=device)
            plt.savefig("immagine.png")
            plt.gcf().canvas.flush_events()

        # Execute one training iteration.
        opt.param_groups[0]['lr'] = lr_ref / np.sqrt(max(iter_idx / lr_iter, 1))
        opt.zero_grad()
        sigma = (torch.randn(batch_size, device=device) * P_std + P_mean).exp().clamp(max=20) # Clamp sigma for stability

        # Generate data and calculate loss based on the training method.
        if score_matching: # This path is only for the original model (new=False)
            clean_samples = gt(classes, device).sample((batch_size,))
            noisy_samples = clean_samples + torch.randn_like(clean_samples) * sigma.unsqueeze(-1)
            gt_scores = gt(classes, device).score(noisy_samples, sigma)
            net_scores = net.score(noisy_samples, sigma, graph=True)
            score_matching_loss = ((sigma ** 2) * ((gt_scores - net_scores) ** 2).sum(-1)).mean()
            score_matching_loss.backward()
            opt.step()
            with torch.no_grad():
                nll = net.loss(noisy_samples, clean_samples, None, sigma)
        else: # This path handles NLL loss for both the new and original models.
            if new:
                noisy_samples, target_mean, target_cov = gt(classes, device).sample((batch_size,), sigma, for_training=True)
                nll = net.loss(noisy_samples, target_mean, target_cov, sigma)
            else: # Original model with NLL loss
                clean_samples = gt(classes, device).sample((batch_size,))
                noisy_samples = clean_samples + torch.randn_like(clean_samples) * sigma.unsqueeze(-1)
                nll = net.loss(noisy_samples, clean_samples, None, sigma)
            
            nll.backward()
            opt.step()

            # For logging purposes, calculate score matching loss intermittently for the original model.
            if not new and iter_idx % 16 == 0:
                with torch.no_grad():
                    gt_scores = gt(classes, device).score(noisy_samples, sigma)
                    net_scores = net.score(noisy_samples, sigma, graph=False)
                    score_matching_loss = ((sigma ** 2) * ((gt_scores - net_scores) ** 2).sum(-1)).mean()
            else:
                score_matching_loss = torch.tensor(float('nan'))

        pbar.set_postfix_str(f"SM Loss: {score_matching_loss.item():.3f}, NLL Loss: {nll.item():.3f}")

        # Update EMA.
        beta = ema_decay
        for p_net, p_ema in zip(net.parameters(), ema.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, beta))

        # Save model snapshot.
        if pkl_pattern is not None and (iter_idx + 1) % pkl_iter == 0:
            pkl_path = pkl_pattern % (iter_idx + 1)
            if os.path.dirname(pkl_path):
                os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            torch.save(ema.cpu().state_dict(), pkl_path)
#----------------------------------------------------------------------------
# Simulate the EDM sampling ODE for the given set of initial sample points.

@torch.no_grad()
def do_sample(net, x_init, guidance=1, gnet=None, num_steps=32, sigma_min=0.002, sigma_max=5, rho=7):
    # Guided denoiser.
    def denoise(x, sigma):
        score = net.score(x, sigma)
        if gnet is not None:
            score = gnet.score(x, sigma).lerp(score, guidance)
        return x + score * (sigma ** 2)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_init.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_cur = x_init
    trajectory = [x_cur]
    net = net.to(x_init.device)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1

        # Euler step.
        d_cur = (x_cur - denoise(x_cur, t_cur)) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        # Record trajectory.
        x_cur = x_next
        trajectory.append(x_cur)
    return torch.stack(trajectory)

#----------------------------------------------------------------------------
# Draw the given set of plot elements using matplotlib.

@torch.no_grad()
def do_plot(
    net=None, guidance=1, gnet=None, elems={'gt_uncond', 'gt_outline', 'samples'},
    view_x=0, view_y=0, view_size=1.6, grid_resolution=400, arrow_len=0.002,
    num_samples=1<<13, seed=1, sample_distance=0, sigma_max=5,
    device=torch.device(default_device),
):
    # Generate initial samples.
    if any(x.startswith(y) for x in elems for y in ['samples', 'trajectories', 'scores']):
        samples = gt('A', device).sample((num_samples,), sigma_max, generator=torch.Generator(device).manual_seed(seed))
        if sample_distance > 0:
            ok = torch.ones(len(samples), dtype=torch.bool, device=device)
            for i in range(1, len(samples)):
                ok[i] = (samples[i] - samples[:i][ok[:i]]).square().sum(-1).sqrt().min() >= sample_distance
            samples = samples[ok]

    # Run sampler.
    if any(x.startswith(y) for x in elems for y in ['samples', 'trajectories']):
        trajectories = do_sample(net=(net or gt('A', device)), x_init=samples, guidance=guidance, gnet=gnet, sigma_max=sigma_max)

    # Initialize plot.
    gridx = torch.linspace(view_x - view_size, view_x + view_size, steps=grid_resolution, device=device)
    gridy = torch.linspace(view_y - view_size, view_y + view_size, steps=grid_resolution, device=device)
    gridxy = torch.stack(torch.meshgrid(gridx, gridy, indexing='xy'), axis=-1)
    plt.xlim(float(gridx[0]), float(gridx[-1]))
    plt.ylim(float(gridy[0]), float(gridy[-1]))
    plt.gca().set_aspect('equal')
    plt.gca().set_axis_off()

    # Plot helper functions.
    def contours(values, levels, colors=None, cmap=None, alpha=1, linecolors='black', linealpha=1, linewidth=2.5):
        values = -(values.max() - values).sqrt().cpu().numpy()
        plt.contourf(gridx.cpu().numpy(), gridy.cpu().numpy(), values, levels=levels, antialiased=True, extend='max', colors=colors, cmap=cmap, alpha=alpha)
        plt.contour(gridx.cpu().numpy(), gridy.cpu().numpy(), values, levels=levels, antialiased=True, colors=linecolors, alpha=linealpha, linestyles='solid', linewidths=linewidth)
    def lines(pos, color='black', alpha=1):
        plt.plot(*pos.cpu().numpy().T, '-', linewidth=5, solid_capstyle='butt', color=color, alpha=alpha)
    def arrows(pos, dir, color='black', alpha=1):
        plt.quiver(*pos.cpu().numpy().T, *dir.cpu().numpy().T * arrow_len, scale=0.6, width=5e-3, headwidth=4, headlength=3, headaxislength=2.5, capstyle='round', color=color, alpha=alpha)
    def points(pos, color='black', alpha=1, size=30):
        plt.plot(*pos.cpu().numpy().T, '.', markerfacecolor='black', markeredgecolor='none', color=color, alpha=alpha, markersize=size)

    # Draw requested plot elements.
    if 'p_net' in elems:          contours(net.logp(gridxy, sigma_max), levels=np.linspace(-2.5, 2.5, num=20)[1:-1], cmap='Greens', linealpha=0.2)
    if 'p_gnet' in elems:         contours(gnet.logp(gridxy, sigma_max), levels=np.linspace(-2.5, 3.5, num=20)[1:-1], cmap='Reds', linealpha=0.2)
    if 'p_ratio' in elems:        contours(net.logp(gridxy, sigma_max) - gnet.logp(gridxy, sigma_max), levels=np.linspace(-2.2, 1.0, num=20)[1:-1], cmap='Blues', linealpha=0.2)
    if 'gt_uncond' in elems:      contours(gt('AB', device).logp(gridxy), levels=[-2.12, 0], colors=[[0.9,0.9,0.9]], linecolors=[[0.7,0.7,0.7]], linewidth=1.5)
    if 'gt_outline' in elems:     contours(gt('A', device).logp(gridxy), levels=[-2.12, 0], colors=[[1.0,0.8,0.6]], linecolors=[[0.8,0.6,0.5]], linewidth=1.5)
    if 'gt_smax' in elems:        contours(gt('A', device).logp(gridxy, sigma_max), levels=[-1.41, 0], colors=['C1'], alpha=0.2, linealpha=0.2)
    if 'gt_shaded' in elems:      contours(gt('A', device).logp(gridxy), levels=np.linspace(-2.5, 3.07, num=15)[1:-1], cmap='Oranges', linealpha=0.2)
    if 'trajectories' in elems:   lines(trajectories.transpose(0, 1), alpha=0.3)
    if 'scores_net' in elems:     arrows(samples, net.score(samples, sigma_max), color='C2')
    if 'scores_gnet' in elems:    arrows(samples, gnet.score(samples, sigma_max), color='C3')
    if 'scores_ratio' in elems:   arrows(samples, net.score(samples, sigma_max) - gnet.score(samples, sigma_max), color='C0')
    if 'samples' in elems:        points(trajectories[-1], size=15, alpha=0.25)
    if 'samples_before' in elems: points(samples)
    if 'samples_after' in elems:  points(trajectories[-1])

#----------------------------------------------------------------------------
# Main command line.

@click.group()
def cmdline():
    """2D toy example from the paper "Guiding a Diffusion Model with a Bad Version of Itself"."""
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

#----------------------------------------------------------------------------
# 'train' subcommand.

@cmdline.command()
@click.option('--new',      help='Does it use the new method?', metavar='BOOL', type=bool,  default=True)
@click.option('--sm',       help='Score Matching', metavar='BOOL',              type=bool,  default=True)
@click.option('--outdir',   help='Output directory', metavar='DIR',             type=str,  default='training-runs')
@click.option('--cls',      help='Target classes', metavar='A|B|AB',            type=str,  default='A', show_default=True)
@click.option('--layers',   help='Number of layers', metavar='INT',             type=int,  default=4, show_default=True)
@click.option('--dim',      help='Hidden dimension', metavar='INT',             type=int,  default=64, show_default=True)
@click.option('--viz',      help='Visualize progress?', metavar='BOOL',         type=bool, default=True, show_default=True)
@click.option('--device',   help='PyTorch device',                              type=str,  default=default_device)
def train(new, sm, outdir, cls, layers, dim, viz, device):
    """Train a 2D toy model with the given parameters."""
    print(f'Will save snapshots to {outdir}')
    os.makedirs(outdir, exist_ok=True)
    # MODIFIED: Changed extension to .pt for PyTorch models
    pkl_pattern = os.path.join(outdir, 'iter_%04d.pt')
    viz_iter = 32 if viz else None
    print('Training...')
    do_train(new=new, score_matching=sm, pkl_pattern=pkl_pattern, classes=cls, num_layers=layers, hidden_dim=dim, viz_iter=viz_iter, device=torch.device(device))
    print('Done.')

#----------------------------------------------------------------------------
# 'plot' subcommand.

@cmdline.command()
@click.option('--net',      help='Main model checkpoint', metavar='PATH', type=str, required=True)
@click.option('--gnet',     help='Guiding model checkpoint', metavar='PATH',type=str, default=None)
@click.option('--new',      help='Use the new model architecture?', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--guidance', help='Guidance weight', metavar='FLOAT', type=float, default=3, show_default=True)
@click.option('--save',     help='Save figure, do not display', metavar='PNG|PDF', type=str, default=None)
@click.option('--layers',   help='Number of layers in loaded models', metavar='INT', type=int, default=4, show_default=True)
@click.option('--dim',      help='Hidden dimension of loaded models', metavar='INT', type=int, default=64, show_default=True)
@click.option('--device',   help='PyTorch device', type=str, default=default_device)
def plot(net, gnet, new, guidance, save, layers, dim, device):
    """Visualize sampling distributions with and without guidance."""
    print('Loading models...')
    device = torch.device(device)

    # MODIFIED: Replaced pickle/dnnlib with torch.load and model instantiation
    main_net = ToyModel(num_layers=layers, hidden_dim=dim, new=new).to(device)
    main_net.load_state_dict(torch.load(net, map_location=device))
    main_net.eval()

    guiding_net = None
    if gnet is not None:
        # For simplicity, assumes guiding net has same architecture.
        guiding_net = ToyModel(num_layers=layers, hidden_dim=dim, new=new).to(device)
        guiding_net.load_state_dict(torch.load(gnet, map_location=device))
        guiding_net.eval()

    # Initialize plot.
    print('Drawing plots...')
    plt.rcParams['font.size'] = 28
    plt.figure(figsize=[48, 25], dpi=40, tight_layout=True)
    fig1_kwargs = dict(view_x=0.30, view_y=0.30, view_size=1.2, num_samples=1<<14, device=device)
    fig2_kwargs = dict(view_x=0.45, view_y=1.22, view_size=0.3, num_samples=1<<12, device=device, sample_distance=0.045, sigma_max=0.03)

    # Draw first row.
    plt.subplot(2, 4, 1)
    plt.title('Ground truth distribution')
    do_plot(elems={'gt_uncond', 'gt_outline', 'samples'}, **fig1_kwargs)
    plt.subplot(2, 4, 2)
    plt.title('Sample distribution without guidance')
    do_plot(net=main_net, elems={'gt_uncond', 'gt_outline', 'samples'}, **fig1_kwargs)
    plt.subplot(2, 4, 3)
    plt.title('Sample distribution with guidance')
    do_plot(net=main_net, gnet=guiding_net, guidance=guidance, elems={'gt_uncond', 'gt_outline', 'samples'}, **fig1_kwargs)
    plt.subplot(2, 4, 4)
    plt.title('Trajectories without guidance')
    do_plot(net=main_net, elems={'gt_shaded', 'trajectories', 'samples_after'}, **fig2_kwargs)

    # Draw second row.
    plt.subplot(2, 4, 5)
    plt.title('PDF of main model')
    do_plot(net=main_net, elems={'p_net', 'gt_smax', 'scores_net', 'samples_before'}, **fig2_kwargs)
    plt.subplot(2, 4, 6)
    plt.title('PDF of guiding model')
    do_plot(net=main_net, gnet=guiding_net, elems={'p_gnet', 'gt_smax', 'scores_gnet', 'samples_before'}, **fig2_kwargs)
    plt.subplot(2, 4, 7)
    plt.title('PDF ratio (main / guiding)')
    do_plot(net=main_net, gnet=guiding_net, elems={'p_ratio', 'gt_smax', 'scores_ratio', 'samples_before'}, **fig2_kwargs)
    plt.subplot(2, 4, 8)
    plt.title('Trajectories with guidance')
    do_plot(net=main_net, gnet=guiding_net, guidance=guidance, elems={'gt_shaded', 'trajectories', 'samples_after'}, **fig2_kwargs)

    # Save or display.
    if save is not None:
        print(f'Saving to {save}')
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=80)
    else:
        print('Displaying...')
        plt.show()
    print('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()