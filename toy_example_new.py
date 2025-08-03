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
from src.gaussian_mixture import GaussianMixture, gt
from src.toy_model import ToyModel


import os
import copy
import warnings
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
# Train a 2D toy model with the given parameters.

def do_train(new=False, score_matching=True,
    classes='A', num_layers=4, hidden_dim=64, batch_size=4<<10, total_iter=4<<10, seed=0,
    P_mean=-2.3, P_std=1.5, sigma_data=0.5, lr_ref=1e-4, lr_iter=512, ema_decay=0.99,
    pkl_pattern=None, pkl_iter=256, viz_iter=32,
    device=torch.device(default_device),
):
    torch.manual_seed(seed)

    # Initialize model.
    net = ToyModel(num_layers=num_layers, hidden_dim=hidden_dim, sigma_data=sigma_data, new=new).to(device).train()
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), betas=(0.9, 0.99))
    print(f"score matching {score_matching}")
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
        if viz_iter is not None and iter_idx % viz_iter == 0:
            for x in plt.gca().lines: x.remove()
            do_plot(ema, elems={'samples'}, device=device)
            plt.savefig("immagine.png")
            plt.gcf().canvas.flush_events()

        # Execute one training iteration.
        opt.param_groups[0]['lr'] = lr_ref / np.sqrt(max(iter_idx / lr_iter, 1))
        opt.zero_grad()
        sigma = (torch.randn(batch_size, device=device) * P_std + P_mean).exp()

        clean_samples, Sigma = gt(classes, device).sample((batch_size,), torch.zeros_like(sigma))
        epsilon = torch.randn_like(clean_samples)
        noisy_samples = clean_samples + epsilon * sigma.unsqueeze(-1)

        if score_matching:
            gt_scores = gt(classes, device).score(noisy_samples, sigma)
            net_scores = net.score(noisy_samples, sigma, graph=True)
            score_matching_loss = ((sigma ** 2) * ((gt_scores - net_scores) ** 2).mean(-1)).mean()
            score_matching_loss.backward()
            with torch.no_grad():
                nll = net.loss(clean_samples, sigma, Sigma)
        else:
            nll = net.loss(clean_samples, sigma, Sigma)
            nll.backward()
            if iter_idx%16==0:
                with torch.no_grad():
                    gt_scores = gt(classes, device).score(noisy_samples, sigma)
                    net_scores = net.score(noisy_samples, sigma, graph=False)
                    score_matching_loss = ((sigma ** 2) * ((gt_scores - net_scores) ** 2).mean(-1)).mean()

        pbar.set_postfix_str(f"Score-Matching Loss: {score_matching_loss.item():.3f}, Negative Log-Likelyhood Loss: {nll.item():.3f}")
        opt.step()

        # Update EMA.
        # MODIFIED: Replaced phema with a standard fixed EMA decay.
        beta = ema_decay
        for p_net, p_ema in zip(net.parameters(), ema.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, beta))

        # Save model snapshot.
        if pkl_pattern is not None and (iter_idx + 1) % pkl_iter == 0:
            pkl_path = pkl_pattern % (iter_idx + 1)
            if os.path.dirname(pkl_path):
                os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            # MODIFIED: Replaced pickle with torch.save
            torch.save({k: v.cpu() for k, v in ema.state_dict().items()}, pkl_path)


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