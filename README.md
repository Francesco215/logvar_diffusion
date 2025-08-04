# A Generalized Diffusion Model Training Framework

> \[!Note]
> This is a weekend-ish project I worked on for fun.



## How It Started

What if we trained a diffusion model capable of estimating the **confidence** of each prediction?

I also decided to follow these tight constraints:

* Maximize the log probability and nothing else.
* The prediction distribution must be Gaussian.
* The result should generalize the standard diffusion model framework.
* If the confidence level is set equal to the diffusion process noise \$\sigma\$, it should reduce exactly to the equations from [EDM](https://arxiv.org/abs/2206.00364).

So, I did the math and ran a small-scale experiment based on [Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/abs/2406.02507) to compare my method to the “standard” diffusion modeling approach.

## Results

In this small-scale experiment, the two models ended up with nearly identical performance. More experiments on complex datasets will be needed to determine whether the method brings real improvements.

<img width="49%" alt="baseline" src="https://github.com/user-attachments/assets/e73f69c6-a6c2-4f82-b497-5ca619e1086d" />

<img width="49%" alt="new_method" src="https://github.com/user-attachments/assets/506d78e7-fab4-455d-837e-188240bc42f6" />

> Left: standard EDM method.
> Right: my confidence-aware method.




## How to Run the Code

The code is lightweight and can even run on a laptop.

To run both experiments (plus the score-matching experiment), simply run the `ablations.py` script.

### Step-by-Step

1. Install `uv` with pip:

   ```bash
   pip install uv
   ```

2. Create and activate a virtual environment:

   ```bash
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

3. Install the package:

   ```bash
   uv pip install -e .
   ```

4. Run the experiment:

   ```bash
   python ablations.py
   ```

Results will be saved to a folder.



# How It Works

> \[!Warning]
> This is the nerd area, proceed with caution

Let’s make score-based diffusion modeling more rigorous.

Suppose we have a dataset containing a single sample \$x\_0\$, and we add noise to it as follows:

$$
\tilde x = x_0 + \sigma \cdot \epsilon
$$

This implies that the distribution is:

$$
p(x) = \mathcal{N}(x \mid x_0, \sigma^2)
$$

**But** what if \$x\_0\$ is itself sampled from a Gaussian? For instance, many VAEs encode images using non-isotropic Gaussians, and the diffusion process then adds noise on top. In that case:

$$
\tilde x = x_{\textrm{VAE}} + \sigma_{\textrm{VAE}} \cdot \epsilon + \sigma_{\textrm{diff}} \cdot \epsilon
$$

Which gives:

$$
p(x) = \mathcal{N}\left(x \mid x_0, \sigma_{\text{VAE}}^2 + \sigma_{\text{diff}}^2\right)
$$

You might think these two processes need different treatments — but we can hit two birds with one stone.

From now on, we consider the general case \$\tilde x \sim p(x)\$:

$$
\tilde x = x_0 + \sigma \cdot \epsilon
$$

And we allow \$\sigma\$ to be **non-isotropic**.



## The Loss Function: From First Principles

How do you choose the loss function?
You don’t — **nature does it for you**.

Given a data point \$x\$ and known noise level \$\sigma^2\$, we want to recover the original \$x\_0\$. That is, we learn two functions \$\mu\_\theta(\tilde x)\$ and \$\sigma\_\phi(\tilde x)\$ such that:

$$
q(x \mid \tilde x) = \mathcal{N}(x \mid \mu_\theta, \sigma_\phi^2)
$$

We then minimize the expected negative log-likelihood:

$$
L = \mathbb{E}_{\tilde x \sim p(x)}\left[-\log q(x \mid \tilde x)\right]
$$

Which decomposes into:

$$
L = \mathrm{KL}(p \| q) + S(p)
$$

Carrying out the KL divergence, we get:

$$
L = \frac{1}{2} \left[\log \sigma_\phi^2 + \frac{\sigma^2 + (\mu_\theta - x_0)^2}{\sigma_\phi^2} + \log(2\pi)\right]
$$

### Properties of the Loss

At the minimum, the optimal variance satisfies:

$$
\frac{\partial L}{\partial \sigma_\phi} = 0 \quad \Rightarrow \quad \sigma_\phi^2 = \sigma^2 + (\mu_\theta - x_0)^2
$$

So the learned variance captures both the added noise and the model’s prediction error — an **adaptive uncertainty estimate**.

Also, the gradient with respect to \$\mu\_\theta\$ is:

$$
\frac{\partial L}{\partial \mu_\theta} = \frac{\mu_\theta - x_0}{\sigma_\phi^2}
$$

This resembles the gradient of MSE, but weighted by the **predicted** error — providing a natural form of **loss weighting**. This is interesting because in [EDM2](https://arxiv.org/abs/2312.02696), a similar weighting was manually engineered, but here it emerges on its own from the equations 🔥.



## The Score Function

In score-based diffusion, we aim to estimate the **score function**:

$$
s(x|\sigma) = \nabla_x \log p(x|\sigma)
$$

Using Tweedie's formula, a natural estimator of the score is:

$$
\mathbb E \left[s(x \mid \sigma)\right] = \frac{x - \mathbb{E}[x_0 \mid x, \sigma]}{\sigma^2} = \frac{x - \mu_\theta}{\sigma^2}
$$

So we don’t need \$\sigma\_\phi\$ to compute the score.



### Solving the ODE

We can solve the denoising ODE using the usual techniques:

$$
\frac{d\mathbf{x}}{d\sigma} = \frac{x - \mu_\theta}{\sigma}
$$



## Preconditioning

When implementing this in practice, we must be cautious — neural networks don't handle very large or very small values well.

Training a model to predict \$\mu\_\theta\$ and \$\sigma\_\phi\$ directly is problematic. Instead, we reparameterize:

$$
\begin{cases}
\mu_\theta(x, \sigma) = c_{\textrm{skip}} \cdot x + c_{\textrm{out}} \cdot F_\theta(c_{\textrm{in}} \cdot x, c_{\textrm{noise}}) \\
\sigma_\phi(x, \sigma) = c_{\textrm{var}} \cdot \exp\left[\frac{1}{2} G_\phi(c_{\textrm{in}} \cdot x, c_{\textrm{noise}})\right]
\end{cases}
$$

Where \$(F\_\theta, G\_\phi)\$ are the neural network outputs, and:

$$
\begin{cases}
c_{\textrm{skip}}(\sigma) = \frac{\sigma_{\textrm{data}}}{\sigma^2 + \sigma_{\textrm{data}}^2} \\
c_{\textrm{out}}(\sigma) = \frac{\sigma \cdot \sigma_{\textrm{data}}}{\sqrt{\sigma^2 + \sigma_{\textrm{data}}^2}} \\
c_{\textrm{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\textrm{data}}^2}} \\
c_{\textrm{noise}}(\sigma) = \frac{1}{4} \log \sigma \\
c_{\textrm{var}}(\sigma) = \sigma \cdot \sqrt{1 + \frac{\sigma_{\textrm{data}}^2}{\sigma^2 + \sigma_{\textrm{data}}^2}}
\end{cases}
$$

We now substitute everything back into the original loss:

$$
L = \frac{1}{2} \left[\log \sigma_\phi^2 + \frac{\sigma^2 + (\mu_\theta - x_0)^2}{\sigma_\phi^2} + \log(2\pi)\right]
$$

After substitution, we get a numerically stable loss for both \$\sigma \to 0\$ and \$\sigma \to \infty\$:

$$
L = G_\phi + e^{-G_\phi} \left[1 + \frac{\sigma_{\textrm{data}}^2}{\sigma^2 + 2\sigma_{\textrm{data}}^2} \left(\left\|F_\theta - \frac{\sigma x_0 - \sigma_{\textrm{data}}^2 \epsilon}{\sigma_{\textrm{data}} \sqrt{\sigma^2 + \sigma_{\textrm{data}}^2}} \right\|^2 - 1\right)\right]
$$

> \[!Hint]
> To verify this, first substitute \$\sigma\_\phi\$ and then \$\mu\_\theta\$ into the original loss.



Using this reparametrization, we can rewrite the log-probability as:

$$
\log q(x \mid \tilde x) = -\frac{1}{2} \left[G_\phi + e^{-G_\phi} \left( \frac{\sigma_{\textrm{data}}^2}{\sigma^2 + 2\sigma_{\textrm{data}}^2} \right) \left\|F_\theta - \frac{x - c_{\textrm{skip}} \cdot \tilde x}{c_{\textrm{out}}} \right\|^2 + \log(2\pi c_{\textrm{var}}^2)\right]
$$

This formulation is **numerically stable** for all values of \$\sigma\$.
