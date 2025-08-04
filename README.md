# A Generalized Diffusion Model Training Framework

Let’s make score-based diffusion modeling more rigorous.

Suppose we have a dataset made of a single sample $x_0$, and we add noise to this sample as follows:

$$
\tilde x = x_0 + \sigma \cdot \epsilon
$$

This implies that the probability distribution is:

$$
p(x) = \mathcal{N}(x \mid x_0, \sigma^2)
$$

**But** what if $x_0$ is itself sampled from a small Gaussian? After all, most VAEs encode images using non-isotropic Gaussians and then the diffusion process adds noise on top. In this case:

$$
\tilde x = x_\textrm{vae} + \sigma_{\text{vae}} \cdot \epsilon + \sigma_{\text{diff}} \cdot \epsilon
$$

which gives:

$$
p(x) = \mathcal{N}\left(x \mid x_0, \sigma_{\text{VAE}}^2 + \sigma_{\text{diff}}^2 \right)
$$

You might think these two processes need separate treatments, but I’ll show you how to hit two pigeons with one stone.

From now on, we’ll consider the case where we sampe $\tilde x \sim p(x)$:

$$
\tilde x = x_0 + \sigma \cdot \epsilon
$$

and allow $\sigma$ to be non-isotropic.


## The Loss Function: From First Principles

How do you choose the loss function? You don’t — nature does it for you.

Given a data point $x$ and known noise level $\sigma^2$, we want to estimate the original distribution of $x_0$. That is, we aim to learn two functions $\mu_\theta(\tilde x)$, $\sigma_\phi(\tilde x)$ such that:

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

Carrying out the KL divergence calculation, assuming known noise level $\sigma$, we get:

$$
L = \frac{1}{2} \left[\log \sigma_\phi^2 + \frac{\sigma^2 + (\mu_\theta - x_0)^2}{\sigma_\phi^2} + \log(2\pi) \right]
$$


### Properties of the Loss

At the minimum, the optimal value of $\sigma_\phi$ satisfies:

$$
\frac{\partial L}{\partial \sigma_\phi} = 0 \quad \Rightarrow \quad \sigma_\phi^2 = \sigma^2 + (\mu_\theta - x_0)^2
$$

So the learned variance converges to the sum of the added noise and the model’s squared prediction error — an adaptive uncertainty estimate.

Also, the gradient with respect to $\mu_\theta$ is:

$$
\frac{\partial L}{\partial \mu_\theta} = \frac{\mu_\theta - x_0}{\sigma_\phi^2}
$$

This resembles the gradient of the MSE loss, but weighted by the expected prediction error $\sigma_\phi^2$. This is good because it also act as a loss-weight mechanism, this way training is more stable.


## The Score Function

The core objective of score-based diffusion models is to estimate the **score function**, defined as:

$$
s(x) = \nabla_x \log p(x)
$$

Thanks to Tweedie's formula the expected value of the score can be expressed as

So a natural estimator of the **true score** of $p(x)$ is:

$$
s(x|\sigma) = \frac {x-\mathbb E[x_0|x,\sigma]}{\sigma^2}= \frac {x-\mathbb \mu_\theta}{\sigma^2}
$$

So we don't need to evaluate $\sigma_\phi$ for the score

### Solving the ODE

The ODE can be solved like with standard diffusion modelling techniques

$$\frac{d\mathbf{x}}{d\sigma} = \frac{x - \mu_\theta}{\sigma}$$



## Preconditioning
When it comes with writing an implementation one has to be careful, Neural networks don't like really large and really small numbers.

Training a model to predict directly $\mu_\theta$ and $\sigma_\phi$ is far from ideal. For this reason we have to re-parametrize it.

$$
\begin{cases}
\mu_\theta(x, \sigma) = c_\textrm{skip}\cdot x + c_\textrm{out}\cdot F_\theta(c_\textrm{in}\cdot x,c_\textrm{noise})\\
\sigma_\phi(x,\sigma) = c_\textrm{var}\cdot\exp\left[\frac 12 G_\phi(c_\textrm{in}\cdot x, c_\textrm{noise})\right]
\end{cases}
$$

Where $(F_\theta,G_\phi)$ is the actual neural network output, and $c_\textrm{skip},\ c_\textrm{out},\ c_\textrm{in},\ c_\textrm{noise}$ are equal to 

$$
\begin{cases}
c_\textrm{skip}(\sigma) = \sigma_\textrm{data}/(\sigma^2 + \sigma_\textrm{data}^2)\\
c_\textrm{out}(\sigma) = \sigma\cdot \sigma_\textrm{data}/\sqrt{\sigma^2 + \sigma_\textrm{data}^2}\\
c_\textrm{in}(\sigma) = 1/\sqrt{\sigma^2 + \sigma_\textrm{data}^2}\\
c_\textrm{noise}(\sigma)=\frac 14 \log\sigma\\
c_\textrm{var}(\sigma) =  \sigma\sqrt{1 + \frac{\sigma_\textrm{data}^2}{\sigma^2+\sigma_\textrm{data}^2}}
\end{cases}
$$

we can now plug all of this in the original loss formula

$$
L = \frac{1}{2} \left[\log \sigma_\phi^2 + \frac{\sigma^2 + (\mu_\theta - x_0)^2}{\sigma_\phi^2} + \log(2\pi) \right]
$$

Substituiting this into the original formula of the loss we get the new numerically stable loss for $\sigma\to 0$ and $\sigma \to \infty$.

$$
L = G_\phi +e^{-G_\phi}\left[1+ \frac{\sigma^2_\textrm{data}}{\sigma^2 + 2\sigma^2_\textrm{data}}\left(\left\| F_{\theta} - \frac{\sigma x_0 - \sigma_\textrm{data}^2 \epsilon}{\sigma_\textrm{data}\sqrt{\sigma^2 + \sigma_\textrm{data}^2}} \right\|^2-1\right)\right]
$$

> Hint
> If you want to verify this formula, first substituite $\sigma_\phi$ and then substituite $\mu_\theta$

Using this parametrization we can re-write the log prob.

$$
\log q(x|\tilde x) =- \frac{1}{2} \left[G_\phi + e^{-G_\phi} \left( \frac{\sigma_\textrm{data}^2}{\sigma^2 + 2\sigma_\textrm{data}^2} \right) \left\| F_{\theta} - \frac{x - c_\textrm{skip} \cdot \tilde x}{c_\textrm{out}} \right\|^2+ \log(2\pi c_\textrm{var}^2)\right]
$$

As you can see, this formulation is numerically stable for all values of $\sigma$
