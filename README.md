# Diffusion Policy Engine (DPE) – minimal demo

**Idea:** turn diffusion from an *artist* into a *scientist*.  
Instead of only denoising, each step nudges samples toward **higher reward**.

In this toy demo, we learn a conditional denoiser for a small vector `p` and add a **reward-guided** term to the loss.  
We use a differentiable reward (a context-shifted Rosenbrock function) so training is simple and stable.

---

## TL;DR training objective

Standard denoising loss +
```python
loss = mse(eps_hat, noise)  -  lambda_R * R(x, p0_pred)
