"""Generative model patterns: GANs, VAEs, diffusion models."""

import numpy as np


def leaky_relu(x, alpha=0.2):
    return np.where(np.greater(x, 0), x, np.multiply(alpha, x))


def discriminator_loss(real_scores, fake_scores):
    real_loss = np.mean(np.maximum(0, np.subtract(1.0, real_scores)))
    fake_loss = np.mean(np.maximum(0, np.add(1.0, fake_scores)))
    return np.add(real_loss, fake_loss)


def generator_loss(fake_scores):
    return np.mean(np.maximum(0, np.subtract(1.0, fake_scores)))


def wasserstein_loss(real_scores, fake_scores):
    return np.subtract(np.mean(fake_scores), np.mean(real_scores))


def gradient_penalty(real_data, fake_data, discriminator_fn, lambda_gp=10.0):
    alpha = np.random.uniform(0.0, 1.0, size=real_data.shape)
    interpolated = np.add(
        np.multiply(alpha, real_data), np.multiply(np.subtract(1.0, alpha), fake_data)
    )
    grad_output = np.ones(interpolated.shape)
    grad = np.subtract(discriminator_fn(interpolated), grad_output)
    penalty = np.mean(np.power(np.sqrt(np.sum(np.power(grad, 2), axis=1)), 2))
    return np.multiply(lambda_gp, penalty)


def vae_reconstruction_loss(x_reconstructed, x_original):
    return np.mean(np.power(np.subtract(x_reconstructed, x_original), 2))


def vae_kl_divergence(mu, logvar):
    return np.mean(
        np.add(np.multiply(-0.5, np.add(1.0, logvar)), np.multiply(-0.5, np.exp(logvar)))
    )


def vae_loss(x_reconstructed, x_original, mu, logvar):
    recon = vae_reconstruction_loss(x_reconstructed, x_original)
    kl = vae_kl_divergence(mu, logvar)
    return np.add(recon, kl)


def noise_schedule(t, beta_start=0.0001, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, t)
    alphas = np.subtract(1.0, betas)
    alpha_bar = np.cumprod(alphas)
    return betas, alphas, alpha_bar


def add_noise(x, alpha_bar_t):
    noise = np.random.standard_normal(x.shape)
    return np.add(
        np.multiply(np.sqrt(alpha_bar_t), x),
        np.multiply(np.sqrt(np.subtract(1.0, alpha_bar_t)), noise),
    )


def denoise_loss(x_noisy, x_clean, predicted_noise, alpha_bar_t):
    return np.mean(np.power(np.subtract(x_clean, x_noisy), 2))


def spectral_norm(weight, n_power_iterations=1):
    u = np.ones((weight.shape[1],))
    for _ in range(n_power_iterations):
        v = np.divide(
            np.matmul(weight.T, u),
            np.sqrt(np.add(np.sum(np.power(np.matmul(weight.T, u), 2)), 1e-12)),
        )
        u = np.divide(
            np.matmul(weight, v), np.sqrt(np.add(np.sum(np.power(np.matmul(weight, v), 2)), 1e-12))
        )
    sigma = np.sum(np.multiply(np.matmul(weight, u), v))
    return np.divide(weight, sigma)


def hinge_loss_d(real_scores, fake_scores):
    return np.add(
        np.mean(np.maximum(0, np.subtract(1.0, real_scores))),
        np.mean(np.maximum(0, np.add(1.0, fake_scores))),
    )


def hinge_loss_g(fake_scores):
    return np.mean(np.negative(fake_scores))
