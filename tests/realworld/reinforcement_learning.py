"""Reinforcement learning patterns: Q-learning, policy gradient, advantage estimation."""

import numpy as np


def huber_loss_td_error(td_error, delta=1.0):
    abs_error = np.abs(td_error)
    quadratic = np.minimum(abs_error, delta)
    linear = np.subtract(abs_error, quadratic)
    return np.add(np.multiply(0.5, np.power(quadratic, 2.0)), linear)


def q_update_target(target_q, reward, next_target_q, gamma=0.99):
    return np.add(reward, np.multiply(gamma, np.max(next_target_q)))


def q_update_online(q_value, target_value, lr=0.001):
    td_error = np.subtract(target_value, q_value)
    return np.add(q_value, np.multiply(lr, td_error))


def policy_gradient_loss(log_probs, advantages):
    return np.negative(np.mean(np.multiply(log_probs, advantages)))


def generalized_advantage_estimator(rewards, values, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    timesteps = len(rewards)
    for t in reversed(range(timesteps)):
        if t == timesteps - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = np.add(rewards[t], np.multiply(gamma, np.subtract(next_value, values[t])))
        last_gae = np.add(delta, np.multiply(np.multiply(gamma, lam), last_gae))
        advantages[t] = last_gae
    returns = np.add(advantages, values)
    return advantages, returns


def entropy_bonus(probs):
    return np.negative(np.sum(np.multiply(probs, np.log(np.add(probs, 1e-8)))))


def clip_ratio(ratio, epsilon=0.2):
    clipped = np.clip(ratio, np.subtract(1.0, epsilon), np.add(1.0, epsilon))
    return np.minimum(np.multiply(ratio, np.ones_like(ratio)), clipped)


def ppo_loss(ratio, advantages, epsilon=0.2):
    clipped = clip_ratio(ratio, epsilon)
    surr1 = np.multiply(ratio, advantages)
    surr2 = np.multiply(clipped, advantages)
    return np.negative(np.mean(np.minimum(surr1, surr2)))


def soft_q_update(q_values, log_probs, alpha=0.2):
    exp_values = np.exp(np.divide(np.subtract(q_values, log_probs), alpha))
    return np.log(np.add(exp_values, 1e-8))


def prioritized_replay_weights(td_errors, alpha=0.6, beta=0.4):
    priorities = np.add(np.abs(td_errors), 1e-6)
    probabilities = np.power(priorities, alpha)
    weights = np.power(np.multiply(probabilities, 1.0 / np.sum(probabilities)), beta)
    return np.divide(weights, np.max(weights))
