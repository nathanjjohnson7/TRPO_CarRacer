# Tust Region Policy Optimzation - CarRacer
Here, we have implemented Trust Region Policy Optimization (TRPO) for a modified version of the gymnasium car racing environment.

Paper: https://arxiv.org/pdf/1502.05477

TRPO is a policy gradient method with a trust region. It consists of an actor and critic. The critic is updated using gradient descent. We try to minimize the loss between the critic's value function predictions and the discounted rewards from the epsiode. The actor is updated by solving the lagrangian dual for a constrained optimization problem where we attempt to maximize the model’s performance in the environment while the KL divergence between the previous model's weights and new model weights don't exceed a certain threshold. This constraint defines the trust region, preventing overly large policy updates that could destabilize learning. We provide a detailed derivation of the actor update below. 

Due to computational constraints, we have modified the environment to output an 11-dimensional feature vector at each step. The features are: an off-track flag, speed, acceleration, angular velocity, angular acceleration, sine of the car's angle relative to the road trajectory, cosine of the car's angle relative to the road trajectory, distance to the left edge of the track, distance to the center of the track, distance to the right edge of the track and an unused placeholder (always zero).
<img width="2400" height="1500" alt="carracing_trpo_training_score_graph" src="https://github.com/user-attachments/assets/bad0d269-2f86-49ef-9203-857ac67d3298" />

The model was only trained on one track (seed=10) due to computational constraints, but it generalizes reasonably well to other tracks. Furthermore, during training we ended episodes as soon as the car left the track, however, as shown in demo3_seed_1, the car is able to correct itself when it leaves the track, despite never experiencing being outside the track during training.



https://github.com/user-attachments/assets/b6a0b33a-d7a1-4dcf-af1f-a2a572bc3af4


https://github.com/user-attachments/assets/0d62edd4-40ca-4e86-8e7a-15956ea93115


https://github.com/user-attachments/assets/bf7ff550-ac5b-43e1-bd35-c02ec4a107a4

## Update Rule Derivation

The TRPO paper states that 1/lambda is "typically treated as an algorithm parameter". In this implementation, we analytically calculate the exact value for lambda, instead.

According to the paper:

$$\theta_{next}= \arg\max_{\theta} L_{\theta_{old}}(\theta) \quad s.t. \quad D_{kl}(\theta||\theta _k) < \delta$$

We use a taylor approximation for both the objective, L, and the KL divergence term:

$$L_{\theta_{old}}(\theta) \approx g^T(\theta-\theta _k) $$

where g is the gradient of the objective. The 0th term of the taylor approximation is 0 since the objective term is the advantage of $$\theta$$ over $$\theta_{old}$$, and the advantage of $$\theta_{old}$$ over $$\theta_{old}$$  is 0.

$$D_{kl}(\theta||\theta _k) \approx \frac{1}{2} (\theta-\theta _k)^T H (\theta-\theta _k)$$

where H is the hessian of the KL divergence. The 0th and 1st terms of the above taylor approxmation are zero since the kl divergence of the policy with itself is 0 and, furthermore, the gradient is 0 at this point since KL divergence is always non-negative.

The Lagrangian formulation:

$$\mathcal{L} = -g^T(\theta-\theta _k) + \lambda(\frac{1}{2} (\theta-\theta _k)^T H (\theta-\theta _k) - \delta)$$
$$\frac{\partial \mathcal{L}}{\partial \theta} = -g + \lambda H(\theta-\theta _k) = 0$$
$$\theta = \frac{1}{\lambda} H^{-1}g + \theta _k$$

Substitute into Lagrangian to get lambda:

$$L = -\frac{1}{\lambda}g^TH^{-1}g + \frac{1}{2\lambda}g^TH^{-1}g - \lambda \delta$$
$$L = -\frac{1}{2\lambda}g^TH^{-1}g  - \lambda \delta$$
$$\frac{\partial \mathcal{L}}{\partial \lambda} = \frac{1}{2\lambda ^2}g^TH^{-1}g - \delta = 0$$
$$\lambda = \sqrt{\frac{g^TH^{-1}g}{2\delta}}$$

Final Update Rule:

$$\theta = \theta _k +\sqrt{\frac{2\delta}{g^TH^{-1}g}} H^{-1}g$$
