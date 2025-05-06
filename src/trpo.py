import torch
import numpy as np
import copy
from torch.distributions import Normal
import wandb  


class TRPO:
    def __init__(self, policy, env, discriminator, max_kl=0.01, damping=0.1):
        self.policy = policy
        self.env = env
        self.discriminator = discriminator
        self.max_kl = max_kl
        self.damping = damping
        self._old_policy = copy.deepcopy(policy)

    @staticmethod
    def flat_params(model):
        return torch.cat([param.data.view(-1) for param in model.parameters()])

    @staticmethod
    def set_flat_params(model, flat_params):
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind+flat_size].view(param.size()))
            prev_ind += flat_size

    def conjugate_gradient(self, Av_func, b, max_iter=10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)

        for _ in range(max_iter):
            Avp = Av_func(p)
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < 1e-10:
                break
        return x

    def fisher_vector_product(self, states, vector):
        self.policy.zero_grad()
        mean, std = self.policy(states)
        dist = Normal(mean, std)

        with torch.no_grad():
            old_mean, old_std = self._old_policy(states)
            old_dist = Normal(old_mean, old_std)

        kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        grad_vector_product = (flat_grad_kl * vector).sum()
        grad_grads = torch.autograd.grad(grad_vector_product, self.policy.parameters(), retain_graph=True)
        flat_grad_grad = torch.cat([grad.contiguous().view(-1) for grad in grad_grads])
        return flat_grad_grad + self.damping * vector

    def update_policy(self, trajectories):
        # Process trajectories
        states = torch.cat([t['states'] for t in trajectories])
        actions = torch.cat([t['actions'] for t in trajectories])

        # Store old policy parameters
        self._old_policy.load_state_dict(self.policy.state_dict())

        # Calculate advantages using discriminator
        with torch.no_grad():
            d_real = self.discriminator(states, actions)
            advantages = torch.log(d_real + 1e-8).squeeze(1)

        # Calculate old log probs using stored policy
        with torch.no_grad():
            old_mean, old_std = self._old_policy(states)
            old_dist = Normal(old_mean, old_std)
            old_log_probs = old_dist.log_prob(actions).sum(-1)

        # Surrogate loss function
        def surrogate_loss():
            mean, std = self.policy(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            return (ratio * advantages).mean()

        # Calculate initial loss and gradient
        loss = surrogate_loss()
        self.policy.zero_grad()
        loss.backward()
        flat_grad = torch.cat([param.grad.view(-1) for param in self.policy.parameters()])


        # Calculate natural gradient direction
        Fvp = lambda v: self.fisher_vector_product(states, v)
        step_dir = self.conjugate_gradient(Fvp, flat_grad)

        # Calculate maximum step size
        shs = 0.5 * (step_dir * Fvp(step_dir)).sum()
        step_size = torch.sqrt(self.max_kl / (shs + 1e-8))
        full_step = step_size * step_dir

        # Backtracking line search
        old_params = self.flat_params(self.policy)
        success = False

        for alpha in [0.5**i for i in range(10)]:  # Try up to 10 backtracking steps
            new_params = old_params + alpha * full_step
            self.set_flat_params(self.policy, new_params)

            with torch.no_grad():
                new_loss = surrogate_loss()
                mean, std = self.policy(states)
                dist = Normal(mean, std)
                kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()

            if new_loss > loss and kl <= self.max_kl:
                success = True
                break

        if not success:
            self.set_flat_params(self.policy, old_params)
            print("TRPO update rejected")
            # Log TRPO rejection to wandb
            wandb.log({"trpo_rejected": 0})
        else:
            # Log TRPO success to wandb
            wandb.log({"trpo_sucess": 1})

        return success