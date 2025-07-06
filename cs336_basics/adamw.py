import math

import torch


class AdamW(torch.optim.Optimizer):
    """
    Implements the AdamW algorithm as described in the assignment prompt, which is
    based on Algorithm 2 from Loshchilov and Hutter [2019].
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        Initializes the AdamW optimizer.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): learning rate α (default: 1e-3).
            betas (Tuple[float, float], optional): coefficients (β1, β2) used for computing
                running averages of gradient and its square (default: (0.9, 0.999)).
            eps (float, optional): term ϵ added to the denominator to improve
                numerical stability (default: 1e-8).
            weight_decay (float, optional): weight decay rate λ (default: 1e-2).
        """
        # Input validation
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Retrieve hyperparameters for the current parameter group
            lr = group['lr']  # α
            beta1, beta2 = group['betas']
            eps = group['eps']  # ϵ
            weight_decay = group['weight_decay']  # λ

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('This AdamW implementation does not support sparse gradients.')

                state = self.state[p]

                # Lazy State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # m: Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v: Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Algorithm step: t starts at 1
                state['step'] += 1
                t = state['step']

                # Algorithm step: m ← β1 m + (1 − β1)g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Algorithm step: v ← β2 v + (1 − β2)g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Algorithm step: αt ← α * sqrt(1-(β2)^t) / (1-(β1)^t)
                # Compute bias correction factors
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # Compute the bias-corrected step size for the gradient update
                # This corresponds to α_t in the prompt's algorithm
                alpha_t = lr * (math.sqrt(bias_correction2) / bias_correction1)

                # Denominator for the main update rule: sqrt(v) + ϵ
                denom = exp_avg_sq.sqrt().add(eps)

                # Algorithm step: θ ← θ − αt * (m / (sqrt(v)+ϵ))
                p.addcdiv_(exp_avg, denom, value=-alpha_t)

                # Algorithm step: θ ← θ − αλθ (Apply weight decay)
                # This is the decoupled weight decay. Note it uses the base learning rate `lr` (α)
                # and is applied to the parameter `p` *after* the main Adam update.
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss