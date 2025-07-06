import math


def learning_rate_scheduler(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    This function implements the schedule in three parts:
    1. Warm-up (t < T_w): Linearly increase LR.
       alpha_t = (t / T_w) * alpha_max
    2. Cosine Annealing (T_w <= t <= T_c): Decay LR from alpha_max to alpha_min.
       alpha_t = alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + cos(pi * (t - T_w) / (T_c - T_w)))
    3. Post-annealing (t > T_c): LR is constant at alpha_min.
       alpha_t = alpha_min

    Args:
        it (int): Iteration number to get learning rate for (t).
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        float: Learning rate at the given iteration under the specified schedule.
    """
    # 1. Warm-up phase
    if it < warmup_iters:
        # Linearly increase the learning rate from near 0 to max_learning_rate
        return max_learning_rate * it / warmup_iters

    # 2. Cosine annealing phase
    # This condition also covers the case where it > cosine_cycle_iters, which will be handled by the else block.
    if it <= cosine_cycle_iters:
        # Ensure we don't divide by zero if warmup_iters equals cosine_cycle_iters
        if warmup_iters == cosine_cycle_iters:
            return min_learning_rate

        # Calculate the progress within the cosine cycle (a value from 0 to 1)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

        # Calculate the cosine decay multiplier (from 1 down to 0)
        # 0.5 * (1 + cos(pi * progress)) will be 1 at progress=0 and 0 at progress=1
        decay_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Calculate the learning rate for the current iteration
        lr_range = max_learning_rate - min_learning_rate
        return min_learning_rate + decay_multiplier * lr_range

    # 3. Post-annealing phase
    # This is executed if it > cosine_cycle_iters
    else:
        return min_learning_rate