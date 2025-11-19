    
    # number of steps to update the policy
    steps_per_epoch: 20000
    # number of iterations to update the policy
    update_iters: 10
    # batch size for each iteration
    batch_size: 128
    # target kl divergence
    target_kl: 0.01
    # entropy coefficient
    entropy_coef: 0.0
    # normalize reward
    reward_normalize: False
    # normalize cost
    cost_normalize: False
    # normalize observation
    obs_normalize: True
    # early stop when kl divergence is bigger than target kl
    kl_early_stop: False
    # use max gradient norm
    use_max_grad_norm: True
    # max gradient norm
    max_grad_norm: 40.0
    # use critic norm
    use_critic_norm: True
    # critic norm coefficient
    critic_norm_coef: 0.001
    # reward discount factor
    gamma: 0.99
    # cost discount factor
    cost_gamma: 0.99
    # lambda for gae
    lam: 0.95
    # lambda for cost gae
    lam_c: 0.95
    # advantage estimation method, options: gae, retrace
    adv_estimation_method: gae
    # standardize reward advantage
    standardized_rew_adv: True
    # standardize cost advantage
    standardized_cost_adv: True
    # penalty coefficient
    penalty_coef: 0.0
    # use cost
    use_cost: True
    # Damping value for conjugate gradient
    cg_damping: 0.1
    # Number of conjugate gradient iterations
    cg_iters: 15
    # Subsampled observation
    fvp_obs: None
    # The sub-sampling rate of the observation
    fvp_sample_freq: 1