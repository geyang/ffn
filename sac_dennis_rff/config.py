import torch

from params_proto import ParamsProto, PrefixProto, Proto


class Args(PrefixProto):
    env_name = "Ant-v2"
    dmc = False
    # train
    train_frames = 1_000_000
    seed_frames = 5_000
    replay_buffer_size = 1_000_000
    seed = 1
    # eval
    eval_frequency = 10000
    eval_episodes = 30
    # misc
    log_frequency_step = 10000
    log_save_tb = True
    checkpoint_freq = 30000
    save_video = False
    save_final_replay_buffer = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Normalization constants
    normalize_obs = False
    obs_bias = None
    obs_scale = None
    # observation
    from_pixels = False
    image_size = 84
    image_pad = 4
    frame_stack = 3

    report_rank = False

    checkpoint_root = Proto(env="$ML_LOGGER_BUCKET/checkpoints")

class Actor(PrefixProto):
    hidden_layers = 2
    hidden_features = 1024
    log_std_bounds = [-5, 2]


class Critic(PrefixProto):
    hidden_layers = 2
    hidden_features = 1024


class Agent(PrefixProto):
    lr = 1e-4
    batch_size = Proto(1024, help="please use a batch size of 512 to reproduce the results in the paper. "
                                 "However, with a smaller batch size it still works well.")
    discount = 0.99
    init_temperature = 0.1
    alpha_lr = 1e-4
    alpha_betas = [0.9, 0.999]
    actor_lr = 1e-4
    actor_betas = [0.9, 0.999]
    actor_update_frequency = 1
    critic_lr = 1e-4
    critic_betas = [0.9, 0.999]
    critic_tau = 0.005
    critic_target_update_frequency = 2
    learnable_temperature = True

    use_rff = False
    scale = None
    actor_fourier_features = None
    critic_fourier_features = None
