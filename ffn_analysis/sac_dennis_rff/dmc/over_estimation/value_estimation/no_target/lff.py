from pathlib import Path

from params_proto.hyper import Sweep

from sac_dennis_rff.config import Args, Actor, Critic, Agent
from ffn_analysis import RUN

with Sweep(RUN, Args, Actor, Critic, Agent) as sweep:
    Args.dmc = True
    Args.checkpoint_root = "gs://ge-data-improbable/checkpoints"
    Args.save_final_replay_buffer = True
    Args.collect_value_estimate = True
    Args.value_estimate_k = 1500

    RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"
    Agent.use_rff = True
    Agent.learnable_temperature = True
    Agent.critic_tau = None

    with sweep.product:
        Args.seed = [100, 200, 300, 400, 500]

        with sweep.zip:
            Args.env_name = ['dmc:Cheetah-run-v1', 'dmc:Acrobot-swingup-v1',
                             'dmc:Hopper-hop-v1',
                             'dmc:Quadruped-walk-v1', 'dmc:Quadruped-run-v1',
                             'dmc:Humanoid-run-v1', 'dmc:Finger-turn_hard-v1', ]
            Agent.scale = [0.0001, 0.003, 0.003, 0.0001,
                            0.0003, 0.001, 0.001,]
            Agent.actor_fourier_features = [680, 240, 600, 3120, 3120, 2680, 480, ]
            Agent.critic_fourier_features = [920, 280, 760, 3600, 3600, 3520, 560, ]
            Args.train_frames = [1_000_000, 1_000_000,
                                 1_000_000,
                                 1_000_000, 1_000_000,
                                 2_000_000, 1_000_000, ]

@sweep.each
def tail(RUN, Args, Actor, Critic, Agent, *_):
    if Agent.learnable_temperature:
        suffix = 'alpha_tune'
    else:
        suffix = f'alpha_fixed-{Agent.init_temperature}'

    RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"{Args.env_name.split(':')[-1][:-3]}/{suffix}/scale-{Agent.scale}/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")