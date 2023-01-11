from pathlib import Path

from params_proto.hyper import Sweep

from sac_dennis_rff.config import Args, Actor, Critic, Agent
from ffn_analysis import RUN

with Sweep(RUN, Args, Actor, Critic, Agent) as sweep:
    Args.dmc = True
    Args.checkpoint_root = "gs://ge-data-improbable/checkpoints"
    Args.save_final_replay_buffer = True

    RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"
    Agent.use_rff = True
    Agent.learnable_temperature = True

    # Args.env_name = 'dmc:Walker-run-v1'
    # Agent.actor_fourier_features = 960
    # Agent.critic_fourier_features = 1200
    Args.env_name = 'dmc:Quadruped-run-v1'
    Agent.actor_fourier_features = 3120
    Agent.critic_fourier_features = 3600

    Args.train_frames = 1_000_000
    Agent.scale = 0.0001


    with sweep.product:
        Args.seed = [100, 200, 300, 400, 500]
        with sweep.zip: # Maintain the update frequency ratio
            # Agent.actor_update_frequency = [2, 3, 4,] # Default is 1 (i.e. 1 update per freq env steps)
            # Agent.critic_update_frequency = [2, 3, 4,] # Default is 1
            # Agent.critic_target_update_frequency = [4, 6, 8,] # Default is 2
            Agent.actor_update_frequency = [5, 6] # Default is 1 (i.e. 1 update per freq env steps)
            Agent.critic_update_frequency = [5, 6] # Default is 1
            Agent.critic_target_update_frequency = [10, 12] # Default is 2c


@sweep.each
def tail(RUN, Args, Actor, Critic, Agent, *_):
    if Agent.learnable_temperature:
        suffix = 'alpha_tune'
    else:
        suffix = f'alpha_fixed-{Agent.init_temperature}'

    RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"{Args.env_name.split(':')[-1][:-3]}/{suffix}/scale-{Agent.scale}/update_freq-{Agent.actor_update_frequency }/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")