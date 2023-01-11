from pathlib import Path

from params_proto.neo_hyper import Sweep

from sac_dennis_rff.config import Args, Actor, Critic, Agent
from ffn_analysis import RUN

with Sweep(RUN, Args, Actor, Critic, Agent) as sweep:
    Args.dmc = True
    Args.checkpoint_root = "gs://ge-data-improbable/checkpoints"
    Args.save_final_replay_buffer = True

    RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"
    Agent.use_rff = True
    Agent.learnable_temperature = True

    Args.train_frames = 1_000_000


    with sweep.product:
        Args.seed = [100, 200, 300, 400, 500]
        with sweep.zip: # Maintain the update frequency ratio
            Args.env_name = ['dmc:Walker-run-v1', 'dmc:Quadruped-run-v1']
            Agent.actor_fourier_features = [960, 3120]
            Agent.critic_fourier_features = [1200, 3600]
            Agent.scale = [0.001, 0.0001]
            Agent.actor_update_frequency = [4, 6]
            Agent.critic_update_frequency = [4, 6]
            Agent.critic_target_update_frequency = [8, 12]

@sweep.each
def tail(RUN, Args, Actor, Critic, Agent, *_):
    if Agent.learnable_temperature:
        suffix = 'alpha_tune'
    else:
        suffix = f'alpha_fixed-{Agent.init_temperature}'

    RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"{Args.env_name.split(':')[-1][:-3]}/{suffix}/scale-{Agent.scale}/update_freq-{Agent.actor_update_frequency }/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")