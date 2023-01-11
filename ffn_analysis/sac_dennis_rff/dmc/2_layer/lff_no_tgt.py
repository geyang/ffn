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

    with sweep.product:
        Args.seed = [100, 200, 300, 400, 500]
        Agent.scale = [0.0001, 0.0003, 0.001, 0.003]

        with sweep.zip:
            Args.env_name = ['dmc:Humanoid-run-v1',]
            Agent.actor_fourier_features = [2680]
            Agent.critic_fourier_features = [3520]
            Args.train_frames = [2_000_000]

@sweep.each
def tail(RUN, Args, Actor, Critic, Agent, *_):
    RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"{Args.env_name.split(':')[-1][:-3]}/scale-{Agent.scale}/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")