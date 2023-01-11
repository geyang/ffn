from pathlib import Path

from params_proto.neo_hyper import Sweep

from sac_dennis_rff.config import Args, Actor, Critic, Agent
from ffn_analysis import RUN

with Sweep(RUN, Args, Actor, Critic, Agent) as sweep:
    Args.dmc = True
    Args.checkpoint_root = "gs://ge-data-improbable/checkpoints"
    Args.save_final_replay_buffer = True

    RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"

    Actor.hidden_layers = 3
    Critic.hidden_layers = 3

    with sweep.product:
        Args.seed = [100, 200, 300, 400, 500]
        with sweep.zip:
            Args.env_name = ['dmc:Cheetah-run-v1', 'dmc:Acrobot-swingup-v1',
                             'dmc:Hopper-hop-v1',
                             'dmc:Quadruped-walk-v1',
                             'dmc:Humanoid-run-v1', 'dmc:Finger-turn_hard-v1',
                             'dmc:Walker-run-v1']
            Args.train_frames = [1_000_000, 1_000_000,
                                 1_000_000,
                                 1_000_000,
                                 2_000_000, 500_000,
                                 1_000_000, ]


@sweep.each
def tail(RUN, Args, Actor, Critic, Agent, *_):
    RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"{Args.env_name.split(':')[-1][:-3]}/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")
