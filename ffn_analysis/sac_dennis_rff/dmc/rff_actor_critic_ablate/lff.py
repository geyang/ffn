from pathlib import Path

from params_proto.hyper import Sweep

from sac_dennis_rff.config import Args, Actor, Critic, Agent
from ffn_analysis import RUN

with Sweep(RUN, Args, Actor, Critic, Agent) as sweep:
    Args.dmc = True
    Args.checkpoint_root = "gs://ge-data-improbable/checkpoints"
    Args.save_final_replay_buffer = True

    RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"
    Agent.learnable_temperature = True

    with sweep.product:
        Args.seed = [100, 200, 300, 400, 500]

        with sweep.zip:
            Args.env_name = ['dmc:Walker-run-v1', 'dmc:Walker-run-v1', 'dmc:Quadruped-run-v1', 'dmc:Quadruped-run-v1']

            Agent.use_critic_rff = [True, False, True, False]
            Agent.use_actor_rff = [False, True, False, True]

            Critic.hidden_layers = [2, 3, 2, 3]
            Actor.hidden_layers = [3, 2, 3, 2]

            Agent.critic_fourier_features = [1200, 1200, 3600, 3600]
            Agent.actor_fourier_features = [960, 960, 3120, 3120]

            Agent.critic_scale = [0.001, 0.001, 0.0001, 0.0001]
            Agent.actor_scale = [0.001, 0.001, 0.0001, 0.0001]



@sweep.each
def tail(RUN, Args, Actor, Critic, Agent, *_):
    if Agent.use_critic_rff:
        assert not Agent.use_actor_rff
        RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"{Args.env_name.split(':')[-1][:-3]}/critic_rff/scale-{Agent.critic_scale}/{Args.seed}")
    else:
        assert Agent.use_actor_rff
        assert not Agent.use_critic_rff
        RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                          job_name=f"{Args.env_name.split(':')[-1][:-3]}/actor_rff/scale-{Agent.actor_scale}/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")