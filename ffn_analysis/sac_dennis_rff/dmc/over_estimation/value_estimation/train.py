if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from ffn_analysis.baselines import RUN
    import jaynes
    from sac_dennis_rff.sac import train
    from sac_dennis_rff.config import Args, Actor, Critic, Agent
    from params_proto.hyper import Sweep

    sweep = Sweep(RUN, Args, Actor, Critic, Agent).load("lff.jsonl")

    jaynes.config('supercloud')

    for i, kwargs in enumerate(sweep):
        with logger.Prefix(RUN.prefix):
            status = logger.read_params('job.status')
            if status != 'completed':
                needs_relaunch(RUN.prefix)
                thunk = instr(train, **kwargs)
                jaynes.run(thunk)

    jaynes.listen()
