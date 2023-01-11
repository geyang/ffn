from sac_dennis_rff.over_estimation import main


if __name__ == '__main__':
    import jaynes
    from ml_logger import logger, RUN, instr, needs_relaunch
    from sac_dennis_rff.config import Args

    exp_root = "/model-free/model-free/sac_dennis_rff/dmc/over_estimation/value_estimation"
    with logger.Prefix(exp_root):
        exp_prefixes = logger.glob("**/*0")

    env_names = ['Quadruped-walk', 'Quadruped-run', 'Hopper-hop']

    # jaynes.config('local' if RUN.debug else 'visiongpu-docker')
    jaynes.config('supercloud ')
    # jaynes.config('supercloud')

    for env_name in env_names:

        prefixes = list(filter(lambda p: (env_name in p) and ('no_target' in p), exp_prefixes))
        print(prefixes)

        for i, prefix in enumerate(prefixes):
            RUN.prefix = exp_root + "/" + prefix + "/analysis"
            # with logger.Prefix(RUN.prefix):
            #     try:
            #         status = logger.read_params('job.status')
            #     except FileNotFoundError:
            #         status = 'errored'
            #
            #     if status != 'completed':
            logger.print(RUN.prefix, color='green')
            checkpoint_root = "gs://ge-data-improbable/checkpoints"

            kwargs = {'Args.env_name': f'dmc:{env_name}-v1',
                      'Args.checkpoint_root': checkpoint_root,
                      'Args.dmc': True, }

            thunk = instr(main, **kwargs)
            jaynes.run(thunk)

    jaynes.listen()
