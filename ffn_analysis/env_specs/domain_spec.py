from cmx import doc

all_envs = """
Acrobot-swingup
Quadruped-run
Quadruped-walk
Humanoid-run
Finger-turn_hard
Walker-run
Cheetah-run
Hopper-hop
""".strip().split("\n")

if __name__ == "__main__":
    import gym

    with doc:
        for env_name in all_envs:
            env = gym.make(f"dmc:{env_name}-v1")
            doc.print(env_name, env.observation_space, env.action_space, sep="|")
