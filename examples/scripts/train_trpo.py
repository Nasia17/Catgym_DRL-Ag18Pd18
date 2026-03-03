#!/usr/bin/env python3
"""Standalone TRPO training script for CatGym."""

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import gym
import gym.wrappers
import tensorflow as tf
import tensorforce
from tensorforce.agents import Agent
from tensorforce.agents.agent import TensorforceJSONEncoder
from tensorforce.execution import Runner

# Ensure repository-local imports work without hard-coded system paths.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from surface_seg.envs.catgym_env import MCSEnv
from surface_seg.utils.callback_new import Callback

SEED = 30
TIMESTEPS = 500
SAVE_DIR = './results/trpo_run'
NUM_PARALLEL = 32
THERMAL_THRESHOLD = 3
NUM_EPISODES = NUM_PARALLEL * 300


def setup_env(recording=True):
    mcs_gym = MCSEnv(
        observation_fingerprints=True,
        observation_forces=True,
        permute_seed=60,
        save_dir=SAVE_DIR,
        timesteps=TIMESTEPS,
        thermal_threshold=THERMAL_THRESHOLD,
        save_every_min=1,
        save_every=50,
        step_size=0.2,
    )

    if recording:
        mcs_gym = gym.wrappers.Monitor(
            mcs_gym,
            os.path.join(SAVE_DIR, 'vid'),
            force=True,
            video_callable=lambda episode_id: (episode_id + 1) % 50 == 0,
        )

    return tensorforce.environments.OpenAIGym(
        mcs_gym, max_episode_timesteps=TIMESTEPS, visualize=False
    )


def save_agent(agent):
    save_agent_dir = os.path.join(SAVE_DIR, 'saved_agent')
    os.makedirs(save_agent_dir, exist_ok=True)
    agent_name = 'agent'

    agent.model.save(
        directory=save_agent_dir,
        filename=agent_name,
        format='tensorflow',
        append=None,
    )

    spec_path = os.path.join(save_agent_dir, agent_name + '.json')
    try:
        with open(spec_path, 'w') as fp:
            spec = OrderedDict(agent.spec)
            spec['internals'] = agent.internals_spec
            spec['initial_internals'] = agent.initial_internals()
            json.dump(obj=spec, fp=fp, cls=TensorforceJSONEncoder)
    except BaseException:
        try:
            with open(spec_path, 'w') as fp:
                spec = OrderedDict()
                spec['states'] = agent.spec['states']
                spec['actions'] = agent.spec['actions']
                spec['internals'] = agent.internals_spec
                spec['initial_internals'] = agent.initial_internals()
                json.dump(obj=spec, fp=fp, cls=TensorforceJSONEncoder)
        except BaseException:
            if os.path.exists(spec_path):
                os.remove(spec_path)
            print('Agent saving failed')


def main():
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('MKL_DEBUG_CPU_TYPE', '5')

    tf.random.set_seed(SEED)

    env = setup_env().environment.env
    print(f'Initial energy: {env.initial_energy}')
    print(f'Thermal energy: {env.thermal_energy}')
    print(f'{THERMAL_THRESHOLD}KT: {THERMAL_THRESHOLD * env.thermal_energy}')

    agent = Agent.create(
        agent=dict(type='trpo'),
        environment=setup_env(recording=False),
        batch_size=1,
        learning_rate=1e-4,
        memory=50000,
        max_episode_timesteps=TIMESTEPS,
        exploration=dict(
            type='decaying',
            unit='timesteps',
            decay='exponential',
            initial_value=0.8,
            decay_steps=50000,
            decay_rate=0.5,
        ),
        parallel_interactions=NUM_PARALLEL,
    )

    print('Agent spec:', agent.spec)

    callback = Callback(NUM_EPISODES, SAVE_DIR)

    def callback_with_progress(runner, parallel):
        episode_reward = runner.episode_rewards[-1] if runner.episode_rewards else None
        print(
            f'Training progress: episode {runner.episodes}/{NUM_EPISODES} | '
            f'Episode reward: {episode_reward}'
        )
        return callback.episode_finish(runner, parallel)

    runner = Runner(
        agent=agent,
        environments=[setup_env(recording=False) for _ in range(NUM_PARALLEL)],
        num_parallel=NUM_PARALLEL,
        remote='multiprocessing',
        max_episode_timesteps=TIMESTEPS,
    )

    print('Starting TRPO training...')
    runner.run(
        num_episodes=NUM_EPISODES,
        callback=callback_with_progress,
        callback_episode_frequency=1,
    )
    runner.close()
    print('Training complete. Saving agent...')
    save_agent(agent)
    print(f'Artifacts saved under {SAVE_DIR}')


if __name__ == '__main__':
    main()
