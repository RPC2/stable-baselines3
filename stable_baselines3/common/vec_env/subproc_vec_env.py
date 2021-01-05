import multiprocessing as mp
from collections import OrderedDict, defaultdict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrappers: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    envs = [env_fn() for env_fn in env_fn_wrappers.var]
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observations = []
                rewards = []
                dones = []
                infos = []
                for env in envs:
                    observation, reward, done, info = env.step(data)
                    if done:
                        # save final observation where user can get it, then reset
                        info["terminal_observation"] = observation
                        observation = env.reset()
                    observations.append(observation)
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)
                remote.send((observations, rewards, dones, infos))
            elif cmd == "seed":
                remote.send([env.seed(data) for env in envs])
            elif cmd == "reset":
                remote.send([env.reset() for env in envs])
            elif cmd == "render":
                remote.send([env.render(data) for env in envs])
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send([(env.observation_space, env.action_space) for env in envs])
            elif cmd == "env_method":
                results = []
                for env_idx in data[0]:
                    env = envs[env_idx]
                    method = getattr(env, data[1])
                    results.append(method(*data[2], **data[3]))
                remote.send(results)
            elif cmd == "get_attr":
                remote.send([getattr(envs[env_idx], data[1]) for env_idx in data[0]])
            elif cmd == "set_attr":
                remote.send([setattr(envs[env_idx], data[1], data[2]) for env_idx in data[0]])
            elif cmd == "is_wrapped":
                remote.send([is_wrapped(envs[env_idx], data[1]) for env_idx in data[0]])
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing environments to different processes and
    thus allowing a significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of workers should not exceed the number of logical
    cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param n_workers: Number of worker processes to spawn and distribute environments among. If None, we spawn one process per
           environment.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None,
                 n_workers: Optional[int] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        if n_workers is None:
            n_workers = n_envs
        if n_envs % n_workers != 0 or not (0 < n_workers <= n_envs):
            raise ValueError(f"n_envs (={n_envs}) must be a positive multiple of n_workers (={n_workers})")
        # this should support uneven partitions of environments, but uneven
        # partitioning will be bad for core utilisation
        env_fns_grouped = np.array_split(np.array(env_fns, dtype='object'), n_workers)
        self._remote_for_env = {}
        for group_idx, group in enumerate(env_fns_grouped):
            for sub_idx in range(len(group)):
                self._remote_for_env[len(self._remote_for_env)] = (group_idx, sub_idx)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_workers)])
        self.processes = []
        for work_remote, remote, env_fn_list in zip(self.work_remotes, self.remotes, env_fns_grouped):
            args = (work_remote, remote, CloudpickleWrapper(env_fn_list))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()[0]
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # each remote returns a tuple of lists, like: ([ob1, obs2, …], [rew1,
        # rew2, …], [done1, done2, …], [info1, info2, …])
        obs, rews, dones, infos = [sum(l, []) for l in zip(*results)]
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return sum((remote.recv() for remote in self.remotes), [])

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = sum((remote.recv() for remote in self.remotes), [])
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = sum((pipe.recv() for pipe in self.remotes), [])
        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes_and_inds = self._get_target_remotes(indices)
        for remote, remote_inds in target_remotes_and_inds:
            remote.send(("get_attr", (remote_inds, attr_name)))
        return sum((remote.recv() for remote, _ in target_remotes_and_inds), [])

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes_and_inds = self._get_target_remotes(indices)
        for remote, remote_inds in target_remotes_and_inds:
            remote.send(("set_attr", (remote_inds, attr_name, value)))
        for remote, _ in target_remotes_and_inds:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes_and_inds = self._get_target_remotes(indices)
        for remote, remote_inds in target_remotes_and_inds:
            remote.send(("env_method", (remote_inds, method_name, method_args, method_kwargs)))
        return sum((remote.recv() for remote, _ in target_remotes_and_inds), [])

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes_and_inds = self._get_target_remotes(indices)
        for remote, remote_inds in target_remotes_and_inds:
            remote.send(("is_wrapped", (remote_inds, wrapper_class)))
        return sum((remote.recv() for remote, _ in target_remotes_and_inds), [])

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        sub_inds_by_remote = defaultdict(list)
        for ind in indices:
            remote_ind, sub_index = self._remote_for_env[ind]
            sub_inds_by_remote.setdefault(remote_ind, []).append(sub_index)
        return [(self.remotes[i], sub_indices) for i, sub_indices in sub_inds_by_remote.items()]


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: gym.spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
