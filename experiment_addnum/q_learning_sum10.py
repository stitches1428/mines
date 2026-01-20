"""
最简单的 Q-Learning 示例：学习找到两个自然数 (a, b) 使得 a + b = 10。

思路（两步决策）：
- 第 1 步选 a
- 第 2 步选 b
- 终止后根据 sum=a+b 与 10 的距离给奖励

另外也提供一种“成对动作”的算法（pair-action）：
- 把 (a,b) 作为一个整体动作，一次性选择
- 这是单步任务，等价于用 Q-learning 学一个带奖励的表格策略（类似多臂老虎机）

这样就可以用最基础的表格型 Q-Learning（Q-table）学出策略：
从 (t=0,sum=0) 出发，贪心选动作得到 (a,b)。

运行：
  python fun/q_learning_sum10.py
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    target: int = 10
    max_action: int = 10  # 动作空间：选择 0..max_action 的自然数


class SumToTargetEnv:
    """
    状态： (t, current_sum)
    - t=0 表示还没选任何数
    - t=1 表示已经选了第一个数
    - t=2 终止（已经选了两个数）
    动作：选一个自然数 x in [0, max_action]
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> tuple[int, int]:
        self.t = 0
        self.current_sum = 0
        return (self.t, self.current_sum)

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        if not (0 <= action <= self.cfg.max_action):
            raise ValueError(f"action must be in [0, {self.cfg.max_action}]")
        if self.t >= 2:
            raise RuntimeError("Episode already done, call reset().")

        self.current_sum += action
        self.t += 1

        done = self.t == 2
        reward = 0.0
        if done:
            # 最简单的奖励：越接近 target 越好，刚好等于 target 时最大
            reward = float(self.cfg.target - abs(self.current_sum - self.cfg.target))
            # reward 范围：[target - |sum-target|]，在动作上限为 10 时 sum∈[0,20] => reward∈[0,10]
        return (self.t, self.current_sum), reward, done


def state_to_index(t: int, s: int, sum_max: int) -> int:
    # 只需要 t=0,1 的 Q 值（t=2 终止态不做决策）
    # index = t * (sum_max+1) + s
    return t * (sum_max + 1) + s


def train_q_learning(
    *,
    episodes: int = 500,
    alpha: float = 0.15,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    seed: int = 0,
    cfg: EnvConfig = EnvConfig(),
) -> tuple[np.ndarray, SumToTargetEnv, int]:
    random.seed(seed)
    np.random.seed(seed)

    env = SumToTargetEnv(cfg)
    n_actions = cfg.max_action + 1

    # 两步最多 sum=2*max_action
    sum_max = 2 * cfg.max_action

    # Q-table 只存 t=0,1 两层
    n_states = 2 * (sum_max + 1)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def epsilon_by_episode(ep: int) -> float:
        # 线性退火
        frac = min(1.0, ep / max(1, episodes - 1))
        return eps_start + frac * (eps_end - eps_start)

    for ep in range(episodes):
        (t, s) = env.reset()
        eps = epsilon_by_episode(ep)

        done = False
        while not done:
            si = state_to_index(t, s, sum_max)
            if random.random() < eps:
                a = random.randint(0, cfg.max_action)
            else:
                a = int(np.argmax(Q[si]))

            (t2, s2), r, done = env.step(a)

            # t2=2 时终止：未来回报为 0
            target = r
            if not done:
                si2 = state_to_index(t2, s2, sum_max)
                target = r + gamma * float(np.max(Q[si2]))

            Q[si, a] = Q[si, a] + alpha * (target - Q[si, a])

            t, s = t2, s2

    return Q, env, sum_max


def greedy_pair(Q: np.ndarray, *, cfg: EnvConfig, sum_max: int) -> tuple[int, int]:
    # 从起点 (t=0,sum=0) 走两步贪心策略
    t, s = 0, 0
    a = int(np.argmax(Q[state_to_index(t, s, sum_max)]))
    t, s = 1, s + a
    b = int(np.argmax(Q[state_to_index(t, s, sum_max)]))
    return a, b


def best_b_for_each_a(Q: np.ndarray, *, cfg: EnvConfig, sum_max: int) -> dict[int, int]:
    # 固定第一步动作 a，第二步状态是 (t=1,sum=a)，输出贪心的 b
    out: dict[int, int] = {}
    for a in range(cfg.max_action + 1):
        t, s = 1, a
        out[a] = int(np.argmax(Q[state_to_index(t, s, sum_max)]))
    return out


def _pair_action_id(a: int, b: int, n: int) -> int:
    return a * n + b


def _pair_from_action_id(action_id: int, n: int) -> tuple[int, int]:
    return int(action_id // n), int(action_id % n)


def train_q_learning_pair_action(
    *,
    episodes: int = 500,
    alpha: float = 0.15,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    seed: int = 0,
    cfg: EnvConfig = EnvConfig(),
) -> tuple[np.ndarray, int]:
    """
    把 (a,b) 当成“一个动作”，一次性选出两个数。
    单步终止：Q(action) ← Q(action) + α (reward - Q(action))
    """
    random.seed(seed)
    np.random.seed(seed)

    n = cfg.max_action + 1  # a,b 都在 0..max_action
    n_actions = n * n
    Q = np.zeros((n_actions,), dtype=np.float32)

    def epsilon_by_episode(ep: int) -> float:
        frac = min(1.0, ep / max(1, episodes - 1))
        return eps_start + frac * (eps_end - eps_start)

    for ep in range(episodes):
        eps = epsilon_by_episode(ep)

        if random.random() < eps:
            action_id = random.randint(0, n_actions - 1)
        else:
            action_id = int(np.argmax(Q))

        a, b = _pair_from_action_id(action_id, n)
        s = a + b
        reward = float(cfg.target - abs(s - cfg.target))

        Q[action_id] = Q[action_id] + alpha * (reward - Q[action_id])

    return Q, n


def best_pair_from_Q(Q_pair: np.ndarray, *, n: int) -> tuple[int, int, float]:
    action_id = int(np.argmax(Q_pair))
    a, b = _pair_from_action_id(action_id, n)
    return a, b, float(Q_pair[action_id])


def learned_best_pairs_from_Q(
    Q_pair: np.ndarray,
    *,
    n: int,
    cfg: EnvConfig,
    tol: float = 1e-3,
) -> list[tuple[int, int]]:
    """
    从训练得到的 Q(action) 中推导“学到的最优动作集合”：
    - 取 Q 值接近最大值（max_q - tol 以内）的动作
    - 再过滤出 a+b==target 的 pair
    """
    max_q = float(np.max(Q_pair))
    action_ids = np.where(Q_pair >= (max_q - tol))[0]
    out: list[tuple[int, int]] = []
    for aid in action_ids.tolist():
        a, b = _pair_from_action_id(int(aid), n)
        if a + b == cfg.target:
            out.append((a, b))
    return sorted(out, key=lambda x: (x[0], x[1]))


def best_b_for_each_a_from_pair_Q(Q_pair: np.ndarray, *, n: int) -> dict[int, int]:
    """
    pair-action 的 Q 是一维的 Q[a*n + b]。
    固定 a 后，在所有 b 上取 argmax，得到“每个 a 的最优 b”。
    """
    out: dict[int, int] = {}
    for a in range(n):
        row = Q_pair[a * n : (a + 1) * n]
        out[a] = int(np.argmax(row))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-learning 找到 a+b=10 的自然数对")
    parser.add_argument(
        "--mode",
        choices=["sequential", "pair", "both"],
        default="both",
        help="sequential: 先选 a 再选 b；pair: (a,b) 作为一个动作；both: 两种都跑",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = EnvConfig(target=10, max_action=10)

    if args.mode in ("sequential", "both"):
        Q, _env, sum_max = train_q_learning(cfg=cfg, episodes=args.episodes, seed=args.seed)

        a, b = greedy_pair(Q, cfg=cfg, sum_max=sum_max)
        print(f"[sequential/greedy] 学到的一个数对: a={a}, b={b}, a+b={a+b}")

        mapping = best_b_for_each_a(Q, cfg=cfg, sum_max=sum_max)
        good_pairs = [(a, bb) for a, bb in mapping.items() if a + bb == cfg.target]
        good_pairs_sorted = sorted(good_pairs, key=lambda x: (x[0], x[1]))

        print("\n[sequential/policy] 每个 a 对应的最优 b（展示前 11 个）：")
        for a in range(cfg.max_action + 1):
            bb = mapping[a]
            print(f"  a={a:2d} -> b={bb:2d} (sum={a+bb:2d})")

        print("\n[sequential/found] 满足 a+b=10 的自然数对（由学到的策略导出）：")
        if good_pairs_sorted:
            print("  " + ", ".join([f"({x},{y})" for x, y in good_pairs_sorted]))
        else:
            print("  (未找到；可以增大 episodes 或调高奖励密度)")

    if args.mode in ("pair", "both"):
        Qp, n = train_q_learning_pair_action(cfg=cfg, episodes=args.episodes, seed=args.seed)
        a2, b2, qv = best_pair_from_Q(Qp, n=n)
        print(f"\n[pair/greedy] 学到的一个数对: a={a2}, b={b2}, a+b={a2+b2} (Q≈{qv:.3f})")

        pair_mapping = best_b_for_each_a_from_pair_Q(Qp, n=n)
        print("\n[pair/policy] 每个 a 对应的最优 b（展示前 11 个）：")
        for a in range(cfg.max_action + 1):
            bb = pair_mapping[a]
            print(f"  a={a:2d} -> b={bb:2d} (sum={a+bb:2d})")

        learned_pairs = learned_best_pairs_from_Q(Qp, n=n, cfg=cfg, tol=1e-3)
        print("\n[pair/found] 从 Q 表推导的“学到的最优数对”（Q 接近最大值，并且 a+b=10）：")
        if learned_pairs:
            print("  " + ", ".join([f"({x},{y})" for x, y in learned_pairs]))
        else:
            print("  (未找到；可以增大 --episodes，或调高探索 eps_start/eps_end)")


if __name__ == "__main__":
    main()


