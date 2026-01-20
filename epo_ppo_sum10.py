"""
PPO-style (with entropy regularization) tabular policy optimization example:
learn two natural numbers (a, b) such that a + b = 50.

Goals:
- Keep a similar output structure to fun/q_learning_sum10.py (modes: sequential / pair / both)
- Add an entropy bonus to reduce premature collapse of exploration
- Record entropy over episodes for both modes
- Optionally log outputs to CSV for later plotting

Run:
  python fun/epo_ppo_sum10.py
  python fun/epo_ppo_sum10.py --mode sequential --episodes 1000
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    target: int = 50
    max_action: int = 50  # action space: choose an integer in [0, max_action]


class SumToTargetEnv:
    """
    State: (t, current_sum)
    - t=0: none picked yet
    - t=1: picked the first number
    - t=2: terminal (picked two numbers)
    Action: pick an integer x in [0, max_action]
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
            reward = float(self.cfg.target - abs(self.current_sum - self.cfg.target))
        return (self.t, self.current_sum), reward, done


def state_to_index(t: int, s: int, sum_max: int) -> int:
    return t * (sum_max + 1) + s


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    ex = np.exp(x, dtype=np.float64)
    p = ex / np.sum(ex)
    return p.astype(np.float64)


def _sample_categorical(p: np.ndarray, rng: random.Random) -> int:
    # p: (n,) sum to 1
    r = rng.random()
    c = 0.0
    for i, pi in enumerate(p.tolist()):
        c += float(pi)
        if r <= c:
            return int(i)
    return int(len(p) - 1)


def _log_prob(p: np.ndarray, a: int) -> float:
    return float(np.log(max(1e-12, float(p[a]))))


def _entropy_grad_from_probs(p: np.ndarray) -> np.ndarray:
    """
    H(p) = -sum p log p
    dH/dlogits = p * (S - (log p + 1)), where S = sum p (log p + 1)
    """
    lp = np.log(np.clip(p, 1e-12, 1.0))
    s = float(np.sum(p * (lp + 1.0)))
    return p * (s - (lp + 1.0))


@dataclass
class PPOConfig:
    episodes: int = 1000
    seed: int = 0
    gamma: float = 0.95
    batch_episodes: int = 20  # how many episodes to collect before an update
    update_epochs: int = 6
    clip_eps: float = 0.2
    lr_pi: float = 0.12
    lr_v: float = 0.20
    ent_start: float = 0.05
    ent_end: float = 0.005
    adv_norm: bool = True


def _linear_schedule(ep: int, total: int, start: float, end: float) -> float:
    frac = min(1.0, ep / max(1, total - 1))
    return start + frac * (end - start)


def train_ppo_sequential(
    *,
    cfg: EnvConfig,
    ppo: PPOConfig,
) -> tuple[np.ndarray, np.ndarray, SumToTargetEnv, int, list[float]]:
    rng = random.Random(ppo.seed)
    np.random.seed(ppo.seed)

    env = SumToTargetEnv(cfg)
    n_actions = cfg.max_action + 1
    sum_max = 2 * cfg.max_action
    n_states = 2 * (sum_max + 1)  # only store t=0,1 (t=2 is terminal)

    logits = np.zeros((n_states, n_actions), dtype=np.float64)
    values = np.zeros((n_states,), dtype=np.float64)

    rollout_states: list[int] = []
    rollout_actions: list[int] = []
    rollout_old_logp: list[float] = []
    rollout_returns: list[float] = []
    entropy_by_episode: list[float] = []

    def flush_and_update(current_ep: int) -> None:
        if not rollout_states:
            return

        s_idx = np.asarray(rollout_states, dtype=np.int64)
        a = np.asarray(rollout_actions, dtype=np.int64)
        old_logp = np.asarray(rollout_old_logp, dtype=np.float64)
        ret = np.asarray(rollout_returns, dtype=np.float64)

        adv = ret - values[s_idx]
        if ppo.adv_norm:
            adv_std = float(np.std(adv) + 1e-8)
            adv = (adv - float(np.mean(adv))) / adv_std

        ent_beta = _linear_schedule(current_ep, ppo.episodes, ppo.ent_start, ppo.ent_end)

        for _ in range(ppo.update_epochs):
            for i in range(len(s_idx)):
                si = int(s_idx[i])
                ai = int(a[i])
                A = float(adv[i])
                R = float(ret[i])

                p = _softmax(logits[si])
                logp = _log_prob(p, ai)
                ratio = float(np.exp(logp - float(old_logp[i])))

                clipped = False
                if (A >= 0.0 and ratio > (1.0 + ppo.clip_eps)) or (A < 0.0 and ratio < (1.0 - ppo.clip_eps)):
                    clipped = True

                # policy update (ascent)
                if not clipped:
                    # d/dlogits of log pi(a|s) is (onehot - p)
                    grad_logp = -p
                    grad_logp[ai] += 1.0
                    logits[si] += ppo.lr_pi * (ratio * A) * grad_logp

                # entropy bonus (always)
                logits[si] += ppo.lr_pi * ent_beta * _entropy_grad_from_probs(p)

                # value update (MSE; ascent on -(V-R)^2 => descent on (V-R)^2)
                v = float(values[si])
                values[si] = v + ppo.lr_v * (R - v)

        rollout_states.clear()
        rollout_actions.clear()
        rollout_old_logp.clear()
        rollout_returns.clear()

    for ep in range(ppo.episodes):
        (t, s) = env.reset()

        traj_states: list[int] = []
        traj_actions: list[int] = []
        traj_old_logp: list[float] = []
        rewards: list[float] = []
        entropies: list[float] = []

        done = False
        while not done:
            si = state_to_index(t, s, sum_max)
            p = _softmax(logits[si])
            # Entropy tracking for this episode: entropy of the policy distribution at the visited state
            entropies.append(float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0)))))
            a = _sample_categorical(p, rng)
            traj_states.append(si)
            traj_actions.append(a)
            traj_old_logp.append(_log_prob(p, a))

            (t2, s2), r, done = env.step(a)
            rewards.append(float(r))
            t, s = t2, s2

        # returns
        G = 0.0
        returns_rev: list[float] = []
        for r in reversed(rewards):
            G = float(r) + ppo.gamma * G
            returns_rev.append(G)
        returns = list(reversed(returns_rev))

        rollout_states.extend(traj_states)
        rollout_actions.extend(traj_actions)
        rollout_old_logp.extend(traj_old_logp)
        rollout_returns.extend(returns)
        entropy_by_episode.append(float(np.mean(entropies)) if entropies else 0.0)

        if (ep + 1) % ppo.batch_episodes == 0:
            flush_and_update(ep)

    flush_and_update(ppo.episodes - 1)
    return logits.astype(np.float64), values.astype(np.float64), env, sum_max, entropy_by_episode


def greedy_pair_from_policy(logits: np.ndarray, *, cfg: EnvConfig, sum_max: int) -> tuple[int, int]:
    t, s = 0, 0
    si = state_to_index(t, s, sum_max)
    a = int(np.argmax(logits[si]))
    t, s = 1, s + a
    si2 = state_to_index(t, s, sum_max)
    b = int(np.argmax(logits[si2]))
    return a, b


def best_b_for_each_a_from_policy(logits: np.ndarray, *, cfg: EnvConfig, sum_max: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for a in range(cfg.max_action + 1):
        si = state_to_index(1, a, sum_max)
        out[a] = int(np.argmax(logits[si]))
    return out


def _pair_from_action_id(action_id: int, n: int) -> tuple[int, int]:
    return int(action_id // n), int(action_id % n)


def train_ppo_pair_action(
    *,
    cfg: EnvConfig,
    ppo: PPOConfig,
) -> tuple[np.ndarray, float, int, list[float]]:
    """
    Treat (a, b) as a single action (single-step bandit-like environment).
    Use PPO + entropy regularization to learn a distribution pi(a,b).
    Since it's one-step, there is no bootstrapping; returns = reward.
    """
    rng = random.Random(ppo.seed)
    np.random.seed(ppo.seed)

    n = cfg.max_action + 1
    n_actions = n * n

    logits = np.zeros((n_actions,), dtype=np.float64)
    v = 0.0  # single-state value baseline

    rollout_actions: list[int] = []
    rollout_old_logp: list[float] = []
    rollout_returns: list[float] = []
    entropy_by_episode: list[float] = []

    def flush_and_update(current_ep: int) -> None:
        nonlocal v
        if not rollout_actions:
            return

        a = np.asarray(rollout_actions, dtype=np.int64)
        old_logp = np.asarray(rollout_old_logp, dtype=np.float64)
        ret = np.asarray(rollout_returns, dtype=np.float64)

        adv = ret - v
        if ppo.adv_norm:
            adv_std = float(np.std(adv) + 1e-8)
            adv = (adv - float(np.mean(adv))) / adv_std

        # Pair-action has n^2 actions; max entropy is ln(n^2). Without scaling, the entropy term can
        # keep the policy near-uniform and stall learning. Scale by 1/n to keep exploration but
        # not drown out the advantage signal.
        ent_beta = _linear_schedule(current_ep, ppo.episodes, ppo.ent_start, ppo.ent_end) * (1.0 / float(n))

        for _ in range(ppo.update_epochs):
            p = _softmax(logits)
            ent_g = _entropy_grad_from_probs(p)

            for i in range(len(a)):
                ai = int(a[i])
                A = float(adv[i])
                R = float(ret[i])

                logp = _log_prob(p, ai)
                ratio = float(np.exp(logp - float(old_logp[i])))

                clipped = False
                if (A >= 0.0 and ratio > (1.0 + ppo.clip_eps)) or (A < 0.0 and ratio < (1.0 - ppo.clip_eps)):
                    clipped = True

                if not clipped:
                    grad_logp = -p
                    grad_logp[ai] += 1.0
                    logits[:] = logits + ppo.lr_pi * (ratio * A) * grad_logp

                # entropy bonus
                logits[:] = logits + ppo.lr_pi * ent_beta * ent_g

                # value update
                v = float(v + ppo.lr_v * (R - v))

                # Refresh p lightly (avoid drifting too far within one epoch)
                p = _softmax(logits)
                ent_g = _entropy_grad_from_probs(p)

        rollout_actions.clear()
        rollout_old_logp.clear()
        rollout_returns.clear()

    for ep in range(ppo.episodes):
        p = _softmax(logits)
        entropy_by_episode.append(float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0)))))
        action_id = _sample_categorical(p, rng)
        rollout_actions.append(action_id)
        rollout_old_logp.append(_log_prob(p, action_id))

        a, b = _pair_from_action_id(action_id, n)
        s = a + b
        reward = float(cfg.target - abs(s - cfg.target))
        rollout_returns.append(reward)

        if (ep + 1) % ppo.batch_episodes == 0:
            flush_and_update(ep)

    flush_and_update(ppo.episodes - 1)
    return logits.astype(np.float64), float(v), n, entropy_by_episode


def best_pair_from_policy_logits(logits: np.ndarray, *, n: int, cfg: EnvConfig) -> tuple[int, int, float]:
    action_id = int(np.argmax(logits))
    a, b = _pair_from_action_id(action_id, n)
    s = a + b
    score = float(cfg.target - abs(s - cfg.target))
    return a, b, score


def best_b_for_each_a_from_pair_policy(logits: np.ndarray, *, n: int) -> dict[int, int]:
    # Fix a, choose argmax over b
    out: dict[int, int] = {}
    for a in range(n):
        row = logits[a * n : (a + 1) * n]
        out[a] = int(np.argmax(row))
    return out


def learned_best_pairs_from_pair_policy(
    logits: np.ndarray,
    *,
    n: int,
    cfg: EnvConfig,
    top_k: int = 200,
) -> list[tuple[int, int]]:
    """
    Derive a set of learned "good" solutions from the policy:
    - take top_k actions by logits
    - filter those with a+b == target
    """
    k = int(min(max(1, top_k), logits.shape[0]))
    # argsort ascending -> take last k and reverse
    ids = np.argsort(logits)[-k:][::-1]
    out: list[tuple[int, int]] = []
    for aid in ids.tolist():
        a, b = _pair_from_action_id(int(aid), n)
        if a + b == cfg.target:
            out.append((a, b))
    return sorted(out, key=lambda x: (x[0], x[1]))


def _write_entropy_csv(
    entropy_path: Path,
    *,
    run_id: str,
    episodes: int,
    seed: int,
    target: int,
    max_action: int,
    sequential_entropy: list[float] | None,
    pair_entropy: list[float] | None,
) -> None:
    entropy_path.parent.mkdir(parents=True, exist_ok=True)
    with entropy_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "episode", "mode", "entropy", "episodes", "seed", "target", "max_action"])
        if sequential_entropy is not None:
            for i, h in enumerate(sequential_entropy, start=1):
                w.writerow([run_id, i, "sequential", float(h), episodes, seed, target, max_action])
        if pair_entropy is not None:
            for i, h in enumerate(pair_entropy, start=1):
                w.writerow([run_id, i, "pair", float(h), episodes, seed, target, max_action])


def _write_summary_csv(
    summary_path: Path,
    *,
    run_id: str,
    episodes: int,
    seed: int,
    target: int,
    max_action: int,
    rows: list[dict[str, object]],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "mode",
        "episodes",
        "seed",
        "target",
        "max_action",
        "batch_episodes",
        "update_epochs",
        "clip_eps",
        "lr_pi",
        "lr_v",
        "ent_start",
        "ent_end",
        "greedy_a",
        "greedy_b",
        "greedy_sum",
        "greedy_score",
        "found_count",
        "found_pairs",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["run_id"] = run_id
            out["episodes"] = episodes
            out["seed"] = seed
            out["target"] = target
            out["max_action"] = max_action
            w.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO/EPO (entropy-regularized) learn pairs with a+b=50")
    parser.add_argument(
        "--mode",
        choices=["sequential", "pair", "both"],
        default="both",
        help="sequential: choose a then b; pair: treat (a,b) as one action; both: run both",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-dir", type=str, default="fun/csv_logs", help="directory to write CSV logs")
    parser.add_argument("--no-csv", action="store_true", help="disable CSV logging")

    # Hyperparameters (separate defaults per mode to increase the number of valid pairs found)
    # Note: pair-action entropy is additionally scaled by 1/n inside train_ppo_pair_action.
    parser.add_argument("--seq-batch-episodes", type=int, default=40)
    parser.add_argument("--seq-update-epochs", type=int, default=8)
    parser.add_argument("--seq-clip-eps", type=float, default=0.2)
    parser.add_argument("--seq-lr-pi", type=float, default=0.14)
    parser.add_argument("--seq-lr-v", type=float, default=0.22)
    parser.add_argument("--seq-ent-start", type=float, default=0.08)
    parser.add_argument("--seq-ent-end", type=float, default=0.02)

    parser.add_argument("--pair-batch-episodes", type=int, default=100)
    parser.add_argument("--pair-update-epochs", type=int, default=4)
    parser.add_argument("--pair-clip-eps", type=float, default=0.2)
    parser.add_argument("--pair-lr-pi", type=float, default=0.10)
    parser.add_argument("--pair-lr-v", type=float, default=0.15)
    parser.add_argument("--pair-ent-start", type=float, default=0.35)
    parser.add_argument("--pair-ent-end", type=float, default=0.10)
    parser.add_argument("--pair-found-top-k", type=int, default=500, help="top-k logits used to derive valid pairs in pair mode")
    args = parser.parse_args()

    cfg = EnvConfig(target=50, max_action=50)
    # Build per-mode PPO configs. This is important because sequential (factorized) and pair (n^2 actions)
    # behave very differently: sequential needs enough exploration on the first step to visit many sums,
    # while pair needs enough entropy to keep many optimal actions near the top.
    ppo_seq = PPOConfig(
        episodes=int(args.episodes),
        seed=int(args.seed),
        batch_episodes=int(args.seq_batch_episodes),
        update_epochs=int(args.seq_update_epochs),
        clip_eps=float(args.seq_clip_eps),
        lr_pi=float(args.seq_lr_pi),
        lr_v=float(args.seq_lr_v),
        ent_start=float(args.seq_ent_start),
        ent_end=float(args.seq_ent_end),
    )
    ppo_pair = PPOConfig(
        episodes=int(args.episodes),
        seed=int(args.seed),
        batch_episodes=int(args.pair_batch_episodes),
        update_epochs=int(args.pair_update_epochs),
        clip_eps=float(args.pair_clip_eps),
        lr_pi=float(args.pair_lr_pi),
        lr_v=float(args.pair_lr_v),
        ent_start=float(args.pair_ent_start),
        ent_end=float(args.pair_ent_end),
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = Path(str(args.csv_dir))
    entropy_csv = csv_dir / f"run_{run_id}_entropy.csv"
    summary_csv = csv_dir / f"run_{run_id}_summary.csv"
    summary_rows: list[dict[str, object]] = []
    ent_seq: list[float] | None = None
    ent_pair: list[float] | None = None

    if args.mode in ("sequential", "both"):
        logits, _values, _env, sum_max, ent_hist = train_ppo_sequential(cfg=cfg, ppo=ppo_seq)
        ent_seq = ent_hist

        mapping = best_b_for_each_a_from_policy(logits, cfg=cfg, sum_max=sum_max)
        # Derive a global best pair from "best b for each a". If feasible solutions exist, this will
        # output one of them, making the greedy line more informative.
        best_pair: tuple[int, int] | None = None
        best_score = -1e18
        for aa in range(cfg.max_action + 1):
            bb = int(mapping[aa])
            s = aa + bb
            score = float(cfg.target - abs(s - cfg.target))
            if score > best_score:
                best_score = score
                best_pair = (aa, bb)
        a, b = best_pair if best_pair is not None else (0, 0)
        print(f"[sequential/greedy] learned pair: a={a}, b={b}, a+b={a+b}")

        good_pairs = [(a, bb) for a, bb in mapping.items() if a + bb == cfg.target]
        good_pairs_sorted = sorted(good_pairs, key=lambda x: (x[0], x[1]))

        show_n = min(11, cfg.max_action + 1)
        print(f"\n[sequential/policy] best b for each a (first {show_n}):")
        for a in range(show_n):
            bb = mapping[a]
            print(f"  a={a:2d} -> b={bb:2d} (sum={a+bb:2d})")

        print(f"\n[sequential/found] pairs with a+b={cfg.target} (derived from learned policy):")
        if good_pairs_sorted:
            print("  " + ", ".join([f"({x},{y})" for x, y in good_pairs_sorted]))
        else:
            print("  (none found; try more episodes or adjust ent_start / batch_episodes)")

        print(f"\n[sequential/count] total valid pairs found: {len(good_pairs_sorted)}")
        # Entropy over episodes (sampled summary)
        # if ent_hist:
        #     stride = max(1, ppo.episodes // 20)  # print at most ~20 points
        #     print(f"[sequential/entropy] entropy over episodes (one point every {stride} episodes):")
        #     for i in range(0, len(ent_hist), stride):
        #         print(f"  ep={i+1:4d}  H≈{ent_hist[i]:.4f}")

        greedy_score = float(cfg.target - abs((a + b) - cfg.target))
        summary_rows.append(
            {
                "mode": "sequential",
                "batch_episodes": int(ppo_seq.batch_episodes),
                "update_epochs": int(ppo_seq.update_epochs),
                "clip_eps": float(ppo_seq.clip_eps),
                "lr_pi": float(ppo_seq.lr_pi),
                "lr_v": float(ppo_seq.lr_v),
                "ent_start": float(ppo_seq.ent_start),
                "ent_end": float(ppo_seq.ent_end),
                "greedy_a": int(a),
                "greedy_b": int(b),
                "greedy_sum": int(a + b),
                "greedy_score": float(greedy_score),
                "found_count": int(len(good_pairs_sorted)),
                "found_pairs": ",".join([f"({x},{y})" for x, y in good_pairs_sorted]),
            }
        )

    if args.mode in ("pair", "both"):
        plogits, _v, n, ent_hist = train_ppo_pair_action(cfg=cfg, ppo=ppo_pair)
        ent_pair = ent_hist
        a2, b2, score = best_pair_from_policy_logits(plogits, n=n, cfg=cfg)
        print(f"\n[pair/greedy] learned pair: a={a2}, b={b2}, a+b={a2+b2} (score≈{score:.3f})")

        pair_mapping = best_b_for_each_a_from_pair_policy(plogits, n=n)
        show_n = min(11, cfg.max_action + 1)
        print(f"\n[pair/policy] best b for each a (first {show_n}):")
        for a in range(show_n):
            bb = pair_mapping[a]
            print(f"  a={a:2d} -> b={bb:2d} (sum={a+bb:2d})")

        top_k = int(args.pair_found_top_k)
        learned_pairs = learned_best_pairs_from_pair_policy(plogits, n=n, cfg=cfg, top_k=top_k)
        print(f"\n[pair/found] pairs with a+b={cfg.target} (from top-{top_k} logits):")
        if learned_pairs:
            print("  " + ", ".join([f"({x},{y})" for x, y in learned_pairs]))
        else:
            print("  (none found; try more episodes or adjust ent_start/ent_end)")

        print(f"\n[pair/count] total valid pairs found: {len(learned_pairs)}")
        # if ent_hist:
        #     stride = max(1, ppo.episodes // 20)
        #     print(f"[pair/entropy] entropy over episodes (one point every {stride} episodes):")
        #     for i in range(0, len(ent_hist), stride):
        #         print(f"  ep={i+1:4d}  H≈{ent_hist[i]:.4f}")

        summary_rows.append(
            {
                "mode": "pair",
                "batch_episodes": int(ppo_pair.batch_episodes),
                "update_epochs": int(ppo_pair.update_epochs),
                "clip_eps": float(ppo_pair.clip_eps),
                "lr_pi": float(ppo_pair.lr_pi),
                "lr_v": float(ppo_pair.lr_v),
                "ent_start": float(ppo_pair.ent_start),
                "ent_end": float(ppo_pair.ent_end),
                "greedy_a": int(a2),
                "greedy_b": int(b2),
                "greedy_sum": int(a2 + b2),
                "greedy_score": float(score),
                "found_count": int(len(learned_pairs)),
                "found_pairs": ",".join([f"({x},{y})" for x, y in learned_pairs]),
            }
        )

    if not bool(args.no_csv):
        _write_entropy_csv(
            entropy_csv,
            run_id=run_id,
            episodes=int(args.episodes),
            seed=int(args.seed),
            target=int(cfg.target),
            max_action=int(cfg.max_action),
            sequential_entropy=ent_seq,
            pair_entropy=ent_pair,
        )
        _write_summary_csv(
            summary_csv,
            run_id=run_id,
            episodes=int(args.episodes),
            seed=int(args.seed),
            target=int(cfg.target),
            max_action=int(cfg.max_action),
            rows=summary_rows,
        )
        print(f"\n[csv] wrote: entropy={entropy_csv.as_posix()}, summary={summary_csv.as_posix()}")


if __name__ == "__main__":
    main()


