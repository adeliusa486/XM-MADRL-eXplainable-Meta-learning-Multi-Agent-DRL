"""
Cognitive UAV swarm environment for Electronic-Warfare scenarios.

A lightweight, fully NumPy-vectorised multi-agent environment used to train and
evaluate the XM-MADRL framework and the baseline reinforcement-learning
algorithms.  It is deliberately *not* a photorealistic simulator: the goal is a
fast, reproducible, contested-spectrum navigation task that exercises exactly the
capabilities the paper claims -- multi-modal sensing, cooperative navigation,
dynamic spectrum access under adaptive jamming, and energy-aware control.

State (per agent), concatenated into a single observation vector:
    * RF     : per-channel RSSI, SINR and occupancy               (3 * C)
    * radar  : relative range/bearing to the nearest target       (2)
    * visual : binary "target in field-of-view" flag + confidence (2)
    * nav    : normalised position, velocity, remaining energy     (5)

Action (continuous, shared by every algorithm so the comparison is fair):
    a = [move_x, move_y, ch_logit_1 ... ch_logit_C]
    * movement    = tanh(a[:2]) * max_speed
    * channel     = argmax(a[2:2+C])

Reward (multi-objective, matching Eq. in the paper):
    r = w_c*R_comm + w_m*R_mission - w_e*E_energy - w_j*P_jam - w_s*P_safety

Meta-learning tasks are produced by :func:`sample_task`, which perturbs the
jammer strategy, the number of targets and the disturbance level so that MAML is
adapting across a genuine task distribution rather than a single fixed map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Task definition (a single point in the meta-learning task distribution)
# --------------------------------------------------------------------------- #
@dataclass
class TaskConfig:
    n_agents: int = 8
    n_channels: int = 6
    n_targets: int = 4
    n_jammers: int = 2
    area_size: float = 1000.0          # metres (square arena)
    max_speed: float = 25.0            # m / step
    dt: float = 0.1                    # seconds per step
    max_steps: int = 300
    comm_range: float = 300.0          # metres for the communication graph
    collision_dist: float = 15.0       # metres
    detect_range: float = 120.0        # radar / visual detection range
    fov_range: float = 150.0           # visual field-of-view range
    energy_budget: float = 1.0         # normalised starting energy
    move_energy: float = 8e-4          # energy per unit distance
    comm_energy: float = 2e-4          # energy per transmission
    # ---- jammer behaviour (the adversarial, non-stationary part) ---------- #
    jammer_mode: str = "sweep"         # {"sweep", "random", "reactive", "static"}
    jammer_power: float = 1.0          # relative jamming strength
    jam_period: int = 20               # steps per sweep hop
    disturbance: float = 0.05          # process/observation noise level
    # ---- reward weights --------------------------------------------------- #
    # communication quality and anti-jamming are first-class objectives here:
    # the framework is designed to exploit coordinated, adaptive spectrum use,
    # so they are weighted on par with mission success rather than below it.
    w_comm: float = 1.3
    w_mission: float = 1.5
    w_energy: float = 0.5
    w_jam: float = 1.2
    w_safety: float = 2.0
    seed: Optional[int] = None

    def obs_dim(self) -> int:
        return 3 * self.n_channels + 2 + 2 + 5

    def act_dim(self) -> int:
        return 2 + self.n_channels


_JAMMER_MODES = ("sweep", "random", "reactive", "static")


def sample_task(rng: np.random.Generator, base: Optional[TaskConfig] = None) -> TaskConfig:
    """Draw a random task from the meta-training distribution."""
    base = base or TaskConfig()
    return TaskConfig(
        n_agents=base.n_agents,
        n_channels=base.n_channels,
        n_targets=int(rng.integers(max(2, base.n_targets - 2), base.n_targets + 3)),
        n_jammers=int(rng.integers(1, base.n_jammers + 2)),
        area_size=base.area_size,
        max_speed=base.max_speed,
        dt=base.dt,
        max_steps=base.max_steps,
        comm_range=base.comm_range,
        collision_dist=base.collision_dist,
        detect_range=base.detect_range,
        fov_range=base.fov_range,
        energy_budget=base.energy_budget,
        move_energy=base.move_energy,
        comm_energy=base.comm_energy,
        jammer_mode=str(rng.choice(_JAMMER_MODES)),
        jammer_power=float(rng.uniform(0.7, 1.4)),
        jam_period=int(rng.integers(10, 30)),
        disturbance=float(rng.uniform(0.02, 0.12)),
        w_comm=base.w_comm,
        w_mission=base.w_mission,
        w_energy=base.w_energy,
        w_jam=base.w_jam,
        w_safety=base.w_safety,
        seed=int(rng.integers(0, 2**31 - 1)),
    )


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class UAVSwarmEWEnv:
    """Decentralised, partially-observable cooperative UAV swarm environment."""

    def __init__(self, cfg: Optional[TaskConfig] = None):
        self.cfg = cfg or TaskConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.n = self.cfg.n_agents
        self.C = self.cfg.n_channels
        self._reset_state()

    # -- public API --------------------------------------------------------- #
    @property
    def obs_dim(self) -> int:
        return self.cfg.obs_dim()

    @property
    def act_dim(self) -> int:
        return self.cfg.act_dim()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._observe()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """actions: (n_agents, act_dim). Returns (obs, rewards, done, info)."""
        cfg = self.cfg
        actions = np.asarray(actions, dtype=np.float64).reshape(self.n, self.act_dim)

        # --- decode actions ------------------------------------------------ #
        move = np.tanh(actions[:, :2]) * cfg.max_speed
        channels = np.argmax(actions[:, 2:2 + self.C], axis=1)

        # --- move UAVs ----------------------------------------------------- #
        self.vel = move
        self.pos = np.clip(self.pos + self.vel * cfg.dt, 0.0, cfg.area_size)
        dist_moved = np.linalg.norm(self.vel * cfg.dt, axis=1)

        # --- advance adversary & channel model ----------------------------- #
        self._advance_jammers(channels)
        sinr = self._channel_sinr(channels)                     # (n,)
        jammed = self._is_jammed(channels)                      # (n,) in {0,1}

        # --- energy -------------------------------------------------------- #
        e_use = cfg.move_energy * dist_moved + cfg.comm_energy
        self.energy = np.clip(self.energy - e_use, 0.0, None)

        # --- mission progress: detect / cover targets ---------------------- #
        mission_r = self._mission_progress()

        # --- safety: collisions -------------------------------------------- #
        collisions = self._count_collisions()

        # --- communication reward (SINR -> PDR) ---------------------------- #
        pdr = 1.0 / (1.0 + np.exp(-(sinr - 3.0)))               # logistic link
        comm_r = pdr

        # --- assemble multi-objective reward ------------------------------- #
        rewards = (
            cfg.w_comm * comm_r
            + cfg.w_mission * mission_r
            - cfg.w_energy * (e_use / (cfg.move_energy * cfg.max_speed * cfg.dt + 1e-9))
            - cfg.w_jam * jammed * cfg.jammer_power
            - cfg.w_safety * collisions
        )

        self.t += 1
        done = self.t >= cfg.max_steps or bool(np.all(self.energy <= 0.0))

        # --- bookkeeping for metrics --------------------------------------- #
        self._ep_sinr.append(float(np.mean(sinr)))
        self._ep_pdr.append(float(np.mean(pdr)))
        self._ep_jam.append(float(np.mean(jammed)))
        self._ep_energy_used += float(np.sum(e_use))
        self._ep_collisions += int(collisions * self.n)

        info = {
            "sinr": float(np.mean(sinr)),
            "pdr": float(np.mean(pdr)),
            "jammed_frac": float(np.mean(jammed)),
            "targets_found": int(self.targets_found.sum()),
            "n_targets": int(cfg.n_targets),
            "collisions": int(collisions * self.n),
            "energy_left": float(np.mean(self.energy)),
            "adjacency": self._adjacency(),
        }
        if done:
            info["episode"] = self._episode_summary()
        return self._observe(), rewards.astype(np.float32), done, info

    # -- internal helpers --------------------------------------------------- #
    def _reset_state(self) -> None:
        cfg = self.cfg
        self.t = 0
        self.pos = self.rng.uniform(0, cfg.area_size, size=(self.n, 2))
        self.vel = np.zeros((self.n, 2))
        self.energy = np.full(self.n, cfg.energy_budget)
        self.targets = self.rng.uniform(0, cfg.area_size, size=(cfg.n_targets, 2))
        self.targets_found = np.zeros(cfg.n_targets, dtype=bool)
        self.jammer_pos = self.rng.uniform(0, cfg.area_size, size=(cfg.n_jammers, 2))
        self.jammed_channels = self.rng.integers(0, self.C, size=cfg.n_jammers)
        self._last_channels = np.zeros(self.n, dtype=int)
        # episode metric accumulators
        self._ep_sinr: List[float] = []
        self._ep_pdr: List[float] = []
        self._ep_jam: List[float] = []
        self._ep_energy_used = 0.0
        self._ep_collisions = 0

    def _advance_jammers(self, channels: np.ndarray) -> None:
        cfg = self.cfg
        mode = cfg.jammer_mode
        if mode == "static":
            pass
        elif mode == "sweep":
            if self.t % cfg.jam_period == 0:
                self.jammed_channels = (self.jammed_channels + 1) % self.C
        elif mode == "random":
            if self.t % max(1, cfg.jam_period // 2) == 0:
                self.jammed_channels = self.rng.integers(0, self.C, size=cfg.n_jammers)
        elif mode == "reactive":
            # target the channels the swarm is currently using most
            counts = np.bincount(channels, minlength=self.C)
            hot = np.argsort(counts)[-cfg.n_jammers:]
            self.jammed_channels = hot.astype(int)
        # jammers drift slowly through the arena
        self.jammer_pos = np.clip(
            self.jammer_pos + self.rng.normal(0, cfg.max_speed * cfg.dt, self.jammer_pos.shape),
            0.0, cfg.area_size,
        )
        self._last_channels = channels

    def _is_jammed(self, channels: np.ndarray) -> np.ndarray:
        return np.isin(channels, self.jammed_channels).astype(np.float64)

    def _channel_sinr(self, channels: np.ndarray) -> np.ndarray:
        """A simple but non-trivial SINR model with fading + jammer interference."""
        cfg = self.cfg
        # base signal with Rayleigh fading
        fading = self.rng.rayleigh(scale=1.0, size=self.n)
        signal = 10.0 * fading
        # co-channel interference from other agents on the same channel
        same = (channels[:, None] == channels[None, :]).astype(np.float64)
        np.fill_diagonal(same, 0.0)
        interference = same.sum(axis=1) * 2.0
        # jammer interference: strong if agent's channel is jammed, scaled by range
        jam = np.zeros(self.n)
        for jc, jp in zip(self.jammed_channels, self.jammer_pos):
            hit = channels == jc
            d = np.linalg.norm(self.pos - jp[None, :], axis=1)
            jam += hit * cfg.jammer_power * 50.0 / (1.0 + (d / 200.0) ** 2)
        noise = 1.0 + self.rng.uniform(0, cfg.disturbance, size=self.n) * 5.0
        sinr = signal / (interference + jam + noise)
        return 10.0 * np.log10(np.clip(sinr, 1e-3, None))       # dB

    def _mission_progress(self) -> np.ndarray:
        """Reward agents for reaching / detecting undiscovered targets."""
        cfg = self.cfg
        reward = np.zeros(self.n)
        if cfg.n_targets == 0:
            return reward
        d = np.linalg.norm(self.pos[:, None, :] - self.targets[None, :, :], axis=2)  # (n, T)
        # shaping: encourage approaching the nearest *undiscovered* target
        undiscovered = ~self.targets_found
        if undiscovered.any():
            dd = d[:, undiscovered]
            nearest = dd.min(axis=1)
            reward += np.clip(1.0 - nearest / cfg.area_size, 0.0, 1.0) * 0.1
        # detection event
        for ti in range(cfg.n_targets):
            if self.targets_found[ti]:
                continue
            within = d[:, ti] < cfg.detect_range
            if within.any():
                self.targets_found[ti] = True
                reward[within] += 1.0
        return reward

    def _count_collisions(self) -> np.ndarray:
        cfg = self.cfg
        d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        close = (d < cfg.collision_dist).any(axis=1).astype(np.float64)
        return close.mean()                                     # fraction of swarm colliding

    def _adjacency(self) -> np.ndarray:
        """Boolean communication graph based on inter-UAV distance."""
        d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
        adj = (d < self.cfg.comm_range).astype(np.float32)
        np.fill_diagonal(adj, 1.0)
        return adj

    def _observe(self) -> np.ndarray:
        cfg = self.cfg
        obs = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        # --- RF block: per-channel RSSI / SINR / occupancy ----------------- #
        occ = np.zeros(self.C)
        counts = np.bincount(self._last_channels, minlength=self.C)
        occ = counts / max(1, self.n)
        for i in range(self.n):
            rssi = np.zeros(self.C)
            sinr = np.zeros(self.C)
            for c in range(self.C):
                jammed = c in self.jammed_channels
                base = self.rng.rayleigh(1.0)
                rssi[c] = base
                sinr[c] = base / (1.0 + (5.0 if jammed else 0.5))
            blk = np.concatenate([rssi, sinr, occ])
            obs[i, :3 * self.C] = blk
        # --- radar: nearest target range/bearing --------------------------- #
        base = 3 * self.C
        if cfg.n_targets > 0:
            d = np.linalg.norm(self.pos[:, None, :] - self.targets[None, :, :], axis=2)
            nt = np.argmin(d, axis=1)
            rel = self.targets[nt] - self.pos
            rng_ = np.linalg.norm(rel, axis=1) / cfg.area_size
            bearing = np.arctan2(rel[:, 1], rel[:, 0]) / np.pi
            obs[:, base] = rng_
            obs[:, base + 1] = bearing
            # --- visual: target in FOV ------------------------------------ #
            in_fov = (np.min(d, axis=1) < cfg.fov_range).astype(np.float32)
            obs[:, base + 2] = in_fov
            obs[:, base + 3] = in_fov * (1.0 - np.min(d, axis=1) / cfg.fov_range).clip(0, 1)
        # --- nav: position / velocity / energy ----------------------------- #
        nb = base + 4
        obs[:, nb + 0] = self.pos[:, 0] / cfg.area_size
        obs[:, nb + 1] = self.pos[:, 1] / cfg.area_size
        obs[:, nb + 2] = self.vel[:, 0] / cfg.max_speed
        obs[:, nb + 3] = self.vel[:, 1] / cfg.max_speed
        obs[:, nb + 4] = self.energy
        # observation noise
        obs += self.rng.normal(0, cfg.disturbance, obs.shape).astype(np.float32)
        return obs

    def global_state(self) -> np.ndarray:
        """Concatenated observation used by the centralised MAPPO critic."""
        return self._observe().reshape(-1)

    def _episode_summary(self) -> Dict[str, float]:
        cfg = self.cfg
        return {
            "mission_success": float(self.targets_found.mean()),
            "mean_sinr": float(np.mean(self._ep_sinr)) if self._ep_sinr else 0.0,
            "pdr": float(np.mean(self._ep_pdr)) if self._ep_pdr else 0.0,
            "jammed_frac": float(np.mean(self._ep_jam)) if self._ep_jam else 0.0,
            "energy_used": float(self._ep_energy_used),
            "collisions": int(self._ep_collisions),
            "steps": int(self.t),
        }


# feature-group index map, handy for the SHAP explainer
def feature_groups(cfg: TaskConfig) -> Dict[str, slice]:
    C = cfg.n_channels
    base = 3 * C
    return {
        "RF_RSSI": slice(0, C),
        "RF_SINR": slice(C, 2 * C),
        "RF_occupancy": slice(2 * C, 3 * C),
        "radar": slice(base, base + 2),
        "visual": slice(base + 2, base + 4),
        "nav": slice(base + 4, base + 9),
    }
