#!/usr/bin/env python3
"""Predator-prey ODE model with invasive deer introduction.

Species:
- Rabbits (prey)
- Foxes (predator)
- Deer (invasive competitor)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass


@dataclass
class ModelParams:
    rabbit_growth: float = 1.1
    rabbit_carrying_capacity: float = 500.0
    rabbit_predation_rate: float = 0.006
    rabbit_deer_competition: float = 0.0015

    fox_efficiency: float = 0.0009
    fox_death_rate: float = 0.45

    deer_growth: float = 0.65
    deer_carrying_capacity: float = 280.0
    deer_rabbit_competition: float = 0.001
    deer_predation_rate: float = 0.0015


def derivatives(t: float, state: tuple[float, float, float], p: ModelParams) -> tuple[float, float, float]:
    """Compute ODE derivatives for rabbits, foxes, and deer."""
    rabbits, foxes, deer = state

    # Rabbits: logistic growth, fox predation, deer competition.
    d_rabbits = (
        p.rabbit_growth * rabbits * (1 - rabbits / p.rabbit_carrying_capacity)
        - p.rabbit_predation_rate * rabbits * foxes
        - p.rabbit_deer_competition * rabbits * deer
    )

    # Foxes: gain from successful predation on rabbits and deer, natural death.
    d_foxes = (
        p.fox_efficiency * foxes * (rabbits + 0.5 * deer)
        - p.fox_death_rate * foxes
    )

    # Deer: invasive species with logistic growth, competition, and some predation.
    d_deer = (
        p.deer_growth * deer * (1 - deer / p.deer_carrying_capacity)
        - p.deer_rabbit_competition * rabbits * deer
        - p.deer_predation_rate * foxes * deer
    )

    _ = t  # system is autonomous, included for solver interface consistency
    return d_rabbits, d_foxes, d_deer


def rk4_step(t: float, y: tuple[float, float, float], dt: float, p: ModelParams) -> tuple[float, float, float]:
    """Single RK4 integrator step."""

    def add(a: tuple[float, float, float], b: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
        return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])

    k1 = derivatives(t, y, p)
    k2 = derivatives(t + dt / 2.0, add(y, k1, dt / 2.0), p)
    k3 = derivatives(t + dt / 2.0, add(y, k2, dt / 2.0), p)
    k4 = derivatives(t + dt, add(y, k3, dt), p)

    next_y = (
        y[0] + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        y[1] + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        y[2] + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    )

    # Prevent negative populations due to numerical drift.
    return tuple(max(0.0, v) for v in next_y)


def simulate(
    params: ModelParams,
    rabbits0: float,
    foxes0: float,
    deer0: float,
    t_end: float,
    dt: float,
    deer_intro_time: float,
    deer_intro_size: float,
) -> list[tuple[float, float, float, float]]:
    """Run simulation and return rows as (time, rabbits, foxes, deer)."""
    t = 0.0
    state = (rabbits0, foxes0, deer0)
    rows = [(t, *state)]
    deer_introduced = deer0 > 0

    while t < t_end:
        if not deer_introduced and t >= deer_intro_time:
            state = (state[0], state[1], state[2] + deer_intro_size)
            deer_introduced = True

        state = rk4_step(t, state, dt, params)
        t = round(t + dt, 10)
        rows.append((t, *state))

    return rows


def write_csv(path: str, rows: list[tuple[float, float, float, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "rabbits", "foxes", "deer"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Foxes-rabbits ODE model with invasive deer introduction")
    parser.add_argument("--rabbits0", type=float, default=220.0)
    parser.add_argument("--foxes0", type=float, default=30.0)
    parser.add_argument("--deer0", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=80.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--deer-intro-time", type=float, default=20.0)
    parser.add_argument("--deer-intro-size", type=float, default=40.0)
    parser.add_argument("--output", default="simulation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ModelParams()
    rows = simulate(
        params=params,
        rabbits0=args.rabbits0,
        foxes0=args.foxes0,
        deer0=args.deer0,
        t_end=args.t_end,
        dt=args.dt,
        deer_intro_time=args.deer_intro_time,
        deer_intro_size=args.deer_intro_size,
    )

    write_csv(args.output, rows)

    final_t, final_r, final_f, final_d = rows[-1]
    print(f"Simulation complete through t={final_t:.2f}")
    print(f"Final populations -> Rabbits: {final_r:.2f}, Foxes: {final_f:.2f}, Deer: {final_d:.2f}")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
