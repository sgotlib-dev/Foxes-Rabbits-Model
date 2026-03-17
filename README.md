# Foxes-Rabbits-Deer ODE Model

This repository contains a quick ordinary differential equation (ODE) model for predator-prey dynamics with an invasive species.

## Species in the system
- **Rabbits**: primary prey, logistic growth.
- **Foxes**: predator, gains from predation and loses population via natural death.
- **Deer**: invasive species introduced part-way through the simulation, competes with rabbits and is weakly preyed on by foxes.

## Model equations
Let:
- \(R\): rabbits
- \(F\): foxes
- \(D\): deer

The model uses:

\[
\frac{dR}{dt} = r_R R\left(1 - \frac{R}{K_R}\right) - a_{RF}RF - c_{RD}RD
\]

\[
\frac{dF}{dt} = e_F F(R + 0.5D) - m_F F
\]

\[
\frac{dD}{dt} = r_D D\left(1 - \frac{D}{K_D}\right) - c_{DR}RD - a_{DF}FD
\]

Numerical integration is done with a 4th-order Runge-Kutta (RK4) solver.

## Run
```bash
python3 predator_prey_model.py --output simulation.csv
```

Example with custom introduction time:
```bash
python3 predator_prey_model.py --deer-intro-time 15 --deer-intro-size 55
```

The script prints final population values and writes a CSV time series.
