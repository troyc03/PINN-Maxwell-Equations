# PINN-Maxwell-Equations

**A research-oriented computational electromagnetics project comparing physics-informed neural networks (PINNs) with classical numerical methods for electromagnetic field problems.**

This repository is being developed as an advanced undergraduate research project in computational electromagnetics, applied mathematics, and scientific computing. Its objective is to investigate how physics-informed machine learning can be used to solve electromagnetic partial differential equations (PDEs), and to evaluate its accuracy, stability, and computational cost against established numerical approaches.

The project combines mathematical formulation, numerical implementation, physical validation, and reproducible computational experiments. It is intended to grow from verified electrostatic benchmarks into time-dependent electromagnetic simulations.

> **Development status:** Early-stage project. The repository is being organized and the numerical framework is planned. Solvers, benchmark results, and performance claims will be documented as they are implemented and validated. Planned capabilities described below should not be interpreted as already completed.

---

## Research question

**Under what conditions can physics-informed neural networks provide accurate and computationally useful solutions to electromagnetic boundary-value and initial-value problems compared with established finite-difference and finite-element methods?**

The project will investigate this question through controlled benchmark problems. Comparisons will use equivalent physical models, source terms, material parameters, and boundary or initial conditions wherever possible.

## Scope and objectives

The work is organized around four complementary areas:

1. **Classical numerical methods** — Develop finite-difference and finite-element reference approaches for selected electrostatic and electromagnetic problems.
2. **Physics-informed neural networks** — Formulate neural approximations whose training objectives incorporate the governing PDEs, boundary conditions, and initial conditions.
3. **Electromagnetic modeling** — Progress from electrostatic potential problems to time-dependent field propagation and more general electromagnetic configurations.
4. **Verification and comparison** — Evaluate numerical error, convergence, physical consistency, robustness, and computational cost using reproducible experiments.

The initial milestones are:

- Reproduce and verify one-dimensional electrostatic Poisson problems.
- Extend the finite-difference formulation to two-dimensional Poisson problems.
- Train PINNs on the same electrostatic benchmark problems.
- Introduce time-dependent Maxwell problems and a finite-difference time-domain (FDTD) reference solver.
- Extend selected problems to two dimensions and compare with finite-element method (FEM) solutions.
- Investigate more challenging configurations, such as heterogeneous material parameters, nontrivial geometries, and inverse problems, where appropriate.

## Mathematical foundation

The first benchmark family concerns electrostatics. In a region with permittivity \(\epsilon\), the electrostatic potential \(\phi\) satisfies

$$
\mathbf{E} = -\nabla \phi,
\qquad
\nabla \cdot (\epsilon \nabla \phi) = -\rho,
$$

where \(\mathbf{E}\) is the electric field and \(\rho\) is the free charge density. For constant permittivity, this reduces to Poisson's equation,

$$
\nabla^2 \phi = -\frac{\rho}{\epsilon}.
$$

In charge-free regions, the equation becomes Laplace's equation, \(\nabla^2\phi=0\).

The broader project will address Maxwell's equations. In differential form, the macroscopic equations are

$$
\nabla \cdot \mathbf{D} = \rho_f,
\qquad
\nabla \cdot \mathbf{B} = 0,
$$

$$
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t},
\qquad
\nabla \times \mathbf{H} = \mathbf{J}_f + \frac{\partial \mathbf{D}}{\partial t}.
$$

Constitutive relations, material models, source terms, and boundary conditions will be specified for each benchmark. Electrostatic Poisson problems are an initial, restricted case; they are not a substitute for the full time-dependent Maxwell system.

## Numerical methods

### Finite-difference methods (FDM)

Finite-difference discretizations will provide baseline numerical solutions for selected PDEs. Initial work will focus on one- and two-dimensional Poisson equations, followed by time-dependent finite-difference time-domain methods.

The implementation and experiments will document grid spacing, discretization order, boundary treatment, stability restrictions where applicable, and convergence behavior.

### Finite-difference time-domain (FDTD)

FDTD methods will be used for selected time-dependent electromagnetic problems. The project will investigate field updates on spatial and temporal grids, source implementation, absorbing or reflecting boundaries, and numerical dispersion and stability.

### Finite-element methods (FEM)

FEM will provide an additional numerical comparison for suitable boundary-value problems and geometries. The exact implementation strategy and software dependencies will be documented as this part of the project develops.

### Physics-informed neural networks (PINNs)

PINNs will approximate unknown fields or potentials using neural networks. Training objectives will combine governing-equation residuals with the relevant boundary-condition, initial-condition, and, where appropriate, data-mismatch terms.

For a generic PDE residual, denoted by \(\mathcal{R}[u_\theta](\mathbf{x},t)\), the physics loss can be defined as the mean squared residual evaluated at \(N_r\) collocation points:

$$
\mathcal{L}_{\mathrm{PDE}} =
\frac{1}{N_r}
\sum_{i=1}^{N_r}
\left|\mathcal{R}[u_\theta](\mathbf{x}_i,t_i)\right|^2.
$$

The complete objective may include additional terms,

$$
\mathcal{L} =
\lambda_r\mathcal{L}_{\mathrm{PDE}}
+
\lambda_b\mathcal{L}_{\mathrm{BC}}
+
\lambda_i\mathcal{L}_{\mathrm{IC}}
+
\lambda_d\mathcal{L}_{\mathrm{data}},
$$

with terms included only when relevant to the problem. The effects of network architecture, collocation-point sampling, loss weighting, optimization, and problem scaling will be investigated rather than assumed to be inconsequential.

## Verification and evaluation

A central requirement is that every numerical result be checked independently. Planned evaluation criteria include:

- **Analytical agreement:** compare against closed-form solutions for problems where these are available.
- **Discretization convergence:** examine error as spatial and temporal resolution are refined.
- **Error metrics:** report suitable absolute and relative norms, including \(L^2\) and \(L^\infty\) errors where appropriate.
- **Physical consistency:** check relevant Maxwell constraints, boundary and initial conditions, and conservation properties.
- **Robustness:** investigate sensitivity to resolution, sampling, initialization, and model parameters.
- **Computational cost:** record training or solve time, memory requirements, and accuracy-versus-cost trade-offs.

PINNs will not be presumed superior to conventional solvers. A meaningful outcome may be that a classical method is more accurate, faster, or more reliable for a particular class of problems. The purpose is to determine where each method is appropriate.

## Planned repository organization

The directory structure will evolve as the implementation grows. The intended organization is:

```text
PINN-Maxwell-Equations/
├── README.md
├── docs/
│   ├── mathematical_foundations/
│   ├── numerical_methods/
│   └── methodology/
├── src/
│   └── maxwell_pinn/
│       ├── equations/
│       ├── finite_difference/
│       ├── finite_element/
│       ├── fdtd/
│       ├── pinn/
│       └── validation/
├── tests/
├── experiments/
│   ├── electrostatics/
│   ├── wave_propagation/
│   └── inverse_problems/
├── benchmarks/
├── results/
└── notebooks/
```

This is a proposed structure, not a claim that all directories or modules currently exist.

## Tools and technologies

The primary implementation environment is planned to be Python, with scientific-computing and machine-learning libraries selected as the solver implementations are developed. Potential tools include:

- **NumPy** and **SciPy** for numerical computation.
- **Matplotlib** for visualization.
- **PyTorch** or **JAX** for differentiable programming and PINN implementations.
- **Jupyter** for exploratory analysis and experiment notebooks.
- **MATLAB** for selected independent numerical comparisons where useful.

Exact dependencies, supported versions, and installation instructions will be added once the project has a working implementation and dependency configuration.

## Reproducibility

Experiments should be reproducible and independently inspectable. As the framework develops, experiment records will include:

- Governing equations, assumptions, units, and material parameters.
- Geometry, source terms, and boundary and initial conditions.
- Spatial and temporal discretization parameters.
- PINN architecture, training configuration, and random seeds.
- Solver tolerances and stopping criteria.
- Error metrics, runtime measurements, and generated results.

Code, figures, and numerical claims will be accompanied by enough methodological detail to make the comparisons interpretable.

## Contributing and development

This is an evolving research codebase. During early development, the priority is mathematical correctness, test coverage, clear interfaces, and reproducible benchmarks rather than maximizing the number of implemented methods.

Contributions should include tests or validation evidence for numerical changes. Changes to governing equations, discretizations, or boundary-condition handling should explain the mathematical assumptions being used.

## References

A formal bibliography will be maintained as the theoretical and numerical components are developed. Relevant references will include authoritative treatments of:

- Classical electromagnetics and Maxwell's equations.
- Numerical methods for elliptic and hyperbolic PDEs.
- Finite-difference and finite-element methods.
- Finite-difference time-domain computational electromagnetics.
- Physics-informed neural networks and scientific machine learning.

Specific references will be cited alongside the methods and experiments they support.

## License

This project is distributed under the license in the repository's `LICENSE` file. Please review that file for the applicable terms.

---

**Project focus:** computational electromagnetics · applied mathematics · numerical PDEs · physics-informed machine learning · scientific computing
