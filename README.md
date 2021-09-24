# DiscretisedFluidQueues
#### Numerical approximation schemes for stochastic fluid queues.

The evolution of stochastic fluid queues can be described by the PDE (when it exists)
\begin{equation}
    \frac{\partial}{\partial t} \mathbf f(x,i,t) = \mathbf f(x,i,t) T - \frac{\partial}{\partial x} \mathbf f(x,i,t) C,
\end{equation}
where $\mathbf f(x,i,t) dx = P(X(t)\in dx, \varphi(t)=i)$ the time-dependent joint density/mass function. 

This package implements finite element and finite volume numerical solvers to approximate the right-hand side of this PDE; 
    + Discontinuous Galerkin: projects the right-hand side of the PDE on to a basis of polynomials,
    + Finite volume:
    + QBD-RAP approximation: use matrix-exponential distributions to model the solution locally on each cell.

```jl
pkg> add https://github.com/angus-lewis/DiscretisedFluidQueues
```
```jl
julia> import DiscretisedFluidQueues
```

Create a model with (for example)
```jl
T = [-2.5 2 0.5; 1 -2 1; 1 2 -3] # generator of the phase
C = [0.0; 2.0; -3.0]    # rates dX/dt

S = DiscretisedFluidQueues.PhaseSet(C) # constructor for phases

bounds = [0.0,12]
model = DiscretisedFluidQueues.FluidQueue(T,S,bounds) # model object
```

Create a mesh (a grid over which to approximate the solution) with (e.g.)
```jl
nbases = 3
dgmesh = DiscretisedFluidQueues.DGMesh(nodes,nbases)

fv_order = 3
fvmesh = DiscretisedFluidQueues.FVMesh(nodes,fv_order)

order = 3
frapmesh = DiscretisedFluidQueues.FRAPMesh(nodes,order)
```

Construct an approximation to the generator with 
```jl
B = StochasticFluidQueues.MakeFullGenerator(am,mesh)
```
`B` is essentially a matrix which we can think of as describing the ODE
\begin{equation}
    \frac{\partial}{\partial t} \mathbf a(t) = \mathbf a(t) B
\end{equation}
where $\mathbf a(t)u(x) \approx f(x,i,t)$ approximates the solution.

Construct an initial distribution with (e.g.)
```jl
f(x,i) = (i-1)/12.0./sum(1:3) # the initial distribution
d = StochasticFluidQueues.SFMDistribution(f,model,mesh)
```

Integrate over time with 
```jl
t = 3.2
dt = StochasticFluidQueues.integrate_time(d,B,t)
```

Reconstruct approximate solution with 
```jl
u = StochasticFluidQueues.cdf(d)
```
Evaluate solution as a function 
```jl
u(0.1,1) # = P(X(3.2)<0.1, phi(3.2)=1)
```

More examples and documentation to come. 

This is my PhD topic. I'll provide a link to a write up when I've done one...
