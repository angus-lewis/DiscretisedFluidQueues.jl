# DiscretisedFluidQueues
### Numerical approximation schemes for stochastic fluid queues.

The evolution of a stochastic fluid queue with generator $[T_{ij}]_{i,j\in S}$ and associated diagonal matrix of rates $C = diag(c_i,i\in S)$ can be described by the PDE (when it exists)

$\cfrac{\partial}{\partial t}  \mathbf f(x,i,t) =  \mathbf f(x,i,t) T - \cfrac{\partial}{\partial x}  \mathbf f(x,i,t) C,$

where $\mathbf f(x,i,t) dx = (f(x,i,t))_{i\in S} = (P(X(t)\in dx, \varphi(t)=i))_{i\in S}$ the time-dependent joint density/mass function. 

This package implements finite element and finite volume numerical solvers to approximate the right-hand side of this PDE; 

* Discontinuous Galerkin: projects the right-hand side of the PDE on to a basis of polynomials,
* Finite volume:
* QBD-RAP approximation: use matrix-exponential distributions to model the solution locally on each cell.

---
### Usage
```jl
pkg> add https://github.com/angus-lewis/DiscretisedFluidQueues
```
```jl
julia> using DiscretisedFluidQueues
```

Create a model with (for example)
```jl
T = [-2.5 2 0.5; 1 -2 1; 1 2 -3] # generator of the phase
C = [0.0; 2.0; -3.0]    # rates dX/dt
b = 10.0 # upper boundary of fluid queue

S = PhaseSet(C) 

model = BoundedFluidQueue(T,S,b) 
```

Create a discretisation mesh (a grid + method with which to approximate the solution) with any of (e.g.)
```jl
nodes = 0.0:1.0:model.b

nbases = 3
mesh = DGMesh(nodes,nbases)

fv_order = 3
fvmesh = FVMesh(collect(nodes),fv_order)

order = 3
frapmesh = FRAPMesh(nodes,order)
```

Combine the model and the discretisation scheme (mesh) to form a discretised fluid queue
```
dq = DiscretisedFluidQueue(model,mesh)
# or dq = DiscretisedFluidQueue(model,fvmesh)
# or dq = DiscretisedFluidQueue(model,frapmesh)
```

Construct an approximation to the generator with 
```jl
B = build_full_generator(dq)
```
`B` is essentially a matrix which we can think of as describing the ODE

$\frac{d}{d t}   \mathbf a(t) =   \mathbf a(t) B,$

where $ \mathbf a(t)$ is a row vector of coefficients and $  \mathbf a(t)  \mathbf u(x,i) \approx  \mathbf f(x,i,t)$ approximates the solution where $ \mathbf u(x,i)$ is a column vector of functions defined by the discretisation scheme.

Construct an initial distribution with (e.g.)
```jl
f(x,i) = (i-1)/model.b./sum((1:3).-1) # the initial distribution
d = SFMDistribution(f,dq)
```

Integrate over time with 
```jl
t = 3.2
dt = integrate_time(d,B,t,StableRK4(0.01);limiter=NoLimiter) # no slope limiter
```

Reconstruct approximate solution from coefficients with 
```jl
u = pdf(dt)
# or u = cdf(dt)
```
Evaluate solution as a function 
```jl
u(0.1,1) # = P(X(3.2)<0.1, phi(3.2)=1)
```

This is my PhD topic; find the [thesis here.](https://github.com/angus-lewis/PhDThesis)


