## Reflecting boundaries

Create a model with (for example)
```jl
T = [-2.5 2 0.5; 1 -2 1; 1 2 -3] # generator of the phase
C = [0.0; 2.0; -3.0]    # rates dX/dt
b = 10.0 # upper boundary of fluid queue
P_lwr = [0.0 1.0 0.0] # transition probabilities upon hitting lower boundary
P_upr = [0.0 0.0 1.0] # transition probabilities upon hitting upper boundary

S = PhaseSet(C) 
model = BoundedFluidQueue(T,S,P_lwr,P_upr,b) 

# Create a discretisation mesh (a grid + method with which to approximate the solution) 
nodes = 0.0:1.0:model.b
order = 3
mesh = FRAPMesh(nodes,order)

# Combine the model and the discretisation scheme (mesh) to form a discretised fluid queue
dq = DiscretisedFluidQueue(model,mesh)

B = build_full_generator(dq)

# initial distribution 
f(x,i) = (i-1)/model.b./sum((1:3).-1) # the initial distribution
d = SFMDistribution(f,dq)

# Integrate over time with 
t = 3.2
dt = integrate_time(d,B,t,StableRK4(0.01))

u = pdf(dt)

# Evaluate solution as a function 
u(0.1,1) # = P(X(3.2)<0.1, phi(3.2)=1)
```

## Slope limited DG scheme and point mass initial conditions

Create a model with (for example)
```jl
# Create a discretisation mesh (a grid + method with which to approximate the solution) 
nodes = 0.0:1.0:model.b
order = 3
mesh = DGMesh(nodes,order)

# Combine the model and the discretisation scheme (mesh) to form a discretised fluid queue
dq = DiscretisedFluidQueue(model,mesh)

B = build_full_generator(dq)

# initial distribution 
d = interior_point_mass(eps(),1,dq) # point mass at x=eps() in phase=1

# Integrate over time with 
dt = integrate_time(d,B,t,StableRK4(0.01);limiter=GeneralisedMUSCL)

u = cdf(dt)

# Evaluate solution as a function 
u(0.1,1) # = P(X(3.2)<0.1, phi(3.2)=1)
```


## Simulate

Create a model with (for example)
```jl
n_sim = 10_000
sims = simulate(model,fixed_time(t),(X=fill(eps(),n_sim), φ=ones(Int,n_sim)))

u = cdf(sims)

# Evaluate solution as a function 
u(0.1,1) # = P(X(3.2)<0.1, phi(3.2)=1)
```