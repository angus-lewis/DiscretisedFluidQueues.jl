import Random

"""
    Simulation

A container to hold simulations. 

All arguments are arrays of the same ength + a model from which the simulations came

# Arguments:
- `t::Array{Float64, 1}`: 
- `φ::Array{Int, 1}`: 
- `X::Array{Float64, 1}`: 
- `n::Array{Int, 1}`: 
- `model::Model`: 
"""
struct Simulation
    # t, times at which SFM is observed (realisations of stopping times)
    # X, array of X(t) values
    # φ, array of φ(t) values 
    # n, number of phase changes by t, n(t)
    # model, the model from which the sims came
    t::Array{Float64,1}
    φ::Array{Int,1}
    X::Array{Float64,1}
    n::Array{Int,1}
    model::Model
    function Simulation(t::Array{Float64,1},φ::Array{Int,1},
        X::Array{Float64,1},n::Array{Int,1},model::Model)
        !(length(t)==length(φ)==length(X)==length(n))&&throw(DomainError("lenght must all be the same"))
        new(t,φ,X,n,model)
    end
end

"""
Simulates a SFM defined by `Model` until the `StoppingTime` has occured,
given the `InitialCondition` on (φ(0),X(0)).

    simulate(
        model::Model,
        StoppingTime::Function,
        InitCondition::NamedTuple{(:φ, :X)},
    )

# Arguments
- `model`: A Model object 
- `StoppingTime`: A function which takes the value of the process at the current
    time and at the time of the last jump of the phase process, as well as the
    `model` object.
    i.e. `StoppingTime(;model,SFM,SFM0,lwer,upr)` where `SFM` and `SFM0` are tuples with
    keys `(:t::Float64, :φ::Int, :X::Float64, :n::Int)` which are the value of
    the SFM at the current time, and time of the previous jump of the phase
    process, repsectively. The `StoppingTime` must return a
    `NamedTuple{(:Ind, :SFM)}` type where `:Ind` is a `:Bool` value stating
    whether the stopping time has occured or not and `:SFM` is a tuple in the
    same form as the input `SFM` but which contains the value of the `SFM` at
    the stopping time.
- `InitCondition`: `NamedTuple` with keys `(:φ, :X)`, `InitCondition.φ` is a
    vector of length `M` of initial states for the phase, and `InitCondition.X`
    is a vector of length `M` of initial states for the level. `M` is the number
    of simulations to be done.

# Output
- A `Simulation` object.
"""
function simulate(
    model::BoundedFluidQueue,
    StoppingTime::Function,
    InitCondition::NamedTuple{(:X,:φ)},
    rng::Random.AbstractRNG=Random.default_rng(),
)
    for n in 1:length(InitCondition.X)
        !(0.0<=InitCondition.X[n]<=model.b)&&throw(DomainError("inital X values must be within the model bounds"))
        !(InitCondition.φ[n]∈(1:n_phases(model)))&&throw(DomainError("inital φ values must be within the model phases"))
    end
    # the transition matrix of the jump chain
    d = LinearAlgebra.diag(model.T)
    P = (model.T - LinearAlgebra.diagm(0 => d)) ./ -d
    CumP = cumsum(P, dims = 2)

    idx₋ = findall(rates(model).<0.0)
    CumP_lwr = zeros(size(P))
    CumP_lwr[idx₋,:] = cumsum(model.P_lwr, dims = 2)
    CumP_upr = zeros(size(P))
    idx₊ = findall(rates(model).>0.0)
    CumP_upr[idx₊,:] = cumsum(model.P_upr, dims = 2)

    Λ = LinearAlgebra.diag(model.T)

    M = length(InitCondition.φ)
    tSims = zeros(Float64, M)
    φSims = zeros(Int, M)
    XSims = zeros(Float64, M)
    nSims = zeros(Int, M)

    for m = 1:M
        SFM0 = (t = 0.0, φ = InitCondition.φ[m], X = InitCondition.X[m], n = 0)
        while true
            S = log(rand(rng)) / Λ[SFM0.φ] # generate exp random variable
            X, φ, t = Update(model, SFM0, S, CumP, CumP_lwr, CumP_upr, rng)
            n = SFM0.n + 1
            SFM = (t = t, φ = φ, X = X, n = n)
            τ = StoppingTime(model, SFM, SFM0)
            if τ.Ind # if the stopping time occurs
                (tSims[m], φSims[m], XSims[m], nSims[m]) = τ.SFM
                break
            end
            SFM0 = SFM
        end
    end
    return Simulation(tSims, φSims, XSims, nSims, model)
end

function pdf(s::Simulation)
    throw(DomainError("cannot reliably construct pdf from simulation,
        either do it yourself, or construct the cdf, see cdf() function"))
end

"""
    cdf(s::Simulation)

Return a empirical cdf of the fluid queue given a simulation. 

Output is a function of two variables (x,i) and gives the empirical distribution 
    function ``P(X(τ)≤x,φ(τ)=i)``
"""
function cdf(s::Simulation)
    n_sims = length(s.t)
    function F(x::Real,i::Int)
        !(i∈phases(s.model)) && throw(DomainError("phase i must be in the support of the model"))
        i_idx = s.φ.==i
        Fxi = sum(s.X[i_idx].<=x)/n_sims
        return Fxi
    end
    return F
end

function cell_probs(s::Simulation,nodes)
    n_sims = length(s.t)
    function p(x::Real,i::Int)
        !(i∈phases(s.model)) && throw(DomainError("phase i must be in the support of the model"))
        _x_in_bounds = (x>nodes[1])&&(x<nodes[end])
        if _x_in_bounds
            idx = findfirst(x.<nodes)
            left_edge = nodes[idx-1]
            right_edge = nodes[idx]
            i_idx = s.φ.==i
            return (membership(s.model.S,i)>0.0) ? (sum(left_edge.<=s.X[i_idx].<right_edge)/n_sims) : (sum(left_edge.<s.X[i_idx].<=right_edge)/n_sims)
        else 
            return 0.0
        end
    end
    return p
end

"""
Returns ``X(t+S) = min(max(X(t) + cᵢS,0),model.b)`` where ``model.b`` is the upper boundary of the model
on the process.

    UpdateX(
        model::Model,
        SFM0::NamedTuple,
        S::Real,
    )

# Arguments
- `model`: a Model object
- `SFM0::NamedTuple` containing at least the keys `:X` giving the value of
    ``X(t)`` at the current time, and `:φ` giving the value of
    ``φ(t)`` at the current time.
- `S::Real`: an elapsed amount of time to evaluate ``X`` at, i.e. ``X(t+S)``.
"""
function UpdateX(model::BoundedFluidQueue,SFM0::NamedTuple,S::Real)
    # given the last position of a SFM, SFM0, a time step of size s, find the
    # position of X at time t
    
    # X moves freely unless a boundary is hit
    X = min( max(SFM0.X+rates(model,SFM0.φ)*S, 0.0), model.b)
    return X
end

function Update(model::BoundedFluidQueue,SFM0::NamedTuple,S::Real,CumP::Matrix{Float64},
    CumP_lwr::Matrix{Float64},CumP_upr::Matrix{Float64},rng::Random.AbstractRNG=Random.default_rng())

    # evolve freely until a boundary is hit
    X =  UpdateX(model,SFM0,S)

    # if boundary is hit, change phase, update time to time when boundary is hit
    if (X==0.0)&&(SFM0.X!=0.0)#&&(rates(model,SFM0.φ)<0.0)
        φ = findfirst(rand(rng).<=CumP_lwr[SFM0.φ,:])
        t = SFM0.t + (X-SFM0.X)/rates(model,SFM0.φ)
    elseif (X==model.b)&&(SFM0.X!=model.b)#&&(rates(model,SFM0.φ)>0.0)
        φ = findfirst(rand(rng).<=CumP_upr[SFM0.φ,:])
        t = SFM0.t + (X-SFM0.X)/rates(model,SFM0.φ)
    else # update phase
        φ = findfirst(rand(rng).<=CumP[SFM0.φ, :])
        t = SFM0.t + S
    end
    return X, φ, t
end

"""
Constructs the `StoppingTime` ``1(t>T)``

    fixed_time( T::Real)

# Arguments
- `T`: a time at which to stop the process

# Output
- `fixed_timeFun`: a function with one methods
    - `fixed_timeFun(
        model::Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )`: a stopping time for a SFM.
"""
function fixed_time(T::Float64)
    # Defines a simple stopping time, 1(t>T).
    # SFM method
    function fixed_timeFun(
        model::Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )
        Ind = SFM.t > T
        if Ind
            s = T - SFM0.t
            X = UpdateX(model, SFM0, s)
            SFM = (t = T, φ = SFM0.φ, X = X, n = SFM0.n)
        end
        return (Ind = Ind, SFM = SFM)
    end
    return fixed_timeFun
end

"""
Constructs the `StoppingTime` ``1(N(t)>n)`` where ``N(t)`` is the number of
jumps of ``φ`` by time ``t``.

    n_jumps( N::Int)

# Arguments
- `N`: a desired number of jumps

# Output
- `n_jumpsFun`: a function with one methods
    - `n_jumpsFun(
        model::Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )`: a stopping time for a SFM.
"""
function n_jumps( N::Int)
    # Defines a simple stopping time, 1(n>N), where n is the number of jumps of φ.
    # SFM method
    function n_jumpsFun(
        model::Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )
        Ind = SFM.n >= N
        return (Ind = Ind, SFM = SFM)
    end
    return n_jumpsFun
end

"""
Constructs the `StoppingTime` which is the first exit of the process ``X(t)``
from the interval ``[u,v]``.

    first_exit_x( u::Real, v::Real)

# Arguments
- `u`: a lower boundary
- `v`: an upper boundary

# Output
- `first_exit_xFun`: a function with one methods
    - `first_exit_xFun(
        model::Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )`: a stopping time for a SFM.
"""
function first_exit_x( u::Real, v::Real)
    # SFM Method
    function first_exit_xFun(
        model::Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )
        Ind = ((SFM.X > v) || (SFM.X < u))
        if Ind
            if SFM.X > v
                X = v
            else
                X = u
            end
            s = (X - SFM0.X) / rates(model,SFM0.φ) # can't exit with c = 0
            t = SFM0.t + s
            SFM = (t = t, φ = SFM0.φ, X = X, n = SFM0.n)
        end
        return (Ind = Ind, SFM = SFM)
    end
    return first_exit_xFun
end
