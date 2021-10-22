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
        lwr::Float64, upr::Float64,
        StoppingTime::Function,
        InitCondition::NamedTuple{(:φ, :X)},
    )

# Arguments
- `model`: A Model object 
- `lwr` and `upr` upper an lower regulated boundaries for the fluid level.
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
    lwr::Float64, upr::Float64,
    StoppingTime::Function,
    InitCondition::NamedTuple{(:φ, :X)},
    rng::Random.AbstractRNG=Random.default_rng(),
)
    # the transition matrix of the jump chain
    d = LinearAlgebra.diag(model.T)
    P = (model.T - LinearAlgebra.diagm(0 => d)) ./ -d
    CumP = cumsum(P, dims = 2)
    Λ = LinearAlgebra.diag(model.T)

    M = length(InitCondition.φ)
    tSims = zeros(Float64, M)
    φSims = zeros(Int, M)
    XSims = zeros(Float64, M)
    nSims = zeros(Int, M)

    for m = 1:M
        SFM0 = (t = 0.0, φ = InitCondition.φ[m], X = InitCondition.X[m], n = 0)
        while 1 == 1
            S = log(rand(rng)) / Λ[SFM0.φ] # generate exp random variable
            t = SFM0.t + S
            X = UpdateXt(model, SFM0, S, lwr, upr)
            φ = findfirst(rand(rng) .< CumP[SFM0.φ, :])
            n = SFM0.n + 1
            SFM = (t = t, φ = φ, X = X, n = n)
            τ = StoppingTime(model, SFM, SFM0, lwr, upr)
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
    function F(x::Float64,i::Int)
        !(i∈phases(s.model)) && throw(DomainError("phase i must be in the support of the model"))
        i_idx = s.φ.==i
        Fxi = sum(s.X[i_idx].<=x)/n_sims
        return Fxi
    end
    return F
end

"""
Returns ``X(t+S) = min(max(X(t) + cᵢS,0),U)`` where ``U`` is some upper bound
on the process.

    UpdateXt(
        model::Model,
        SFM0::NamedTuple,
        S::Real,
        lwr::Float64,
        upr::Float64,
    )

# Arguments
- `model`: a Model object
- `SFM0::NamedTuple` containing at least the keys `:X` giving the value of
    ``X(t)`` at the current time, and `:φ` giving the value of
    ``φ(t)`` at the current time.
- `S::Real`: an elapsed amount of time to evaluate ``X`` at, i.e. ``X(t+S)``.
- `lwr::Float64`: lower regulated boundary for the fluid level
- `upr::Float64`: upper regulated boundary for the fluid level
"""
function UpdateXt(
    model::Model,
    SFM0::NamedTuple,
    S::Real,
    lwr::Float64,
    upr::Float64,
)
    # given the last position of a SFM, SFM0, a time step of size s, find the
    # position of X at time t
    X = min( max(SFM0.X+rates(model,SFM0.φ)*S, lwr), upr)
    return X
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
        lwr::Float64,
        upr::Float64,
    )`: a stopping time for a SFM.
"""
function fixed_time(T::Float64)
    # Defines a simple stopping time, 1(t>T).
    # SFM method
    function fixed_timeFun(
        model::Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
        lwr::Float64,
        upr::Float64,
    )
        Ind = SFM.t > T
        if Ind
            s = T - SFM0.t
            X = UpdateXt(model, SFM0, s, lwr, upr)
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
        lwr::Float64,
        upr::Float64,
    )`: a stopping time for a SFM.
"""
function n_jumps( N::Int)
    # Defines a simple stopping time, 1(n>N), where n is the number of jumps of φ.
    # SFM method
    function n_jumpsFun(
        model::Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
        lwr::Float64,
        upr::Float64,
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
        lwr::Float64,
        upr::Float64,
    )`: a stopping time for a SFM.
"""
function first_exit_x( u::Real, v::Real)
    # SFM Method
    function first_exit_xFun(
        model::Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
        lwr::Float64,
        upr::Float64,
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
