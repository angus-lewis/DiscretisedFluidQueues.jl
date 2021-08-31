"""


Constructor for SimMesh object. 
"""
struct SimMesh <: Mesh 
    Nodes::Array{Float64,1}
    NBases::Int
    Fil::Dict{String,BitArray{1}}
end 
function SimMesh(
    model::Model,
    Nodes::Array{Float64,1},
    NBases::Int;
    Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
)
    NIntervals = length(Nodes) - 1 # the number of intervals
    Δ = (Nodes[2:end] - Nodes[1:end-1]) # interval width
    CellNodes = zeros(Float64, NBases, NIntervals)
    for i = 1:NIntervals
        CellNodes[:, i] .= (Nodes[i+1] + Nodes[i]) / 2 
    end
    TotalNBases = NBases * NIntervals

    ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
    if isempty(Fil)
        idxPlus = model.r.r(Nodes[1:end-1].+Δ[:]/2).>0
        idxZero = model.r.r(Nodes[1:end-1].+Δ[:]/2).==0
        idxMinus = model.r.r(Nodes[1:end-1].+Δ[:]/2).<0
        for i in 1:NPhases(model)
            Fil[string(i)*"+"] = idxPlus[:,i]
            Fil[string(i)*"0"] = idxZero[:,i]
            Fil[string(i)*"-"] = idxMinus[:,i]
            if model.C[i] .<= 0
                Fil["p"*string(i)*"+"] = [model.r.r(model.Bounds[1,1])[i]].>0
                Fil["p"*string(i)*"0"] = [model.r.r(model.Bounds[1,1])[i]].==0
                Fil["p"*string(i)*"-"] = [model.r.r(model.Bounds[1,1])[i]].<0
            end
            if model.C[i] .>= 0
                Fil["q"*string(i)*"+"] = [model.r.r(model.Bounds[1,end])[i]].>0
                Fil["q"*string(i)*"0"] = [model.r.r(model.Bounds[1,end])[i]].==0
                Fil["q"*string(i)*"-"] = [model.r.r(model.Bounds[1,end])[i]].<0
            end
        end
    end
    CurrKeys = keys(Fil)
    for ℓ in ["+", "-", "0"], i = 1:NPhases(model)
        if !in(string(i) * ℓ, CurrKeys)
            Fil[string(i)*ℓ] = falses(NIntervals)
        end
        if !in("p" * string(i) * ℓ, CurrKeys) && model.C[i] <= 0
            Fil["p"*string(i)*ℓ] = falses(1)
        end
        if !in("p" * string(i) * ℓ, CurrKeys) && model.C[i] > 0
            Fil["p"*string(i)*ℓ] = falses(0)
        end
        if !in("q" * string(i) * ℓ, CurrKeys) && model.C[i] >= 0
            Fil["q"*string(i)*ℓ] = falses(1)
        end
        if !in("q" * string(i) * ℓ, CurrKeys) && model.C[i] < 0
            Fil["q"*string(i)*ℓ] = falses(0)
        end
    end
    for ℓ in ["+", "-", "0"]
        Fil[ℓ] = falses(NIntervals * NPhases(model))
        Fil["p"*ℓ] = trues(0)
        Fil["q"*ℓ] = trues(0)
        for i = 1:NPhases(model)
            idx = findall(Fil[string(i)*ℓ]) .+ (i - 1) * NIntervals
            Fil[string(ℓ)][idx] .= true
            Fil["p"*ℓ] = [Fil["p"*ℓ]; Fil["p"*string(i)*ℓ]]
            Fil["q"*ℓ] = [Fil["q"*ℓ]; Fil["q"*string(i)*ℓ]]
        end
    end
    return SimMesh(NBases, CellNodes, Fil, Δ, NIntervals, Nodes, TotalNBases, "lagrange")
end
function SimMesh()
    return SimMesh(
        0,
        Array{Real,2}(undef,0,0),
        Dict{String,BitArray{1}}(),
        Array{Float64,1}(undef,0),
        0,
        Array{Float64,1}(undef,0),
        0,
        "lagrange",
    )
end

"""
Simulates a SFM defined by `Model` until the `StoppingTime` has occured,
given the `InitialCondition` on (φ(0),X(0)).

    SimSFM(
        model::SFFM.Model,
        StoppingTime::Function,
        InitCondition::NamedTuple{(:φ, :X)},
    )

# Arguments
- `model`: A Model object 
- `StoppingTime`: A function which takes the value of the process at the current
    time and at the time of the last jump of the phase process, as well as the
    `model` object.
    i.e. `StoppingTime(;model,SFM,SFM0)` where `SFM` and `SFM0` are tuples with
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
- a tuple with keys
    - `t::Array{Float64,1}` a vector of length `M` containing the values of
        `t` at the `StoppingTime`.
    - `φ::Array{Float64,1}` a vector of length `M` containing the values of
        `φ` at the `StoppingTime`.
    - `X::Array{Float64,1}` a vector of length `M` containing the values of
        `X` at the `StoppingTime`.
    - `n::Array{Float64,1}` a vector of length `M` containing the number of
        transitions of `φ` at the `StoppingTime`
"""
function SimSFM(
    model::SFFM.Model,
    StoppingTime::Function,
    InitCondition::NamedTuple{(:φ, :X)},
)
    # the transition matrix of the jump chain
    d = LinearAlgebra.diag(model.T)
    P = (model.T - LinearAlgebra.diagm(0 => d)) ./ -d
    CumP = cumsum(P, dims = 2)
    Λ = LinearAlgebra.diag(model.T)

    M = length(InitCondition.φ)
    tSims = zeros(Float64, M)
    φSims = zeros(Float64, M)
    XSims = zeros(Float64, M)
    nSims = zeros(Float64, M)

    for m = 1:M
        SFM0 = (t = 0.0, φ = InitCondition.φ[m], X = InitCondition.X[m], n = 0)
        while 1 == 1
            S = log(rand()) / Λ[SFM0.φ] # generate exp random variable
            t = SFM0.t + S
            X = UpdateXt(model, SFM0, S)
            φ = findfirst(rand() .< CumP[SFM0.φ, :])
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
    return (t = tSims, φ = φSims, X = XSims, n = nSims)
end

"""
Simulates a SFFM defined by `model` until the `StoppingTime` has occured,
given the `InitialCondition` on (φ(0),X(0),Y(0)).

    SimSFFM(
        model::SFFM.Model,
        StoppingTime::Function,
        InitCondition::NamedTuple{(:φ, :X, :Y)},
    )

# Arguments
- `model`: A Model object
- `StoppingTime`: A function which takes the value of the process at the current
    time and at the time of the last jump of the phase process, as well as the
    `model` object.
    i.e. `StoppingTime(;model,SFFM,SFFM0)` where `SFFM` and `SFFM0` are tuples
    with keys `(:t::Float64, :φ::Int, :X::Float64, :Y::Float64, :n::Int)` which
    are the value of the SFFM at the current time, and time of the previous jump
    of the phase process, repsectively. The `StoppingTime` must return a
    `NamedTuple{(:Ind, :SFFM)}` type where `:Ind` is a `:Bool` value stating
    whether the stopping time has occured or not and `:SFFM` is a tuple in the
    same form as the input `SFFM` but which contains the value of the SFFM at
    the stopping time.
- `InitCondition`: `NamedTuple` with keys `(:φ, :X, :Y)`, `InitCondition.φ` is a
    vector of length `M` of initial states for the phase, `InitCondition.X` is a
    vector of length `M` of initial states for the X-level, `InitCondition.Y` is
    a vector of length `M` of initial states for the Y-level. `M` is the number
    of simulations to be done.

# Output
- a tuple with keys
    - `t::Array{Float64,1}` a vector of length `M` containing the values of
        `t` at the `StoppingTime`.
    - `φ::Array{Float64,1}` a vector of length `M` containing the values of
        `φ` at the `StoppingTime`.
    - `X::Array{Float64,1}` a vector of length `M` containing the values of
        `X` at the `StoppingTime`.
    - `Y::Array{Float64,1}` a vector of length `M` containing the values of
        `Y` at the `StoppingTime`.
    - `n::Array{Float64,1}` a vector of length `M` containing the number of
        transitions of `φ` at the `StoppingTime`
"""
function SimSFFM(
    model::SFFM.Model,
    StoppingTime::Function,
    InitCondition::NamedTuple{(:φ, :X, :Y)},
)
    d = LinearAlgebra.diag(model.T)
    P = (model.T - LinearAlgebra.diagm(0 => d)) ./ -d
    CumP = cumsum(P, dims = 2)
    Λ = LinearAlgebra.diag(model.T)

    M = length(InitCondition.φ)
    tSims = Array{Float64,1}(undef, M)
    φSims = Array{Float64,1}(undef, M)
    XSims = Array{Float64,1}(undef, M)
    YSims = Array{Float64,1}(undef, M)
    nSims = Array{Float64,1}(undef, M)

    for m = 1:M
        SFFM0 = (
            t = 0.0,
            φ = InitCondition.φ[m],
            X = InitCondition.X[m],
            Y = InitCondition.Y[m],
            n = 0.0,
        )
        if !(model.Bounds[1, 1] <= SFFM0.X <= model.Bounds[1, 2]) ||
           !(model.Bounds[2, 1] <= SFFM0.Y <= model.Bounds[2, 2]) ||
           !in(SFFM0.φ, 1:NPhases(model))
            (tSims[m], φSims[m], XSims[m], YSims[m], nSims[m]) =
                (t = NaN, φ = NaN, X = NaN, Y = NaN, n = NaN)
        else
            while 1 == 1
                S = log(rand()) / Λ[SFFM0.φ]
                t = SFFM0.t + S
                X = UpdateXt(model, SFFM0, S)
                Y = UpdateYt(model, SFFM0, S)
                φ = findfirst(rand() .< CumP[SFFM0.φ, :])
                n = SFFM0.n + 1.0
                SFFM = (t = t, φ = φ, X = X, Y = Y, n = n)
                τ = StoppingTime(model, SFFM, SFFM0)
                if τ.Ind
                    (tSims[m], φSims[m], XSims[m], YSims[m], nSims[m]) = τ.SFFM
                    break
                end
                SFFM0 = SFFM
            end
        end
    end
    return (t = tSims, φ = φSims, X = XSims, Y = YSims, n = nSims)
end

"""
Returns ``X(t+S) = min(max(X(t) + cᵢS,0),U)`` where ``U`` is some upper bound
on the process.

    UpdateXt(
        model::SFFM.Model,
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
function UpdateXt(
    model::SFFM.Model,
    SFM0::NamedTuple,
    S::Real,
)
    # given the last position of a SFM, SFM0, a time step of size s, find the
    # position of X at time t
    X = min(
        max(SFM0.X + model.C[SFM0.φ] * S, model.Bounds[1, 1]),
        model.Bounds[1, 2],
    )
    return X
end

"""
Returns ``Y(t+S)`` given ``Y(t)``.

    UpdateYt(
        model::SFFM.Model,
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
        S::Real,
    )

# Arguments
- `model`: a Model object
- `SFFM0::NamedTuple` containing at least the keys `:X` giving the value of
    ``X(t)`` at the current time, and `:Y` giving the value of ``Y(t)`` at the
    current time, and `:φ` giving the value of `φ(t)`` at the current time.
- `S::Real`: an elapsed amount of time to evaluate ``X`` at, i.e. ``X(t+S)``.
"""
function UpdateYt(
    model::SFFM.Model,
    SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    S::Real,
)
    # given the last position of a SFFM, SFFM0, a time step of size s, find the
    # position of Y at time t
    if model.C[SFFM0.φ] == 0
        Y = SFFM0.Y + S * model.r.r(SFFM0.X)[SFFM0.φ]
    else
        X = UpdateXt(model, SFFM0, S)
        ind = (X.==model.Bounds[1, :])[:]
        if any(ind)
            # time at which Xt hits u or v
            t0 = (model.Bounds[1, ind][1] - SFFM0.X) / model.C[SFFM0.φ]
            Y =
                SFFM0.Y +
                (model.r.R(X)[SFFM0.φ] - model.r.R(SFFM0.X)[SFFM0.φ]) /
                model.C[SFFM0.φ] +
                model.r.r(model.Bounds[1, ind][1])[SFFM0.φ] * (S - t0)
        else
            Y =
                SFFM0.Y +
                (model.r.R(X)[SFFM0.φ] - model.r.R(SFFM0.X)[SFFM0.φ]) /
                model.C[SFFM0.φ]
        end
    end
    return Y
end

"""
Constructs the `StoppingTime` ``1(t>T)``

    FixedTime( T::Real)

# Arguments
- `T`: a time at which to stop the process

# Output
- `FixedTimeFun`: a function with two methods
    - `FixedTimeFun(
        model::SFFM.Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )`: a stopping time for a SFM.
    - `FixedTimeFun(
        model::SFFM.Model,
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )`: a stopping time for a SFFM
"""
function FixedTime( T::Real)
    # Defines a simple stopping time, 1(t>T).
    # SFM method
    function FixedTimeFun(
        model::SFFM.Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )
        Ind = SFM.t > T
        if Ind
            s = T - SFM0.t
            X = UpdateXt(model, SFM0, s)
            SFM = (t = T, φ = SFM0.φ, X = X, n = SFM0.n)
        end
        return (Ind = Ind, SFM = SFM)
    end
    # SFFM METHOD
    function FixedTimeFun(
        model::SFFM.Model,
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )
        Ind = SFFM.t > T
        if Ind
            s = T - SFFM0.t
            X = UpdateXt(model, SFFM0, s)
            Y = UpdateYt(model, SFFM0, s)
            SFFM = (T, SFFM0.φ, X, Y, SFFM0.n)
        end
        return (Ind = Ind, SFFM = SFFM)
    end
    return FixedTimeFun
end

"""
Constructs the `StoppingTime` ``1(N(t)>n)`` where ``N(t)`` is the number of
jumps of ``φ`` by time ``t``.

    NJumps( N::Int)

# Arguments
- `N`: a desired number of jumps

# Output
- `NJumpsFun`: a function with two methods
    - `NJumpsFun(
        model::SFFM.Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )`: a stopping time for a SFM.
    - `NJumpsFun(
        model::SFFM.Model,
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )`: a stopping time for a SFFM
"""
function NJumps( N::Int)
    # Defines a simple stopping time, 1(n>N), where n is the number of jumps of φ.
    # SFM method
    function NJumpsFun(
        model::SFFM.Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )
        Ind = SFM.n >= N
        return (Ind = Ind, SFM = SFM)
    end
    # SFFM method
    function NJumpsFun(
        model::SFFM.Model,
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )
        Ind = SFFM.n >= N
        return (Ind = Ind, SFFM = SFFM)
    end
    return NJumpsFun
end

"""
Constructs the `StoppingTime` which is the first exit of the process ``X(t)``
from the interval ``[u,v]``.

    FirstExitX( u::Real, v::Real)

# Arguments
- `u`: a lower boundary
- `v`: an upper boundary

# Output
- `FirstExitXFun`: a function with two methods
    - `FirstExitXFun(
        model::SFFM.Model,
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )`: a stopping time for a SFM.
    - `FirstExitXFun(
        model::SFFM.Model,
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )`: a stopping time for a SFFM
"""
function FirstExitX( u::Real, v::Real)
    # SFM Method
    function FirstExitXFun(
        model::SFFM.Model,
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
            s = (X - SFM0.X) / model.C[SFM0.φ] # can't exit with c = 0
            t = SFM0.t + s
            SFM = (t = t, φ = SFM0.φ, X = X, n = SFM0.n)
        end
        return (Ind = Ind, SFM = SFM)
    end
    # SFFM Method
    function FirstExitXFun(
        model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )
        Ind = ((SFFM.X > v) || (SFFM.X < u))
        if Ind
            if (SFFM.X > v)
                X = v
            else
                X = u
            end
            s = (X - SFFM0.X) / model.C[SFFM0.φ] # can't exit with c = 0.
            t = SFFM0.t + s
            Y = UpdateYt(model, SFFM0, s)
            SFFM = (t, SFFM0.φ, X, Y, SFFM0.n)
        end
        return (Ind = Ind, SFFM = SFFM)
    end
    return FirstExitXFun
end

"""
Constructs the `StoppingTime` which is the first exit of the process ``Y(t)``
from the interval ``[u,v]``. ASSUMES ``Y(t)`` is monotonic between jumps.

    FirstExitY( u::Real, v::Real)

# Arguments
- `u`: a lower boundary
- `v`: an upper boundary

# Output
- `FirstExitYFun`: a function with one method
    - `FirstExitYFun(
        model::SFFM.Model,
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )`: a stopping time for a SFFM
"""
function FirstExitY( u::Real, v::Real)
    # SFFM Method
    function FirstExitYFun(
        model::SFFM.Model,
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )
        Ind = ((SFFM.Y < u) || (SFFM.Y > v))
        if Ind
            idx = [SFFM.Y < u; SFFM.Y > v]
            boundaryHit = [u;v][idx][1]
            YFun(t) = UpdateYt(model, SFFM0, t) - boundaryHit
            S = SFFM.t - SFFM0.t
            tstar = fzero(YFun, 0, S)
            X = UpdateXt(model, SFFM0, tstar)
            t = SFFM0.t + tstar
            Y = boundaryHit
            SFFM = (t, SFFM0.φ, X, Y, SFFM0.n)
        end
        return (Ind = Ind, SFFM = SFFM)
    end
    return FirstExitYFun
end

"""
Finds zero of `f` using the bisection method on the interval `[a,b]` with
error `err`.

    fzero( f::Function, a::Real, b::Real; err::Float64 = 1e-8)

"""
function fzero( f::Function, a::Real, b::Real; err::Float64 = 1e-6)
    # finds zeros of f using the bisection method
    c = a + (b - a) / 2
    while a < c < b
        fc = f(c)
        if abs(fc) < err
            break
        end
        p = f(a) * fc 
        if p < 0
            a, b = a, c
        elseif p > 0
            a, b = c, b
        elseif fc == 0
            break
        else 
            a = a+sqrt(eps())
        end
        c = a + (b - a) / 2
    end
    return c
end

"""
Convert from simulations of a SFM or SFFM to a distribution.

    Sims2Dist(
        model::SFFM.Model,
        mesh::SFFM.Mesh,
        sims::NamedTuple;
        type::String = "density",
    )

# Arguments
- `Model`: a Model object
- `mesh`: a Mesh object as output from `MakeMesh`
- `sims::Array`: a named tuple as output of `SFFMSim` or `SFMSim`
- `type::String`: an (optional) declaration of what type of distribution you
    want to convert to. Options are `"probability"` to return the probabilities
    ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``is the kth cell, `"cumulative"` to
    return the CDF evaluated at cell edges, or `"density"` to return an
    approximation to the density ar at the CellNodes(mesh).

# Output
- a tuple with keys
(pm=pm, distribution=yvals, x=xvals, type=type)
    - `pm::Array{Float64}`: a vector containing the point masses, the first
        `sum(model.C.<=0)` entries are the left hand point masses and the last
        `sum(model.C.>=0)` are the right-hand point masses.
    - `distribution::Array{Float64,3}`:
        - if `type="cumulative"` returns a `2×NIntervals×NPhases` array
            containing the CDF evaluated at the cell edges as contained in
            `x` below. i.e. `distribution[1,:,i]` returns the cdf at the
            left-hand edges of the cells in phase `i` and `distribution[2,:,i]`
            at the right hand edges.
        - if `type="probability"` returns a `1×NIntervals×NPhases` array
            containing the probabilities ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``
            is the kth cell.
        - if `type="density"` returns a `NBases×NIntervals×NPhases` array
            containing the kde estimate of the density function evaluated at the
            cell nodes as contained in `x` below.
    - `x::Array{Float64,2}`:
        - if `type="cumulative"` returns a `2×NIntervals×NPhases` array
            containing the cell edges as contained. i.e. `x[1,:]`
            returns the left-hand edges of the cells and `x[2,:]` at the
            right-hand edges.
        - if `type="probability"` returns a `1×NIntervals×NPhases` array
            containing the cell centers.
        - if `type="density"` returns a `NBases×NIntervals×NPhases` array
            containing the cell nodes.
    - `type`: as input in arguments.
"""
function Sims2Dist(
    model::SFFM.Model,
    mesh::SFFM.Mesh,
    sims::NamedTuple,
    type::Type{T} = SFFMProbability,
) where {T<:SFFMDistribution} 

    if type == SFFMDensity
        distribution = zeros(Float64, NBases(mesh), NIntervals(mesh), NPhases(model))
    elseif type == SFFMProbability
        distribution = zeros(Float64, 1, NIntervals(mesh), NPhases(model))
    elseif type == SFFMCDF
        distribution = zeros(Float64, 2, NIntervals(mesh), NPhases(model))
    end
    pm = zeros(Float64, sum(model.C .<= 0) + sum(model.C .>= 0))
    pc = 0
    qc = 0
    xvals = CellNodes(mesh)
    for i = 1:NPhases(model)
        # find the simluated value of imterest for this iteration
        whichsims =
            (sims.φ .== i) .&
            (sims.X .!= model.Bounds[1, 1]) .&
            (sims.X .!= model.Bounds[1, end])
        data = sims.X[whichsims]
        totalprob = sum(whichsims) / length(sims.φ)
        if type == SFFMProbability
            if length(data)!=0
                h = StatsBase.fit(StatsBase.Histogram, data, mesh.Nodes)
                h = h.weights ./ sum(h.weights) * totalprob
                distribution[:, :, i] = h
            end
            if NBases(mesh) == 1
                xvals = CellNodes(mesh)
            else
                xvals = CellNodes(mesh)[1, :] + Δ(mesh) / 2
            end
        elseif type == SFFMDensity
            if length(data)!=0
                if model.Bounds[1, end] == Inf
                    U = KernelDensity.kde(data)
                else
                    U = KernelDensity.kde(
                        data,
                        boundary = (model.Bounds[1, 1], model.Bounds[1, end]),
                    )
                end
                distribution[:, :, i] =
                    reshape(
                        KernelDensity.pdf(U, CellNodes(mesh)[:])',
                        NBases(mesh),
                        NIntervals(mesh),
                    ) * totalprob
            end
        elseif type == SFFMCDF
            if length(data)!=0
                h = StatsBase.fit(StatsBase.Histogram, data, mesh.Nodes)
                tempDist = h.weights ./ sum(h.weights) * totalprob
                tempDist = cumsum(tempDist)
                distribution[1, 2:end, i] = tempDist[1:end-1]
                distribution[2, :, i] = tempDist
                if NBases(mesh) == 1
                    xvals = [mesh.Nodes[1:end-1]';mesh.Nodes[2:end]']
                else
                    xvals = CellNodes(mesh)[[1;end], :]
                end
            end
        end

        if model.C[i] <= 0
            pc = pc + 1
            whichsims = (sims.φ .== i) .& (sims.X .== model.Bounds[1, 1])
            p = sum(whichsims) / length(sims.φ)
            pm[pc] = p
            if type == SFFMCDF
                distribution[:,:,i] = distribution[:,:,i] .+ pm[pc]
            end
        end
        if model.C[i] >= 0
            qc = qc + 1
            whichsims = (sims.φ .== i) .& (sims.X .== model.Bounds[1, end])
            p = sum(whichsims) / length(sims.φ)
            if type == SFFMCDF
                pm[sum(model.C .<= 0)+qc] = p + distribution[end,end,i]
            else
                pm[sum(model.C .<= 0)+qc] = p
            end
        end
    end
    if type == SFFMDensity && NBases(mesh) == 1
        distribution = [1; 1] .* distribution
        xvals = [mesh.Nodes[1:end-1]';mesh.Nodes[2:end]']
    end
    return type(pm, distribution, xvals)
end
