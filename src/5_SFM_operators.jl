# Functions and operators on models
"""
    _model_dicts(model::Model) 

    input: a Model object
outputs:
     - SDict: a dictionary with keys `"+","-","0","bullet"`
    and corresponding values `findall(model.C .> 0)`, `findall(model.C .< 0)`,
    `findall(model.C .== 0)`, `findall(model.C .!= 0)`, respectively.
     - TDict: a dictionary of submatrices of `T` with keys
    `"ℓm"` with ``ℓ,m∈{+,-,0,bullet}`` and corresponding values
    `model.T[S[ℓ],S[m]]`.
"""
function _model_dicts(model::Model) 
    nPhases = n_phases(model)
    SDict = Dict{String,Array}("S" => 1:nPhases)
    SDict["+"] = findall(rates(model) .> 0)
    SDict["-"] = findall(rates(model) .< 0)
    SDict["0"] = findall(rates(model) .== 0)
    SDict["bullet"] = findall(rates(model) .!= 0)

    TDict = Dict{String,Array}("T" => model.T)
    for ℓ in ["+" "-" "0" "bullet"], m in ["+" "-" "0" "bullet"]
        TDict[ℓ*m] = model.T[SDict[ℓ], SDict[m]]
    end

    return SDict, TDict
end

"""
Construct and evaluate ``Ψ(s)`` for a triditional SFM.

Uses newtons method to solve the Ricatti equation
``D⁺⁻(s) + Ψ(s)D⁻⁺(s)Ψ(s) + Ψ(s)D⁻⁻(s) + D⁺⁺(s)Ψ(s) = 0.``

    psi_fun_x( model::Model; s = 0, MaxIters = 1000, err = 1e-8)

# Arguments
- `model`: a Model object
- `s::Real`: a value to evaluate the LST at
- `MaxIters::Int`: the maximum number of iterations of newtons method
- `err::Float64`: an error tolerance for terminating newtons method. Terminates
    when `max(Ψ_{n} - Ψ{n-1}) .< eps`.

# Output
- `Ψ(s)::Array{Float64,2}` the matrix ``Ψ``
"""
function psi_fun_x( model::Model; s = 0, MaxIters = 1000, err = 1e-8)
    SDict, TDict = _model_dicts(model)

    T00inv = inv(TDict["00"] - s * LinearAlgebra.I)
    # construct the generator Q(s)
    Q =
        (1 ./ abs.(rates(model)[SDict["bullet"]])) .* (
            TDict["bulletbullet"] - s * LinearAlgebra.I -
            TDict["bullet0"] * T00inv * TDict["0bullet"]
        )

    model_without_zero_phases = BoundedFluidQueue(Q,model.S[SDict["bullet"]])

    ~, QDict = _model_dicts(model_without_zero_phases)

    Ψ = zeros(Float64, length(SDict["+"]), length(SDict["-"]))
    A = QDict["++"]
    B = QDict["--"]
    D = QDict["+-"]
    # use netwons method to solve the Ricatti equation
    for n in 1:MaxIters
        Ψ = LinearAlgebra.sylvester(A,B,D)
        if maximum(abs.(sum(Ψ,dims=2).-1)) < err
            break
        end
        A = QDict["++"] + Ψ * QDict["-+"]
        B = QDict["--"] + QDict["-+"] * Ψ
        D = QDict["+-"] - Ψ * QDict["-+"] * Ψ
    end

    return Ψ
end

"""
Construct the vector ``ξ`` containing the distribution of the phase at the time
when ``X(t)`` first hits `0`.

    xi_x( model::Model, Ψ::Array)

# Arguments
- `model`: a Model object
- `Ψ`: an array as output from `psi_fun_x`

# Output
- the vector `ξ`
"""
function xi_x( model::Model, Ψ::Array)
    # the system to solve is [ξ 0](-[B₋₋ B₋₀; B₀₋ B₀₀])⁻¹[B₋₊; B₀₊]Ψ = ξ
    # writing this out and using block inversion (as described on wikipedia)
    # we can solve this in the following way
    SDict, TDict = _model_dicts(model)

    T00inv = inv(TDict["00"])
    invT₋₋ =
        inv(TDict["--"] - TDict["-0"] * T00inv * TDict["0-"])
    invT₋₀ = -invT₋₋ * TDict["-0"] * T00inv

    A =
        -(
            invT₋₋ * TDict["-+"] * Ψ + invT₋₀ * TDict["0+"] * Ψ +
            LinearAlgebra.I
        )
    b = zeros(1, size(TDict["--"], 1))
    A[:, 1] .= 1.0 # normalisation conditions
    b[1] = 1.0 # normalisation conditions

    ξ = b / A

    return ξ
end

"""
Construct the stationary distribution of the SFM

    stationary_distribution_x( model::Model, Ψ::Array, ξ::Array)

# Arguments
- `model`: a Model object
- `Ψ`: an array as output from `psi_fun_x`
- `ξ`: an array as returned from `xi_x`

# Output
- `pₓ::Array{Float64,2}`: the point masses of the SFM
- `πₓ(x)` a function with two methods
    - `πₓ(x::Real)`: for scalar inputs, returns the stationary density evaluated
        at `x` in all phases.
    - `πₓ(x::Array)`: for array inputs, returns an array with the same shape
        as is output by Coeff2Dist.
- `K::Array{Float64,2}`: the matrix in the exponential of the density.
"""
function stationary_distribution_x( model::Model, Ψ::Array, ξ::Array)
    # using the same block inversion trick as in xi_x
    SDict, TDict = _model_dicts(model)
    
    T00inv = inv(TDict["00"])
    invT₋₋ =
        inv(TDict["--"] - TDict["-0"] * T00inv * TDict["0-"])
    invT₋₀ = -invT₋₋ * TDict["-0"] * T00inv

    Q =
        (1 ./ abs.(rates(model)[SDict["bullet"]])) .* (
            TDict["bulletbullet"] -
            TDict["bullet0"] * T00inv * TDict["0bullet"]
        )

    model_without_zero_phases = BoundedFluidQueue(Q,model.S[SDict["bullet"]])

    ~, QDict = _model_dicts(model_without_zero_phases)
    
    K = QDict["++"] + Ψ * QDict["-+"]

    A = -[invT₋₋ invT₋₀]

    # unnormalised values
    αpₓ = ξ * A

    απₓ = αpₓ *
        [TDict["-+"]; TDict["0+"]] *
        -inv(K) *
        [LinearAlgebra.I(length(SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(rates(model)[SDict["bullet"]]))

    απₓ0 = -απₓ * [TDict["+0"];TDict["-0"]] * T00inv

    # normalising constant
    α = sum(αpₓ) + sum(απₓ) + sum(απₓ0)

    # normalised values
    # point masses
    pₓ = αpₓ/α
    # density method for scalar x-values
    idx = [findall(rates(model).>0);
        findall(rates(model).<0);
        findall(rates(model).==0)]
    function πₓ(x::Real)
        out = zeros(n_phases(model))
        out[idx] = (pₓ *
        [TDict["-+"]; TDict["0+"]] *
        exp(K*x) *
        [LinearAlgebra.I(length(SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(rates(model)[SDict["bullet"]])) *
        [LinearAlgebra.I(sum(rates(model) .!= 0)) [TDict["+0"];TDict["-0"]] * -T00inv])
        return out 
    end
    # density method for arrays so that πₓ returns an array with the same shape
    # as is output by Coeff2Dist
    function πₓ(x::Array)
        temp = πₓ.(x)
        Evalπₓ = zeros(Float64, size(x,1), size(x,2), n_phases(model))
        for cell in 1:size(x,2)
            for basis in 1:size(x,1)
                Evalπₓ[basis,cell,:] = temp[basis,cell]
            end
        end
        return Evalπₓ
    end

    # CDF method for scalar x-values
    function Πₓ(x::Real)
        out = zeros(n_phases(model))
        out[idx] = [zeros(1,sum(rates(model).>0)) pₓ] .+
        pₓ *
        [TDict["-+"]; TDict["0+"]] *
        (exp(K*x) - LinearAlgebra.I) / K *
        [LinearAlgebra.I(length(SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(rates(model)[SDict["bullet"]])) *
        [LinearAlgebra.I(sum(rates(model) .!= 0)) [TDict["+0"];TDict["-0"]] * -T00inv]
        return out 
    end
    # CDF method for arrays so that Πₓ returns an array with the same shape
    # as is output by Coeff2Dist
    function Πₓ(x::Array)
        temp = Πₓ.(x)
        Evalπₓ = zeros(Float64, size(x,1), size(x,2), n_phases(model))
        for cell in 1:size(x,2)
            for basis in 1:size(x,1)
                Evalπₓ[basis,cell,:] = temp[basis,cell]
            end
        end
        return Evalπₓ
    end

    return pₓ, πₓ, Πₓ, K
end

function stationary_distribution_x( model::Model)
    Ψ = psi_fun_x( model)
    ξ = xi_x( model, Ψ)
    pₓ, πₓ, Πₓ, K = stationary_distribution_x(model,Ψ,ξ)
    return pₓ, πₓ, Πₓ, K
end

function expected_crossings(moodel::BoundedFluidQueue)
    rate_reverse_model = BoundedFluidQueue(model.T,-model.C,model.P_lwr,model.P_upr)
    ~, TDict = _model_dicts(model)
    ~, T̃Dict = _model_dicts(rate_reverse_model)
    Ψ = psi_fun_x(model)
    Ψ̃ = psi_fun_x(rate_reverse_model)
    K = TDict["++"] + Ψ*TDict["-+"]
    K̃ = T̃Dict["++"] + Ψ̃*T̃Dict["-+"]
    n₊ = size(TDict["++"],1)
    n₋ = size(TDict["--"],1)
    I₊ = LinearAlgebra.I(n₊)
    I₋ = LinearAlgebra.I(n₋)
    N(x) = [I₊ exp(K*model.b); exp(K̃*model.b) I₋]\[exp(K*x) zeros(n₊,n₋); zeros(n₋,n₊) exp(K̃*(model.b-x))]*
        [I₊ Ψ; Ψ̃ I₋]
    return N(x)
end

function hat_matrices(model::BoundedFluidQueue)
    rate_reverse_model = BoundedFluidQueue(model.T,-model.C,model.P_lwr,model.P_upr)
    ~, TDict = _model_dicts(model)
    # ~, T̃Dict = _model_dicts(rate_reverse_model)
    Ψ = psi_fun_x(model)
    Ψ̃ = psi_fun_x(rate_reverse_model)
    U = TDict["--"] + TDict["-+"]*Ψ
    Ũ = TDict["++"] + TDict["+-"]*Ψ̃
    n₊ = size(TDict["++"],1)
    n₋ = size(TDict["--"],1)
    I₊ = LinearAlgebra.I(n₊)
    I₋ = LinearAlgebra.I(n₋)
    b = model.b
    Λ̂₁₁ = (I₊ - Ψ*Ψ̃)*exp(Ũ*b)/(I₊ - Ψ*exp(U*b)*Ψ̃*exp(Ũ*b))
    Ψ̂₁₂ = (Ψ - exp(Ũ*b)*Ψ*exp(U*b))/(I₋ - Ψ̃*exp(Ũ*b)*Ψ*exp(U*b))
    Ψ̂₂₁ = (Ψ̃ - exp(U*b)*Ψ̃*exp(Ũ*b))/(I₊ - Ψ*exp(U*b)*Ψ̃*exp(Ũ*b))
    Λ̂₂₂ = (I₋ - Ψ̃*Ψ)*exp(U*b)/(I₋ - Ψ̃*exp(Ũ*b)*Ψ*exp(U*b))
    return Λ̂₁₁, Ψ̂₁₂, Ψ̂₂₁, Λ̂₂₂
end

function H_matrix(model)
    Λ̂₁₁, Ψ̂₁₂, Ψ̂₂₁, Λ̂₂₂ = hat_matrices(model)
    hat_matrix = [Λ̂₁₁, Ψ̂₁₂; Ψ̂₂₁, Λ̂₂₂]
    S, TDict = _model_dicts(model)
    n₊ = size(TDict["++"],1)
    n₋ = size(TDict["--"],1)
    n = length(model.C)
    Ĥ = hat_matrix*[zeros(n₊,n₊) model.P_upr[:,S["-"]]; zeros(n₋,n₋) model.P_lwr[:,S["+"]]] + 
        hat_matrix*[model.P_upr[:,[i∉S["-"] for i in 1:n]] zeros(n₊,n-n₊); model.P_lwr[:,[i∉S["+"] for i in 1:n]] zeros(n₋,n-n₋)]*
            [-inv(model.T[[i∉S["-"] for i in 1:n],[i∉S["-"] for i in 1:n]]) zeros(n-n₋,n-n₊); 
            zeros(n-n₊,n-n₋) -inv(model.T[[i∉S["+"] for i in 1:n],[i∉S["+"] for i in 1:n]])]*
                [zeros(n-n₋,n₊) model.T[[i∉S["-"] for i in 1:n],S["-"]];
                model.T[[i∉S["+"] for i in 1:n],S["+"]] zeros(n-n₊,n₋)]
    return Ĥ
end

function nu(model)
    A = H_matrix(model) - LinearAlgebra.I
    A[:,1] .= 1.0
    b = zeros(1,size(A,1))
    b[1] = 1.0
    return b/A
end

function normalising_constant_matrix(model)
    rate_reverse_model = BoundedFluidQueue(model.T,-model.C,model.P_lwr,model.P_upr)
    ~, TDict = _model_dicts(model)
    ~, T̃Dict = _model_dicts(rate_reverse_model)
    Ψ = psi_fun_x(model)
    Ψ̃ = psi_fun_x(rate_reverse_model)
    K = TDict["++"] + Ψ*TDict["-+"]
    K̃ = T̃Dict["++"] + Ψ̃*T̃Dict["-+"]
    n₊ = size(TDict["++"],1)
    n₋ = size(TDict["--"],1)
    I₊ = LinearAlgebra.I(n₊)
    I₋ = LinearAlgebra.I(n₋)
    N = [I₊ exp(K*model.b); exp(K̃*model.b) I₋]\[K\(exp(K*b)-I₊) zeros(n₊,n₋); zeros(n₋,n₊) K̃\(exp(K̃*b)-I₋)]*
        [I₊ Ψ; Ψ̃ I₋]
    return N
end

function stationary_distribution_x(model::BoundedFluidQueue)
    ν = nu(model)
    S, T = _model_dicts(model)
    # n₊ = length(S["+"])
    # n₋ = length(S["-"])
    # n = length(model.C)
    N = expected_crossings(model)
    N_const = normalising_constant_matrix(model)
    # ν₊ = ν[1:n₊]
    # ν₋ = ν[(n₊+1):end]
    Cinv= diagm(1.0./[abs.(model.C[S["+"]]);abs.(model.C[S["-"]])])
    norm_vec = ν*N_const*Cinv
    z = sum(norm_vec) + sum(norm_vec*[T["+0"];T["-0"]]*(-inv(T["00"])))
    idx = [findall(rates(model).>0);
        findall(rates(model).<0);
        findall(rates(model).==0)]
    function πₓ(x)
        vec = ν*N(x)*Cinv
        zϕ̂ = vec*[T["+0"];T["-0"]]*(-inv(T["00"]))
        out = [vec zϕ̂]./z
        return out[idx]
    end
    return πₓ
end

