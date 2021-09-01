abstract type Model end

"""
Construct a SFFM model object.

    FluidFluidQueue(
        T::Array{Float64,2},
        C::Array{Float64,1},
        r::NamedTuple{(:r, :R)};
        Bounds::Array{<:Real,2} = [-Inf Inf; -Inf Inf],
    )

# Arguments
- `T::Array{Float64,2}`: generator matrix for the CTMC ``φ(t)``
- `C::Array{Float64,1}`: vector of rates ``d/dt X(t)=C[i]`` for ``i=φ(t)``.
- `r::NamedTuple{(:r, :R)}`: rates for the second fluid.
    - `:r(x::Array{Real})`, a function  which takes arrays of x-values and
        returns a row vector of values for each x-value. i.e. `:r([0;1])`
        returns a `2×NPhases` array where the first row contains all the
        ``rᵢ(0)`` and row 2 all the ``rᵢ(1)`` values.
    - `:R(x::Array{Real})`: has the same structure/behaviour as ``:r`` but
        returns the integral of ``:r``. i.e. `Rᵢ(x)=∫ˣrᵢ(y)dy`.
- `Bounds::Array{<:Real,2}`: contains the bounds for the model. The first row
    are the L and R bounds for ``X(t)`` and the second row the bounds for
    ``Y(t)`` (although the bounds for ``Y(t)`` don't actually do anything yet).

# Outputs
- a model object which is a tuple with fields
    - `:T`: as input
    - `:C`: as input
    - `:r`: a named tuple with fields `(:r, :R, :a)`, `:r` and `:R` are as input
        and `:a = abs.(:r)` returns the absolute values of the rates.
    - `Bounds`: as input
)
"""
struct Phase
    c::Float64
    m::Int # membership to + or -
    function Phase(c::Float64,m::Int)
        !((m==1)||(m==-1))&&throw(DomainError("m is +1 or -1 only"))
        !((sign(c)==m)||(sign(c)==0))&&throw(DomainError("sign(c) must be 0 or m"))
        return new(c,m)
    end
end
const PhaseSet = Array{Phase,1}
PhaseSet(c::Array{Float64,1},m::Array{Int,1}) = 
    (length(m)==length(c))&&[Phase(c[i],m[i]) for i in 1:length(c)]
# getindex(ph::PhaseSet,i::Int) = ph.S[i]
n_phases(S::PhaseSet) = length(S)
get_rates(S::PhaseSet,i::Int) = S[i].c
get_rates(S::PhaseSet) = [S[i].c for i in 1:n_phases(S)]
get_membership(S::PhaseSet,i::Int) = S[i].m
get_membership(S::PhaseSet) = [S[i].m for i in 1:n_phases(S)]

phases(S::PhaseSet) = 1:n_phases(S)
N₋(C::Array{<:Real,1}) = sum(C.<=0)
N₊(C::Array{<:Real,1}) = sum(C.>=0)
N₋(S::PhaseSet) = sum(get_rates(S).<=0)
N₊(S::PhaseSet) = sum(get_rates(S).>=0)

checksquare(A::AbstractArray{<:Any,2}) = !(size(A,1)==size(A,2)) ? throw(DomainError(A," must be square")) : nothing

struct FluidQueue <: Model
    T::Array{<:Real,2}
    S::PhaseSet
    bounds::Array{<:Real,1}
    function FluidQueue(T::Array{<:Real,2}, S::PhaseSet, bounds::Array{<:Real,1})
        checksquare(T)
        !all(isapprox.(sum(T,dims=2),0, atol=1e-5))&&throw(DomainError(T, "row sums must be 0 (tol=1e-5)"))
        !all(sum(T,dims=2).==0)&&@warn "row sums of T must be 0 (tol=1e-5)"
        !(size(T,1)==length(S))&&throw(DomainError("PhaseSet must have length the dimension of T"))
        !((length(bounds)==2)&&(bounds[1]<bounds[2]))&&throw(DomainError("bound must have 2 entries, upper and lower"))
        return new(T,S,bounds)
    end
end 
get_rates(m::FluidQueue) = get_rates(m.S)
get_rates(m::FluidQueue,i::Int) = get_rates(m.S,i)
n_phases(m::FluidQueue) = n_phases(m.S)
phases(m::FluidQueue) = 1:n_phases(m.S)


function _duplicate_zero_states(T::Array{<:Real,2},C::Array{<:Real,1})
    n_0 = sum(C.==0)
    
    # assign rates for the augmented model
    C_aug = Array{Float64,1}(undef,length(C)+n_0)
    c_zero = 0
    plus_idx = falses(length(C_aug)) # zero states associated with +
    neg_idx = falses(length(C_aug)) # zero states associated with -
    for i in 1:length(C)
        C_aug[i+c_zero] = C[i]
        if C[i] == 0
            C_aug[i+c_zero+1] = C[i]
            plus_idx[i+c_zero] = true
            neg_idx[i+c_zero+1] = true
            c_zero += 1
        end
    end
    
    m_aug = Int.(sign.(C_aug)) .+ plus_idx .- neg_idx

    # assign augmented generator
    c_zero = 0
    T_aug = zeros(length(C_aug),length(C_aug))
    for i in 1:length(C)
        if C[i] == 0
            # duplicate 
            T_aug[i+c_zero,(C_aug.!=0).|(plus_idx)] = T[i,:]
            T_aug[i+c_zero+1,(C_aug.!=0).|(neg_idx)] = T[i,:]
            c_zero += 1
        elseif C[i] < 0
            T_aug[i+c_zero,(C_aug.!=0).|(neg_idx)] = T[i,:]
        elseif C[i] > 0 
            T_aug[i+c_zero,(C_aug.!=0).|(plus_idx)] = T[i,:]
        end
    end
    return T_aug, C_aug, m_aug
end

function augment_model(model::FluidQueue)
    if (any(get_rates(model).==0))
        T_aug, C_aug, m_aug = _duplicate_zero_states(model.T,get_rates(model))
        S_aug = PhaseSet(C_aug,m_aug)
        return FluidQueue(T_aug,S_aug,model.bounds)
    else # no zero states, no augmentation needed
       return model 
    end
end

export Model, Phase, PhaseSet, n_phases, get_rates, get_membership, phases, 
    checksquare, N₋, N₊, FluidQueue, augment_model
