"""
    Model 

Abstract type represnting a FluidQueue or DiscretisedFluidQueue
"""
abstract type Model end

"""
    Phase

Represents a phase of the fluid queue.

# Arguments:
- `c::Float64`: the rate of change of the fluid X(t) associated with the phase
- `m::Int64`: either -1 or 1, membership of the phase with either the set of positive or negative phases.
    Default is positive, i.e. 1. Only relevant for FRAPApproximation.
- `lpm::Bool`: whether the phase has a point mass at the left/lower boundary
- `rpm::Bool`: whether the phase has a point mass at the right/upper boundary
"""
struct Phase
    c::Float64
    m::Int64 # membership to +1 or -1
    lpm::Bool # left point mass
    rpm::Bool # right point mass
    function Phase(c::Float64,m::Int,lpm::Bool,rpm::Bool)
        !((m===1)||(m===-1))&&throw(DomainError("m is 1, -1 only"))
        !((sign(c)==m)||(sign(c)==0.0))&&throw(DomainError("sign(c) must be m or 0"))
        !((c<0.0)==lpm||(c==0.0))&&throw(DomainError("negative phases must have lpm=true"))
        !((c>0.0)==rpm||(c==0.0))&&throw(DomainError("positive phases must have rpm=true"))
        return new(c,m,lpm,rpm)
    end
end
Phase(c::Float64) = Phase(c,-1+2*Int(_strictly_pos(c)),c<=0,c>=0)
# Phase(c::Int) = Phase(Float64(c),-1+2*Int(_strictly_pos(c)),c<=0,c>=0)
Phase(c::Float64,m::Int) = Phase(c,m,c<=0,c>=0)

"""
    const PhaseSet = Array{Phase,1}

A container for phases.
"""
const PhaseSet = Array{Phase,1}
PhaseSet(c::Array{Float64,1}) = [Phase(c[i]) for i in 1:length(c)]
PhaseSet(c::Array{Float64,1},m::Array{Int,1}) = (length(m)==length(c))&&[Phase(c[i],m[i]) for i in 1:length(c)]
"""
    PhaseSet(c::Array{Float64, 1}[, m::Array{Int, 1}, lpm::BitArray, rpm::BitArray]) 

Construct an Array of phases.

# Arguments:
- `c`: Array of rates
- `m`: (optional) array of memberships, see Phase()
- `lpm`: (optional) array of left point mass indicators, see Phase()
- `rpm`: (optional) array of right point mass indicators, see Phase()
"""
PhaseSet(c::Array{Float64,1},m::Array{Int,1},lpm::BitArray,rpm::BitArray) = 
    (length(m)==length(c)==length(lpm)==length(rpm))&&[Phase(c[i],m[i],lpm[i],rpm[i]) for i in 1:length(c)]

"""
    n_phases(S::PhaseSet)   

Return the number of phases from a PhaseSet, FluidQueue, or DiscretisedFluidQueue
"""
n_phases(S::PhaseSet) = length(S)
"""
    rates(S::PhaseSet, i::Int)

Return the rate of the phase from a PhaseSet, FluidQueue, or DiscretisedFluidQueue
"""
rates(S::PhaseSet,i::Int) = S[i].c
rates(S::PhaseSet) = [S[i].c for i in 1:n_phases(S)]
"""
    membership(S::PhaseSet, i::Int)

Return the membership of phases from a PhaseSet
"""
membership(S::PhaseSet,i::Int) = S[i].m
membership(S::PhaseSet) = [S[i].m for i in 1:n_phases(S)]
"""
    phases(S::PhaseSet)

Return the iterator 1:n_phases(S), for a PhaseSet, FluidQueue, or DiscretisedFluidQueue
"""
phases(S::PhaseSet) = 1:n_phases(S)

_strictly_neg(x::Float64) = (x<0.0) || (x.===-0.0)
_strictly_pos(x::Float64) = !_strictly_neg(x)
_strictly_neg(x::Int) = (x===0) ? true : throw(DomainError("invalid membership detected"))
_strictly_pos(x::Int) = (x===0) ? true : throw(DomainError("invalid membership detected"))

_has_left_boundary(i::Phase) = i.lpm
_has_right_boundary(i::Phase) = i.rpm
_has_left_boundary(S::PhaseSet,i::Int) = S[i].lpm
_has_right_boundary(S::PhaseSet,i::Int) = S[i].rpm
_has_left_boundary(S::PhaseSet) = _has_left_boundary.(S)
_has_right_boundary(S::PhaseSet) = _has_right_boundary.(S)

"""
    N₋(S::PhaseSet)

Return the number of left/lower point masses of a PhaseSet, FluidQueue or DiscretisedFluidQueue
"""
N₋(S::PhaseSet) = sum(_has_left_boundary(S))
"""
    N₊(S::PhaseSet) = begin

Return the number of right/upper point masses of a PhaseSet, FluidQueue or DiscretisedFluidQueue
"""
N₊(S::PhaseSet) = sum(_has_right_boundary(S))

checksquare(A::AbstractArray{<:Any,2}) = !(size(A,1)==size(A,2)) ? throw(DomainError(A," must be square")) : nothing

"""
    FluidQueue <: Model

Constructor for a fluid queue model.

# Arguments:
- `T::Array{<:Real, 2}`: Generator of the phase process
- `S::PhaseSet`: An array of phases describing the evolution of the fluid level in each phase.
"""
struct FluidQueue <: Model
    T::Array{<:Real,2}
    S::PhaseSet
    function FluidQueue(T::Array{<:Real,2}, S::PhaseSet)
        checksquare(T)
        !all(isapprox.(sum(T,dims=2),0, atol=1e-5))&&throw(DomainError(T, "row sums must be 0 (tol=1e-5)"))
        !all(sum(T,dims=2).==0)&&@warn "row sums of T must be 0 (tol=1e-5)"
        !(size(T,1)==length(S))&&throw(DomainError("PhaseSet must have length the dimension of T"))
        return new(T,S)
    end
end 
"""
    FluidQueue(T::Array{<:Real,2},c::Array{Float64,1})

Alias to `FluidQueue(T,PhaseSet(c))`.
"""
FluidQueue(T::Array{<:Real,2},c::Array{Float64,1}) = FluidQueue(T,PhaseSet(c))

rates(m::FluidQueue) = rates(m.S)
rates(m::FluidQueue,i::Int) = rates(m.S,i)
n_phases(m::FluidQueue) = n_phases(m.S)
phases(m::FluidQueue) = 1:n_phases(m.S)
N₋(m::FluidQueue) = N₋(m.S)
N₊(m::FluidQueue) = N₊(m.S)

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
    
    m_aug = Int.(sign.(C_aug))
    m_aug[plus_idx] .= +1
    m_aug[neg_idx] .= -1
    lpm_aug = (C_aug.<0.0) .| neg_idx
    rpm_aug = (C_aug.>0.0) .| plus_idx

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
    return T_aug, C_aug, m_aug, lpm_aug, rpm_aug
end

"""
    augment_model(model::FluidQueue)

Given a FluidQueue, return a FluidQueue with twice as many phases with rate 0, one set associated 
with m=1 phases and one associated with m=-1 phases. 
"""
function augment_model(model::FluidQueue)
    if (any(rates(model).==0))
        T_aug, C_aug, m_aug, lpm_aug, rpm_aug = _duplicate_zero_states(model.T,rates(model))
        S_aug = PhaseSet(C_aug,m_aug,lpm_aug,rpm_aug)
        return FluidQueue(T_aug,S_aug)
    else # no zero states, no augmentation needed
       return model 
    end
end

