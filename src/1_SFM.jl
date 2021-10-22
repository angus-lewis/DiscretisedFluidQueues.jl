"""
    Model 

Abstract type represnting a BoundedFluidQueue or DiscretisedFluidQueue
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

Return the number of phases from a PhaseSet, BoundedFluidQueue, or DiscretisedFluidQueue
"""
n_phases(S::PhaseSet) = length(S)
"""
    rates(S::PhaseSet, i::Int)

Return the rate of the phase from a PhaseSet, BoundedFluidQueue, or DiscretisedFluidQueue
"""
rates(S::PhaseSet,i::Int) = S[i].c
rates(S::PhaseSet) = [S[i].c for i in 1:n_phases(S)]

negative_phases(S::PhaseSet,i::Int) = S[i].c.<0.0
negative_phases(S::PhaseSet) = [S[i].c.<0.0 for i in 1:n_phases(S)]
positive_phases(S::PhaseSet,i::Int) = S[i].c.>0.0
positive_phases(S::PhaseSet) = [S[i].c.>0.0 for i in 1:n_phases(S)]

"""
    membership(S::PhaseSet, i::Int)

Return the membership of phases from a PhaseSet
"""
membership(S::PhaseSet,i::Int) = S[i].m
membership(S::PhaseSet) = [S[i].m for i in 1:n_phases(S)]
"""
    phases(S::PhaseSet)

Return the iterator 1:n_phases(S), for a PhaseSet, BoundedFluidQueue, or DiscretisedFluidQueue
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

Return the number of left/lower point masses of a PhaseSet, BoundedFluidQueue or DiscretisedFluidQueue
"""
N₋(S::PhaseSet) = sum(_has_left_boundary(S))
"""
    N₊(S::PhaseSet) = begin

Return the number of right/upper point masses of a PhaseSet, BoundedFluidQueue or DiscretisedFluidQueue
"""
N₊(S::PhaseSet) = sum(_has_right_boundary(S))

checksquare(A::AbstractArray{<:Any,2}) = !(size(A,1)==size(A,2)) ? throw(DomainError(A," must be square")) : nothing

function _fluid_queue_checks(T,S)
    checksquare(T)
    !(size(T,1)==length(S))&&throw(DomainError("PhaseSet must have length the size(T,1)"))
    return 
end

"""
    BoundedFluidQueue <: Model

Constructor for a fluid queue model. 

# Arguments:
- `T::Array{Float64, 2}`: Generator of the phase process
- `S::PhaseSet`: An array of phases describing the evolution of the fluid level in each phase.
- `P_lwr::Array{Float64,2}`: Probabilities of phase change upon hitting the lower boundary. 
    A |S₋| by |S| array.
- `P_upr::Array{Float64,2}`: Probabilities of phase change upon hitting the upper boundary. 
    A |S₊| by |S| array.
"""
struct BoundedFluidQueue <: Model
    T::Array{Float64,2}
    S::PhaseSet
    P_lwr::Matrix{Float64}
    P_upr::Matrix{Float64}
    function BoundedFluidQueue(T::Array{Float64,2}, S::PhaseSet, P_lwr::Matrix{Float64}, P_upr::Matrix{Float64})
        _fluid_queue_checks(T,S)
        !(all(sum(T,dims=2).≈0.0))&&@warn "row sums of T should be 0"
        !all(sum(P_lwr,dims=2).≈1.0)&&throw(DomainError("P_lwr sum should be 1"))#@warn "row sums of P_lwr should be 1.0"
        !all(sum(P_upr,dims=2).≈1.0)&&throw(DomainError("P_lwr sum should be 1"))#@warn "row sums of P_upr should be 1.0"
        !((sum(rates(S).<0.0)==size(P_lwr,1))&&(size(T,2)==size(P_lwr,2)))&&throw(DomainError("P_lwr must be |S₋| by |S|"))
        !((sum(rates(S).>0.0)==size(P_upr,1))&&(size(T,2)==size(P_upr,2)))&&throw(DomainError("P_upr must be |S₊| by |S|"))
        return new(T,S,P_lwr,P_upr)
    end
end
"""
    BoundedFluidQueue(T::Array{Float64,2}, S::PhaseSet)

Constructor for a fluid queue model with regulated boundaries by default 

# Arguments:
- `T::Array{Float64, 2}`: Generator of the phase process
- `S::PhaseSet`: An array of phases describing the evolution of the fluid level in each phase.
"""
function BoundedFluidQueue(T::Array{Float64,2}, S::PhaseSet)
    _fluid_queue_checks(T,S)
    P_lwr = zeros(sum(negative_phases(S)),size(T,2))
    P_upr = zeros(sum(positive_phases(S)),size(T,2))
    i_lwr = 0
    i_upr = 0
    C = rates(S)
    for j in 1:size(T,2)
        if C[j] < 0.0
            i_lwr += 1
            P_lwr[i_lwr,j] = 1.0
        elseif C[j] > 0.0
            i_upr += 1
            P_upr[i_upr,j] = 1.0
        end
    end
    return BoundedFluidQueue(T,S,P_lwr,P_upr)
end
"""
    BoundedFluidQueue(T::Array{Float64,2},c::Array{Float64,1})

Alias to `BoundedFluidQueue(T,PhaseSet(c))`.
"""
BoundedFluidQueue(T::Array{Float64,2},c::Array{Float64,1}) = BoundedFluidQueue(T,PhaseSet(c))

rates(m::Model) = rates(m.S)
rates(m::Model,i::Int) = rates(m.S,i)

negative_phases(m::Model,i::Int) = negative_phases(m.S,i)
negative_phases(m::Model) = negative_phases(m.S)
positive_phases(m::Model,i::Int) = positive_phases(m.S,i)
positive_phases(m::Model) = positive_phases(m.S)

n_phases(m::Model) = n_phases(m.S)
phases(m::Model) = 1:n_phases(m.S)
N₋(m::Model) = N₋(m.S)
N₊(m::Model) = N₊(m.S)

function _duplicate_zero_states_C(T::Array{Float64,2},C::Array{Float64,1})
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
    return C_aug, plus_idx, neg_idx
end
function _duplicate_zero_states_membership(C_aug,neg_idx,plus_idx)
    m_aug = Int.(sign.(C_aug))
    m_aug[plus_idx] .= +1
    m_aug[neg_idx] .= -1
    lpm_aug = (C_aug.<0.0) .| neg_idx
    rpm_aug = (C_aug.>0.0) .| plus_idx
    return m_aug, lpm_aug, rpm_aug
end
function _duplicate_zero_states_T(T,C,C_aug,neg_idx,plus_idx)
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
    return T_aug
end
function _duplicate_zero_states_P(neg_idx,plus_idx,C_aug,P_lwr,P_upr)
    P_lwr_aug = zeros(sum(C_aug.<0.0),length(neg_idx))
    P_upr_aug = zeros(sum(C_aug.>0.0),length(plus_idx))
    c_zero_lwr = 0
    c_zero_upr = 0
    for i in 1:length(neg_idx)
        if !((plus_idx[i])&&(C_aug[i]==0.0))
            c_zero_lwr += 1
            P_lwr_aug[:,i] = P_lwr[:,c_zero_lwr]
        end
        if !((neg_idx[i])&&(C_aug[i]==0.0))
            c_zero_upr += 1
            P_upr_aug[:,i] = P_upr[:,c_zero_upr]
        end
    end
    return P_lwr_aug, P_upr_aug
end

function _duplicate_zero_states(T::Array{Float64,2},C::Array{Float64,1},P_lwr::Array{Float64,2},P_upr::Array{Float64,2})
    C_aug, plus_idx, neg_idx = _duplicate_zero_states_C(T,C)
    m_aug, lpm_aug, rpm_aug = _duplicate_zero_states_membership(C_aug,neg_idx,plus_idx)
    T_aug = _duplicate_zero_states_T(T,C,C_aug,neg_idx,plus_idx)
    P_lwr_aug, P_upr_aug = _duplicate_zero_states_P(neg_idx,plus_idx,C_aug,P_lwr,P_upr)
    
    return T_aug, C_aug, m_aug, lpm_aug, rpm_aug, P_lwr_aug, P_upr_aug
end

"""
    augment_model(model::Model)

Given a BoundedFluidQueue, return a BoundedFluidQueue with twice as many phases with rate 0, one set associated 
with m=1 phases and one associated with m=-1 phases. 
"""
function augment_model(model::Model)
    if (any(rates(model).==0))
        T_aug, C_aug, m_aug, lpm_aug, rpm_aug, P_lwr_aug, P_upr_aug = 
            _duplicate_zero_states(model.T,rates(model),model.P_lwr,model.P_upr)
        S_aug = PhaseSet(C_aug,m_aug,lpm_aug,rpm_aug)
        return BoundedFluidQueue(T_aug,S_aug,P_lwr_aug,P_upr_aug)
    else # no zero states, no augmentation needed
       return model 
    end
end

