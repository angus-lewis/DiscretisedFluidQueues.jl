"""
    SFMDistribution{T <: Mesh} 

Representation of the distribution of a DiscretisedFluidQueue. 

The field `coeffs` is a row-vector and indexing a distribution indexes this field. Similarly, 
arithmetic operations on a distribution act on this vector.

# Arguments:
- `coeffs::Array{Float64, 2}`: A row vector of coefficients which encode the distribution of a 
    DiscretisedFluidQueue and can be used to reconstruct a solution.
- `dq::DiscretisedFluidQueue{T}`: 
"""
struct SFMDistribution{T<:Mesh} <: AbstractMatrix{Float64}
    coeffs::Array{Float64,2}
    dq::DiscretisedFluidQueue{T}
    SFMDistribution{T}(coeffs::Array{Float64,2},dq::DiscretisedFluidQueue{T}) where T<:Mesh = 
        (size(coeffs,1)==1) ? new(coeffs,dq) : throw(DimensionMismatch("coeffs must be a row-vector"))
end

SFMDistribution(coeffs::Array{Float64,2},dq::DiscretisedFluidQueue{T}) where T = 
    SFMDistribution{T}(coeffs,dq)

"""
    SFMDistribution(dq::DiscretisedFluidQueue{T})

A blank initialiser for a fluid queue distribution a distribution with `coeffs=zeros`.
"""
SFMDistribution(dq::DiscretisedFluidQueue{T}) where T = 
    SFMDistribution{T}(zeros(1,n_bases_per_phase(dq)*n_phases(dq)+N₊(dq))+N₋(dq))

size(d::SFMDistribution) = size(d.coeffs)
size(d::SFMDistribution,dim::Int) = size(d.coeffs,dim)
getindex(d::SFMDistribution,i,j) = d.coeffs[i,j]
getindex(d::SFMDistribution,i) = d.coeffs[i]
# setindex!(d::SFMDistribution,x,i,j) = throw(DomainError("inserted value(s) must be Float64"))
setindex!(d::SFMDistribution,x,i,j) = (d.coeffs[i,j]=x)
# setindex!(d::SFMDistribution,x,i) = throw(DomainError("inserted value(s) must be Float64"))
setindex!(d::SFMDistribution,x,i) = (d.coeffs[i]=x)
# length(d::SFMDistribution) = prod(size(d.coeffs))
# itertate(d::SFMDistribution, i=1, args...;kwargs...) = iterate(d.coeffs, i, args...; kwargs...)
# lastindex(d::SFMDistribution) = d.coeffs[length(d)]
# Base.BroadcastStyle(::Type{<:SFMDistribution}) = Broadcast.ArrayStyle{SFMDistribution}()

show(io::IO, mime::MIME"text/plain", d::SFMDistribution) = show(io, mime, d.coeffs)

fast_mul(u::SFMDistribution,B::AbstractMatrix{Float64}) = SFMDistribution(u.coeffs*B,u.dq)
fast_mul(B::AbstractMatrix{Float64},u::SFMDistribution) = 
    throw(DomainError("you can only premultiply a SFMDistribution"))
fast_mul(u::SFMDistribution,x::Real) = SFMDistribution(u.coeffs*x,u.dq)
fast_mul(x::Real,u::SFMDistribution) = fast_mul(u,x)

include("11a_approximation.jl")
include("11b_reconstruction.jl")
