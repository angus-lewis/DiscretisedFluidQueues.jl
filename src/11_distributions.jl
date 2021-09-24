# import Base: getindex, size, *

struct SFMDistribution{T<:Mesh} <: AbstractArray{Float64,2} 
    coeffs::Array{Float64,2}
    dq::DiscretisedFluidQueue{T}
    SFMDistribution{T}(coeffs::Array{Float64,2},dq::DiscretisedFluidQueue{T}) where T<:Mesh = 
        (size(coeffs,1)==1) ? new(coeffs,dq) : throw(DimensionMismatch("coeffs must be a row-vector"))
end

SFMDistribution(coeffs::Array{Float64,2},dq::DiscretisedFluidQueue{T}) where T = 
    SFMDistribution{T}(coeffs,dq)

SFMDistribution(dq::DiscretisedFluidQueue{T}) where T = 
    SFMDistribution{T}(zeros(1,total_n_bases(dq.mesh)*n_phases(dq.mesh)+N₊(dq.model.S))+N₋(dq.model.S))

+(f::SFMDistribution,g::SFMDistribution) = throw(DomainError("cannot add SFMDistributions with differen mesh types"))
function +(f::SFMDistribution{T},g::SFMDistribution{T}) where T<:Mesh
    !((f.dq.model==g.model)&&(f.dq.mesh==g.dq.mesh))&&throw(DomainError("SFMDistributions need the same model & mesh"))
    return SFMDistribution{T}(f.d+g.d,f.dq)
end

size(d::SFMDistribution) = size(d.coeffs)
getindex(d::SFMDistribution,i::Int,j::Int) = d.coeffs[i,j]
setindex!(d::SFMDistribution,x,i::Int,j::Int) = throw(DomainError("inserted value(s) must be Float64"))
setindex!(d::SFMDistribution,x::Float64,i::Int,j::Int) = (d.coeffs[i,j]=x)
*(u::SFMDistribution,B::AbstractArray{Float64,2}) = SFMDistribution(*(u.coeffs,B),u.dq)
*(B::AbstractArray{Float64,2},u::SFMDistribution) = *(u,B)
*(u::SFMDistribution,B::Number) = SFMDistribution(*(u.coeffs,B),u.dq)
*(B::Number,u::SFMDistribution) = *(B,u)

include("11a_approximation.jl")
include("11b_reconstruction.jl")

