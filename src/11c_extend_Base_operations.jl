# # Extend Base operations for my types or existing concrete types only
# # else, fallback to methods AbstractMatrix{Float64}

# array_types = (:(SparseArrays.SparseMatrixCSC{Float64,Int}),:(Matrix{Float64}))
# my_types = (:LazyGenerator, :FullGenerator, :SFMDistribution)
# for mt in my_types
#     for at in (array_types..., :Float64, :Int)
#         @eval *(B::$mt,A::$at) = fast_mul(B,A)
#         @eval *(A::$at,B::$mt) = fast_mul(A,B)
#     end
# end

# for mt in (:LazyGenerator, :FullGenerator)
#     @eval *(A::$mt,B::$mt) = fast_mul(A,B)
# end

# +(f::SFMDistribution,g::SFMDistribution) = 
#     throw(DomainError("cannot add SFMDistributions with different DiscretisedFluidQueues"))
# function +(f::SFMDistribution{T},g::SFMDistribution{T}) where T<:Mesh
#     !(f.dq==g.dq)&&throw(DomainError("SFMDistributions need the same model & mesh"))
#     return SFMDistribution{T}(f.coeffs+g.coeffs,f.dq)
# end