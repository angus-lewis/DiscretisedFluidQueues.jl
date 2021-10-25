"""
    FullGenerator{T} <: Generator{T}

An Matrix or SparseMatrixCSC representation of the generator of a generator of a DiscretisedFluidQueue.

Higher memory requirements than LazyGenerator (as blocks are duplicated) but much faster matrix arithmetic.
"""
struct FullGenerator{T} <: Generator{T}
    B::Union{Array{Float64,2}, SparseArrays.SparseMatrixCSC{Float64,Int}}
    dq::DiscretisedFluidQueue{T}
end
FullGenerator(B,dq) = FullGenerator{typeof(dq.mesh)}(B,dq)

size(B::FullGenerator) = size(B.B)
getindex(B::FullGenerator,i) = B.B[i]
getindex(B::FullGenerator,i,j) = B.B[i,j]
setindex!(B::FullGenerator,x::Float64,i,j) = (B.B[i,j]=x)
sum(B::FullGenerator; kwargs...) = sum(B.B; kwargs...)

fast_mul(A::AbstractArray{<:Real,2}, B::FullGenerator) = A*B.B
fast_mul(B::FullGenerator, A::AbstractArray{<:Real,2}) = fast_mul(A,B)
fast_mul(A::FullGenerator, B::FullGenerator) = A.B*B.B
fast_mul(A::FullGenerator, x::Real) = A.B*x
fast_mul(x::Real,A::FullGenerator) = fast_mul(A, x)


function show(io::IO, mime::MIME"text/plain", B::FullGenerator)
    if VERSION >= v"1.6"
        show(io, mime, B.B)
    else
        show(io, mime, Matrix(B.B))
    end
end

"""
    build_full_generator(dq::DiscretisedFluidQueue; v::Bool = false)

Returns a SparseMatrixCSC generator of a DiscretisedFluidQueue.
"""
function build_full_generator(dq::DiscretisedFluidQueue; v::Bool=false) 
    lazy = build_lazy_generator(dq; v=v)
    return build_full_generator(lazy)
end

"""
    build_full_generator(lzB::LazyGenerator)

Returns a SparseMatrixCSC representation of a LazyGenerator.
"""
function build_full_generator(lzB::LazyGenerator)
    B = fast_mul(SparseArrays.SparseMatrixCSC{Float64,Int}(LinearAlgebra.I(size(lzB,1))),lzB)
    return FullGenerator(B,lzB.dq)
end

