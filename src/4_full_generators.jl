"""
    FullGenerator <: Generator

An Matrix or SparseMatrixCSC representation of the generator of a generator of a DiscretisedFluidQueue.

Higher memory requirements than LazyGenerator (as blocks are duplicated) but much faster matrix arithmetic.
"""
struct FullGenerator <: Generator 
    B::Union{Array{Float64,2}, SparseArrays.SparseMatrixCSC{Float64,Int}}
end

size(B::FullGenerator) = size(B.B)
# size(B::FullGenerator,dim::Int) = size(B.B,dim)
# length(B::FullGenerator) = prod(size(B))
getindex(B::FullGenerator,i) = B.B[i]
# setindex!(B::FullGenerator,x,i,j) = 
    # throw(DomainError("number to insert must be Float64"))
setindex!(B::FullGenerator,x::Float64,i,j) = (B.B[i,j]=x)
sum(B::FullGenerator; kwargs...) = sum(B.B; kwargs...)
# itertate(B::FullGenerator, i=1, args...;kwargs...) = iterate(B.B, i, args...; kwargs...)
# Base.BroadcastStyle(::Type{<:FullGenerator}) = Broadcast.ArrayStyle{FullGenerator}()


# for f in (:+,:-,:*)
#     @eval $f(A::AbstractArray{<:Real,2}, B::FullGenerator) = $f(A,B.B)
#     @eval $f(B::FullGenerator, A::AbstractArray{<:Real,2}) = $f(B.B,A)
#     @eval $f(A::FullGenerator, B::FullGenerator) = $f(A.B,B.B)
# end
# +(A::AbstractArray{<:Real,2}, B::FullGenerator) = A+B.B
# +(B::FullGenerator, A::AbstractArray{<:Real,2}) = B.B+A
# +(A::FullGenerator, B::FullGenerator) = A.B+B.B
# -(A::AbstractArray{<:Real,2}, B::FullGenerator) = A-B.B
# -(B::FullGenerator, A::AbstractArray{<:Real,2}) = B.B-A
# -(A::FullGenerator, B::FullGenerator) = A.B-B.B
# *(A::AbstractArray{<:Real,2}, B::FullGenerator) = A*B.B
# *(B::FullGenerator, A::AbstractArray{<:Real,2}) = B.B*A
# *(A::FullGenerator, B::FullGenerator) = A.B*B.B


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
    B = SparseArrays.SparseMatrixCSC{Float64,Int}(LinearAlgebra.I(size(lzB,1)))*lzB
    return FullGenerator(B)
end

