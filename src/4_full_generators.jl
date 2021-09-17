struct FullGenerator <: Generator 
    B::Union{Array{Float64,Int64}, SparseArrays.SparseMatrixCSC{Float64,Int}}
end

size(B::FullGenerator) = size(B.B)
getindex(B::FullGenerator,i::Int,j::Int) = B.B[i,j]
setindex!(B::FullGenerator,x::Float64,i::Int,j::Int) = B.B[i,j]=x

+(A::AbstractArray{<:Real,2}, B::FullGenerator) = A+B.B
+(B::FullGenerator, A::AbstractArray{<:Real,2}) = B.B+A
+(A::FullGenerator, B::FullGenerator) = A.B+B.B
-(A::AbstractArray{<:Real,2}, B::FullGenerator) = A-B.B
-(B::FullGenerator, A::AbstractArray{<:Real,2}) = B.B-A
-(A::FullGenerator, B::FullGenerator) = A.B-B.B
*(A::AbstractArray{<:Real,2}, B::FullGenerator) = A*B.B
*(B::FullGenerator, A::AbstractArray{<:Real,2}) = B.B*A
*(A::FullGenerator, B::FullGenerator) = A.B*B.B


function show(io::IO, mime::MIME"text/plain", B::FullGenerator)
    if VERSION >= v"1.6"
        show(io, mime, B.B)
    else
        show(io, mime, Matrix(B.B))
    end
end

function MakeFullGenerator(model::Model, mesh::Mesh; v::Bool=false)
    lazy = MakeLazyGenerator(model,mesh; v=v)
    return materialise(lazy)
end

function materialise(lzB::LazyGenerator)
    B = SparseArrays.SparseMatrixCSC{Float64,Int}(LinearAlgebra.I(size(lzB,1)))*lzB
    return FullGenerator(B)
end