struct FullGenerator <: Generator 
    # BDict::PartitionedGenerator
    B::Union{Array{Float64,Int64}, SparseArrays.SparseMatrixCSC{Float64,Int}}
    # Fil::IndexDict
end

FullGenerator(BDict::PartitionedGenerator,B::Union{Array{Float64,Int64}, SparseArrays.SparseMatrixCSC{Float64,Int}}) = 
    FullGenerator(BDict,B,IndexDict())

size(B::FullGenerator) = size(B.B)
getindex(B::FullGenerator,i::Int,j::Int) = B.B[i,j]
getindex(B::FullGenerator,plus_minus_index::PlusMinusIndex,phase_index::PhaseIndex) = 
    B.BDict[plus_minus_index, phase_index]
getindex(B::FullGenerator,idx::Tuple{PlusMinusIndex,PhaseIndex}) = 
    getindex(B,idx[1],idx[2])
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
        println("\n with partitions")
        show(io, mime, keys(B.BDict))
    else
        show(io, mime, Matrix(B.B))
        println("\n with partitions")
        show(io, mime, keys(B.BDict))
    end
end
# show(B::FullGenerator) = show(stdout, B)

function MakeLazyGenerator(model::Model, mesh::Mesh; v::Bool=false)
    throw(DomainError("Can construct LazyGenerator for DGMesh, FRAPMesh, only"))
end
function MakeFullGenerator(model::Model, mesh::Mesh; v::Bool=false)
    lazy = MakeLazyGenerator(model,mesh; v=v)
    return materialise(lazy)
end

function materialise(lzB::LazyGenerator)
    B = SparseArrays.SparseMatrixCSC{Float64,Int}(LinearAlgebra.I(size(lzB,1)))*lzB
    # BDict = MakeDict(B,lzB.C,size(lzB.D,1),lzB.Fil)
    # BDict = MakeDict(lzB) # in the interest of speed use the above
    return FullGenerator(BDict,B)
end