struct FRAPMesh <: Mesh 
    nodes::Array{Float64,1}
    # n_bases::Int
    me::AbstractMatrixExponential
    # Fil::IndexDict
end 
FRAPMesh(nodes::Array{Float64,1},n_bases::Int) = FRAPMesh(nodes,MakeME(CMEParams[n_bases]))

# Convenience constructors
# function FRAPMesh(
#     model::Model,
#     nodes::Array{<:Real,1},
#     n_bases::Int;
#     # Fil::IndexDict=IndexDict(),
#     v::Bool = false,
# )

#     # ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
#     # if isempty(Fil)
#     #     Fil = MakeFil(model, nodes)
#     # end

#     mesh = FRAPMesh(
#         nodes,
#         n_bases,
#         Fil,
#     )
#     v && println("UPDATE: DGMesh object created with fields ", fieldnames(DGMesh))
#     return mesh
# end
function FRAPMesh()
    FRAPMesh(Array{Float64,1}(undef,0),0)
end

"""

    n_bases(mesh::FRAPMesh)
    
Number of bases in a cell
"""
n_bases(mesh::FRAPMesh) = _order(mesh.me)


"""

    cell_nodes(mesh::FRAPMesh)

The cell centre
"""
cell_nodes(mesh::FRAPMesh) = Array(((mesh.nodes[1:end-1] + mesh.nodes[2:end]) / 2 )')

"""

    basis(mesh::FRAPMesh)

Constant ""
"""
basis(mesh::FRAPMesh) = ""

function MakeLazyGenerator(
    model::Model,
    mesh::FRAPMesh;
    v::Bool=false,
)
    me = mesh.me::AbstractMatrixExponential
    blocks = (me.s*me.a, me.S, me.s*me.a)

    boundary_flux = (in = me.s[:], out = me.a[:])

    # T = model.T
    # C = model.C
    # delta = Δ(mesh)
    D = me.D

    # signChangeIndex = zeros(Bool,NPhases(model),NPhases(model))
    # for i in 1:NPhases(model), j in 1:NPhases(model)
    #     if ((sign(model.C[i])!=0) && (sign(model.C[j])!=0))
    #         signChangeIndex[i,j] = (sign(model.C[i])!=sign(model.C[j]))
    #     elseif (sign(model.C[i])==0)
    #         signChangeIndex[i,j] = sign(model.C[j])>0
    #     elseif (sign(model.C[j])==0)
    #         signChangeIndex[i,j] = sign(model.C[i])>0            
    #     end
    # end
    
    out = LazyGenerator(model,mesh,blocks,boundary_flux,D)
    v && println("UPDATE: LazyGenerator object created with keys ", keys(out))
    return out
end
# function MakeLazyGenerator(model::Model, mesh::FRAPMesh; v::Bool=false)
#     me = MakeME(CMEParams[n_bases(mesh)])
#     return MakeLazyGenerator(model, mesh, me; v=v)
# end
# function MakeFullGenerator(model::Model, mesh::Mesh, me::AbstractMatrixExponential; v::Bool=false)
#     lazy = MakeLazyGenerator(model,mesh,me; v=v)
#     return materialise(lazy)
# end


# function MakeB(model::Model, mesh::FRAPMesh, me::ME)
#     N₊ = sum(model.C .>= 0)
#     N₋ = sum(model.C .<= 0)

#     F = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
#     UpDiagBlock = me.s*me.a
#     LowDiagBlock = me.s*me.a
#     for i = ["+","-"]
#         F[i] = SparseArrays.spzeros(Float64, total_n_bases(mesh), total_n_bases(mesh))
#         for k = 1:NIntervals(mesh)
#             idx = (1:n_bases(mesh)) .+ (k - 1) * n_bases(mesh)
#             if k > 1
#                 idxup = (1:n_bases(mesh)) .+ (k - 2) * n_bases(mesh)
#                 if i=="+"
#                     F[i][idxup, idx] = UpDiagBlock
#                 elseif i=="-"
#                     F[i][idx, idxup] = LowDiagBlock
#                 end # end if C[i]
#             end # end if k>1
#         end # for k in ...
#     end # for i in ...

#     signChangeIndex = zeros(Bool,NPhases(model),NPhases(model))
#     for i in 1:NPhases(model), j in 1:NPhases(model)
#         if ((sign(model.C[i])!=0) && (sign(model.C[j])!=0))
#             signChangeIndex[i,j] = (sign(model.C[i])!=sign(model.C[j]))
#         elseif (sign(model.C[i])==0)
#             signChangeIndex[i,j] = sign(model.C[j])>0
#         elseif (sign(model.C[j])==0)
#             signChangeIndex[i,j] = sign(model.C[i])>0            
#         end
#     end
#     B = SparseArrays.spzeros(
#         Float64,
#         NPhases(model) * total_n_bases(mesh) + N₋ + N₊,
#         NPhases(model) * total_n_bases(mesh) + N₋ + N₊,
#     )
#     B[N₋+1:end-N₊,N₋+1:end-N₊] = SparseArrays.kron(
#             model.T.*signChangeIndex,
#             SparseArrays.kron(SparseArrays.I(NIntervals(mesh)),me.D)
#         ) + SparseArrays.kron(
#             model.T.*(1 .- signChangeIndex),
#             SparseArrays.I(total_n_bases(mesh))
#         )

#     # Boundary conditions
#     T₋₋ = model.T[model.C.<=0,model.C.<=0]
#     T₊₋ = model.T[model.C.>=0,:].*((model.C.<0)')
#     T₋₊ = model.T[model.C.<=0,:].*((model.C.>0)')
#     T₊₊ = model.T[model.C.>=0,model.C.>=0]
#     # yuck
#     inLower = [
#         LinearAlgebra.kron(LinearAlgebra.diagm(abs.(model.C).*(model.C.<=0)),me.s)[:,model.C.<=0]; 
#         LinearAlgebra.zeros((NIntervals(mesh)-1)*NPhases(model)*n_bases(mesh),N₋)
#     ]
#     outLower = [
#         LinearAlgebra.kron(T₋₊,me.a) LinearAlgebra.zeros(N₋,N₊+(NIntervals(mesh)-1)*NPhases(model)*n_bases(mesh))
#     ]
#     inUpper = [
#         LinearAlgebra.zeros((NIntervals(mesh)-1)*NPhases(model)*n_bases(mesh),N₊);
#         LinearAlgebra.kron(LinearAlgebra.diagm(abs.(model.C).*(model.C.>=0)),me.s)[:,model.C.>=0]
#     ]
#     outUpper = [
#         LinearAlgebra.zeros(N₊,N₋+(NIntervals(mesh)-1)*NPhases(model)*n_bases(mesh)) LinearAlgebra.kron(T₊₋,me.a)
#     ]
    
#     QBDidx = MakeQBDidx(model,mesh)
#     B[1:N₋,QBDidx] = [T₋₋ outLower]
#     B[end-N₊+1:end,QBDidx] = [outUpper T₊₊]
#     B[QBDidx[N₋+1:end-N₊],1:N₋] = inLower
#     B[QBDidx[N₋+1:end-N₊],(end-N₊+1):end] = inUpper
#     for i = 1:NPhases(model)
#         idx = ((i-1)*total_n_bases(mesh)+1:i*total_n_bases(mesh)) .+ N₋
#         if model.C[i] > 0
#             B[idx, idx] += model.C[i] * (SparseArrays.kron(
#                     SparseArrays.I(NIntervals(mesh)), me.S
#                     ) + F["+"])
#         elseif model.C[i] < 0
#             B[idx, idx] += abs(model.C[i]) * (SparseArrays.kron(
#                 SparseArrays.I(NIntervals(mesh)), me.S
#                 ) + F["-"])
#         end
#     end

#     BDict = MakeDict(B, model, mesh)

#     return FullGenerator(BDict, B, mesh.Fil)
# end

# function MakeB(model::Model, mesh::FRAPMesh, order::Int)
#     me = MakeME(CMEParams[order], mean = 1)
#     return MakeB(model, mesh, me)
# end