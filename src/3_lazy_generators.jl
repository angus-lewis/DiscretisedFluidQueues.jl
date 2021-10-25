"""
    Generator

Abstract type representing a discretised infinitesimal generator of a FLuidQueue. 
Behaves much like a square matrix. 
"""
abstract type Generator{T<:Mesh} <: AbstractMatrix{Float64} end 
# checksquare(A::Generator) = !(size(A,1)==size(A,2)) ? throw(DomainError(A," must be square")) : nothing

const UnionVectors = Union{StaticArrays.SVector,Vector{Float64}}
const UnionArrays = Union{Array{Float64,2},StaticArrays.SMatrix}
const BoundaryFluxTupleType = Union{
    NamedTuple{(:upper,:lower), Tuple{
        NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}},
        NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}}
        }
    },
    NamedTuple{(:upper, :lower), Tuple{
        NamedTuple{(:in, :out), Tuple{StaticArrays.SVector, StaticArrays.SVector}},
        NamedTuple{(:in, :out), Tuple{StaticArrays.SVector, StaticArrays.SVector}}
        }
    }
}

struct OneBoundaryFlux{T<:Union{StaticArrays.SVector,Vector{Float64}}}
    in::T
    out::T
end
struct BoundaryFlux{T<:Union{StaticArrays.SVector,Vector{Float64}}}
    upper::OneBoundaryFlux{T}
    lower::OneBoundaryFlux{T}
end


"""
    LazyGenerator{<:Mesh} <: Generator{<:Mesh}

A lazy representation of a block matrix with is a generator of a DiscretisedFluidQueue.

Lower memory requirements than FullGenerator but aritmetic operations and indexing may be slower.

# Arguments:
- `dq::DiscretisedFluidQueue`: 
- `blocks::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}`: 
    Block matrices describing the flow of mass within and between cells. `blocks[1]` is the lower 
    diagonal block describing the flow of mass from cell k+1 to cell (k for phases 
    with negative rate only). `blocks[2] (blocks[3])` is the 
    diagonal block describing the flow of mass within a cell for a phase with positive (negative) rate.
    `blocks[4]` is the upper diagonal block describing the flow of mass from cell k to k+1 (for phases 
    with positive rate only).  
- `boundary_flux::BoundaryFlux`: A named tuple structure such that 
        - `boundary_flux.lower.in`: describes flow of density into lower boundary
        - `boundary_flux.lower.out`: describes flow of density out of lower boundary
        - `boundary_flux.upper.in`: describes flow of density into upper boundary
        - `boundary_flux.upper.out`: describes flow of density out of upper boundary
- `D::Union{Array{Float64, 2}, LinearAlgebra.Diagonal{Bool, Array{Bool, 1}}}`: An array describing 
    how the flow of density changes when the phase process jumps between phases with different memberships.
    This is the identity for FV and DG schemes. 
"""
struct LazyGenerator{T}  <: Generator{T}
    dq::DiscretisedFluidQueue{T}
    blocks::NTuple{4,AbstractMatrix{Float64}}
    boundary_flux::BoundaryFlux
    D::AbstractMatrix{Float64}
    function LazyGenerator(
        dq::DiscretisedFluidQueue{T},
        blocks::NTuple{4,AbstractMatrix{Float64}},
        boundary_flux::BoundaryFlux,
        D::AbstractMatrix{Float64},
    ) where T
        s = size(blocks[1])
        for b in 1:4
            checksquare(blocks[b]) 
            !(s == size(blocks[b])) && throw(DomainError("blocks must be the same size"))
        end
        checksquare(D)
        !(s == size(D)) && throw(DomainError("blocks must be the same size as D"))
        
        return new{T}(dq,blocks,boundary_flux,D)
    end
end
function LazyGenerator(
    dq::DiscretisedFluidQueue,
    blocks::NTuple{3,AbstractMatrix{Float64}},
    boundary_flux::OneBoundaryFlux,
    D::AbstractMatrix{Float64},
)
    blocks = (blocks[1],blocks[2],blocks[2],blocks[3])
    boundary_flux = BoundaryFlux(boundary_flux, boundary_flux)
    return LazyGenerator(dq,blocks,boundary_flux,D)
end

"""
    @static_generator(lz)

Convert the block matrices and vectors within `lz` to `StaticArrays`.
This is not much faster. I think all the conditionals in `*` are the main 
bottle-neck for speed... 
"""
macro static_generator(lz)
    out = quote 
        tmp = $(esc(lz))
        sz = size(tmp.blocks[1], 1)
        ex_smatrix = StaticArrays.SMatrix{sz, sz, Float64}
        ex_svector = StaticArrays.SVector{sz, Float64}
        b1 = ex_smatrix(tmp.blocks[1])
        b2 = ex_smatrix(tmp.blocks[2])
        b3 = ex_smatrix(tmp.blocks[3])
        b4 = ex_smatrix(tmp.blocks[4])
        uprin = ex_svector(tmp.boundary_flux.upper.in)
        uprout = ex_svector(tmp.boundary_flux.upper.out)
        lwrin = ex_svector(tmp.boundary_flux.lower.in)
        lwrout = ex_svector(tmp.boundary_flux.lower.out)
        D = ex_smatrix(tmp.D)
        dq = tmp.dq

        # return 
        LazyGenerator(dq,(b1,b2,b3,b4), 
            BoundaryFlux(OneBoundaryFlux(uprin,uprout),OneBoundaryFlux(lwrin,lwrout)),
            D)
    end
    return out
end

function static_generator(lz) 
    sz = size(lz.blocks[1], 1)
    ex_smatrix = StaticArrays.SMatrix{sz, sz, Float64}
    ex_svector = StaticArrays.SVector{sz, Float64}
    b1 = ex_smatrix(lz.blocks[1])
    b2 = ex_smatrix(lz.blocks[2])
    b3 = ex_smatrix(lz.blocks[3])
    b4 = ex_smatrix(lz.blocks[4])
    uprin = ex_svector(lz.boundary_flux.upper.in)
    uprout = ex_svector(lz.boundary_flux.upper.out)
    lwrin = ex_svector(lz.boundary_flux.lower.in)
    lwrout = ex_svector(lz.boundary_flux.lower.out)
    D = ex_smatrix(lz.D)
    dq = lz.dq

    # return 
    out = LazyGenerator(dq,(b1,b2,b3,b4), 
        BoundaryFlux(OneBoundaryFlux(uprin,uprout),OneBoundaryFlux(lwrin,lwrout)),
        D)
    return out
end

# I think this is a duplicate: delete?

# function LazyGenerator(
#     blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}},
#     boundary_flux::NamedTuple{(:in, :out),Tuple{Array{Float64,1},Array{Float64,1}}},
#     D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
# )
#     blocks = (blocks[1],blocks[2],blocks[2],blocks[3])
#     boundary_flux = (upper = boundary_flux, lower = boundary_flux)
#     return LazyGenerator(blocks, boundary_flux, D)
# end

"""
    build_lazy_generator(dq::DiscretisedFluidQueue; v::Bool = false)

Build a lazy representation of the generator of a discretised fluid queue.
"""
function build_lazy_generator(dq::DiscretisedFluidQueue; v::Bool=false)
    throw(DomainError("Can construct LazyGenerator for DGMesh, FRAPMesh, only"))
end

"""
    size(B::LazyGenerator)
"""
function size(B::LazyGenerator)
    sz = total_n_bases(B.dq) + N₋(B.dq) + N₊(B.dq)
    return (sz,sz)
end
size(B::LazyGenerator, n::Int) = 
    (n∈[1,2]) ? size(B)[n] : throw(DomainError("Lazy generator is a matrix, index must be 1 or 2"))
# length(B::LazyGenerator) = prod(size(B))
# iterate(v::LazyGenerator, i=1) = (length(v) < i ? nothing : (v[i], i + 1))
# Base.BroadcastStyle(::Type{<:LazyGenerator}) = Broadcast.ArrayStyle{LazyGenerator}()

_check_phase_index(i::Int,model::Model) = 
    (i∉phases(model)) && throw(DomainError("i is not a valid phase in model"))
_check_mesh_index(k::Int,mesh::Mesh) = 
    !(1<=k<=n_intervals(mesh)) && throw(DomainError("k in not a valid cell"))
_check_basis_index(p::Int,mesh::Mesh) = !(1<=p<=n_bases_per_cell(mesh))

# function _map_to_index_interior(i::Int,k::Int,p::Int,dq::DiscretisedFluidQueue)
#     # i phase
#     # k cell
#     # p basis
    
#     _check_phase_index(i,dq.model)
#     _check_mesh_index(k,dq.mesh)
#     _check_basis_index(p,dq.mesh)

#     P = n_bases_per_cell(dq)
#     KP = n_bases_per_phase(dq)

#     idx = (i-1)*KP + (k-1)*P + p
#     return N₋(dq) + idx 
# end
# function _map_to_index_boundary(i::Int,dq::DiscretisedFluidQueue)
#     # i phase
#     _check_phase_index(i,dq.model)
#     if _has_left_boundary(dq.model.S,i) 
#         idx = N₋(dq.model.S[1:i])
#     else _has_right_boundary(dq.model.S,i)
#         N = n_phases(dq)
#         KP = n_bases_per_phase(dq)
#         idx = N₊(dq.model.S[1:i]) + KP*N + N₋(dq)
#     end
#     return idx
# end

function _map_to_index_interior((k,ip)::NTuple{3,Int},dq::DiscretisedFluidQueue)
    # i phase
    # k cell
    # p basis
    
    _check_phase_index(i,dq.model)
    _check_mesh_index(k,dq.mesh)
    _check_basis_index(p,dq.mesh)

    idx = (i-1)*n_bases_per_cell(dq) + (k-1)*n_bases_per_level(dq) + p
    return N₋(dq) + idx 
end
function _map_to_index_boundary((k,i,p)::NTuple{3,Int},dq::DiscretisedFluidQueue)
    # i phase
    # k cell
    # p basis
    _check_phase_index(i,dq.model)
    !(p==1)&&throw(DomainError("p must be 1 at the boundary"))
    !((k==0)||(k==n_intervals(dq)+1))&&throw(DomainError("k not a boundary index of dq"))
    if (k==0)&&(_has_left_boundary(dq.model.S,i))
        idx = N₋(dq.model.S[1:i])
    elseif (k==n_intervals(dq)+1)&&_has_right_boundary(dq.model.S,i)
        idx = N₊(dq.model.S[1:i]) + total_n_bases(dq) + N₋(dq)
    else 
        throw(DomainError(string((i,k,p))*" is not a valid index of dq"))
    end
    return idx 
end
function _map_to_index((i,k,p)::NTuple{3,Int},dq::DiscretisedFluidQueue)
    if (k==0)||(k==n_intervals(dq)+1)
        idx = _map_to_index_boundary((k,i,p),dq)
    else 
        idx = _map_to_index_interior((k,i,p),dq)
    end
    return idx
end
# function Base.getindex(B::Generator,(i,k,p)::NTuple{3,Int},(j,l,q)::NTuple{3,Int})
#     row = _map_to_index((i,k,p),B.dq)
#     col = _map_to_index((j,l,q),B.dq)
#     return B[row,col]
# end
function Base.getindex(B::Generator,(is,ks,ps)::Tuple,(js,ls,qs)::Tuple)
    (typeof(is)==Colon)&&(is=1:n_phases(B.dq))
    (typeof(js)==Colon)&&(js=1:n_phases(B.dq))
    (typeof(ks)==Colon) ? (ks=0:n_intervals(B.dq)+1; ks_colon=true) : (ks_colon=false)
    (typeof(ls)==Colon) ? (ls=0:n_intervals(B.dq)+1; ls_colon=true) : (ls_colon=false)
    (typeof(ps)==Colon)&&(ps=1:n_bases_per_cell(B.dq))
    (typeof(qs)==Colon)&&(qs=1:n_bases_per_cell(B.dq))
    
    rows = Int[]
    for i in is
        for k in ks 
            for p in ps 
                if (k==0)&&(p==1)&&_has_left_boundary(B.dq.model.S,i)
                    push!(rows,(_map_to_index((i,0,1),B.dq)*ones(Int,sum(ps.==1)))...)
                    break
                elseif (k==n_intervals(B.dq)+1)&&(p==1)&&_has_right_boundary(B.dq.model.S,i)
                    push!(rows,(_map_to_index((i,n_intervals(B.dq)+1,1)*ones(Int,sum(ps.==1)),B.dq))...)
                    break
                elseif (k!=0)&&(k!=n_intervals(B.dq)+1)
                    push!(rows,_map_to_index((i,k,p),B.dq))
                end
            end
        end
    end

    cols = Int[]
    for j in js
        for l in ls 
            for q in qs 
                if (l==0)&&(q==1)&&_has_left_boundary(B.dq.model.S,j)
                    push!(cols,(_map_to_index((j,0,1),B.dq)*ones(Int,sum(qs.==1)))...)
                    break
                elseif (l==n_intervals(B.dq)+1)&&(q==1)&&_has_right_boundary(B.dq.model.S,j)
                    push!(cols,(_map_to_index((j,n_intervals(B.dq)+1,1),B.dq)*ones(Int,sum(qs.==1)))...)
                    break
                elseif (l!=0)&&(l!=n_intervals(B.dq)+1)
                    push!(cols,_map_to_index((j,l,q),B.dq))
                end
            end
        end
    end


    # rows = [_map_to_index((i,k,p),B.dq) for i in is for k in ks for p in ps]
    # cols = [_map_to_index((j,l,q),B.dq) for j in js for l in ls for q in qs]

    # if (ks_colon)&&(1∈ps)
    #     lwr_bndry_idx_row = Int[]
    #     upr_bndry_idx_row = Int[]
    #     for i in is
    #         _has_left_boundary(B.dq.model.S,i)&&push!(lwr_bndry_idx_row,_map_to_index((i,0,1),B.dq))
    #         _has_right_boundary(B.dq.model.S,i)&&push!(upr_bndry_idx_row,_map_to_index((i,n_intervals(B.dq)+1,1),B.dq))
    #     end
    #     rows = [lwr_bndry_idx_row; rows; upr_bndry_idx_row]
    # end
    # if (ls_colon)&&(1∈ps)
    #     lwr_bndry_idx_col = Int[]
    #     upr_bndry_idx_col = Int[]
    #     for i in is
    #         _has_left_boundary(B.dq.model.S,i)&&push!(lwr_bndry_idx_col,_map_to_index((i,0,1),B.dq))
    #         _has_right_boundary(B.dq.model.S,i)&&push!(upr_bndry_idx_col,_map_to_index((i,n_intervals(B.dq)+1,1),B.dq))
    #     end
    #     cols = [lwr_bndry_idx_col; cols; upr_bndry_idx_col]
    # end
    # @show rows
    # @show cols 
    return B[rows,cols]
end

_is_boundary_index(n::Int,B::LazyGenerator) = (n∈1:N₋(B.dq))||(n∈(size(B,1).-(0:N₊(B.dq)-1)))
function _map_from_index_interior(n::Int,B::LazyGenerator)
    # n matrix index to map to phase, cell, basis
    (!(1<=n<=size(B,1))||_is_boundary_index(n,B))&&throw(DomainError(n,"not a valid interior index"))
    
    n -= (N₋(B.dq)+1)
    P = n_bases_per_cell(B.dq)

    k = n÷n_bases_per_level(B.dq) + 1
    i = mod(n,n_bases_per_level(B.dq))÷n_bases_per_cell(B.dq) + 1
    p = mod(n,P) + 1
    return k, i, p
end
function _map_from_index_boundary(n::Int,B::LazyGenerator)
    # n matrix index to map to phase at boundary
    (!_is_boundary_index(n,B))&&throw(DomainError("not a valid boundary index"))
    
    if n>N₋(B.dq)
        i₊ = n-N₋(B.dq)-total_n_bases(B.dq)
        i = 1
        for j in phases(B.dq)
            (i₊==N₊(B.dq.model.S[1:j])) && break
            i += 1
        end
    else 
        i₋ = n
        i = 1
        for j in phases(B.dq)
            (i₋==N₋(B.dq.model.S[1:j])) && break
            i += 1
        end
    end

    return i
end


function fast_mul(u::AbstractMatrix{Float64}, B::LazyGenerator)
    output_type = typeof(u)
    
    sz_u_1 = size(u,1)
    sz_u_2 = size(u,2)
    sz_B_1 = size(B,1)
    sz_B_2 = size(B,2)
    !(sz_u_2 == sz_B_1) && throw(DomainError("Dimension mismatch, u*B, length(u) must be size(B,1)"))

    if output_type <: SparseArrays.SparseMatrixCSC
        v = SparseArrays.spzeros(sz_u_1,sz_B_2)
    else 
        v = zeros(sz_u_1,sz_B_2)
    end

    Kp = n_bases_per_cell(B.dq) # K = n_intervals(mesh), p = n_bases_per_cell(mesh)
    C = rates(B.dq)
    n₋ = N₋(B.dq)
    n₊ = N₊(B.dq)

    # boundaries
    # at lower
    _lmul_at_lwr_bndry!(v,u,B.dq.model.T,B.dq.model.S,n₋) 
    # in to lower 
    _lmul_into_lwr_bndry!(v,u,B.dq.model,B.dq.mesh,Kp,n₋,C,B.boundary_flux.lower.in)

    # out of lower 
    _lmul_out_lwr_bndry!(v,u,B.dq.mesh,B.dq.model,Kp,n₋,B.boundary_flux.lower.in,B.boundary_flux.lower.out)

    # at upper
    _lmul_at_upr_bndry!(v,u,B.dq.model.T,B.dq.model.S,n₊) 
    # in to upper
    _lmul_into_upr_bndry!(v,u,B.dq.model,B.dq.mesh,Kp,n₋,n₊,C,B.boundary_flux.upper.in)
    
    # out of upper 
    _lmul_out_upr_bndry!(v,u,B.dq.mesh,B.dq.model,Kp,n₋,n₊,B.boundary_flux.upper.in,B.boundary_flux.upper.out)

    # innards
    _lmul_innards!(v,u,B.dq.model,B.dq.mesh,B.blocks,B.D,C,n₋)
    return v
end
function _lmul_innards_generic!(v,u,model,mesh,tmp_blocks,D,C,n₋,k)
    for j in 1:n_phases(model)
        i_idx = (k-1)*n_phases(model)*n_bases_per_cell(mesh) + n₋ .+ (1:n_bases_per_cell(mesh)) 
        j_idx = i_idx .+ (j-1)*n_bases_per_cell(mesh) 
        (k>1) && ((C[j]>0.0) && (v[:,j_idx] += C[j]*u[:,j_idx.-n_bases_per_cell(mesh)*n_phases(model)]*tmp_blocks[4]))
        (k<n_intervals(mesh)) && ((C[j]<0.0) && (v[:,j_idx] += abs(C[j])*u[:,j_idx.+n_bases_per_cell(mesh)*n_phases(model)]*tmp_blocks[1]))
        for i in 1:n_phases(model)
            if (typeof(mesh)<:FRAPMesh)&&(membership(model.S,i)!=membership(model.S,j)) # + to - change
                v[:,j_idx] += u[:,i_idx]*D*model.T[i,j]
            elseif (i!=j)||(C[i]==0.0)
                v[:,j_idx] += u[:,i_idx]*model.T[i,j]
            elseif C[i]>0.0 # i==j
                v[:,j_idx] += u[:,i_idx]*C[i]*tmp_blocks[2] + u[:,i_idx]*model.T[i,j]
            elseif C[i]<0.0 # i==j
                v[:,j_idx] += u[:,i_idx]*abs(C[i])*tmp_blocks[3] + u[:,i_idx]*model.T[i,j]
            end
            i_idx = i_idx .+ n_bases_per_cell(mesh)
        end
    end
end
function _lmul_innards!(v,u,model,mesh::Mesh{Vector{Float64}},blocks,D,C,n₋)
    for k in 1:n_intervals(mesh)
        if 1<k<n_intervals(mesh) 
            tmp_blocks = (blocks[1]/Δ(mesh,k+1), blocks[2]/Δ(mesh,k), blocks[3]/Δ(mesh,k), blocks[4]/Δ(mesh,k-1))
        elseif k==1
            tmp_blocks = (blocks[1]/Δ(mesh,k+1), blocks[2]/Δ(mesh,k), blocks[3]/Δ(mesh,k), zeros(0,0))
        else
            tmp_blocks = (zeros(0,0), blocks[2]/Δ(mesh,k), blocks[3]/Δ(mesh,k), blocks[4]/Δ(mesh,k-1))
        end
        _lmul_innards_generic!(v,u,model,mesh,tmp_blocks,D,C,n₋,k)
    end
end
function _lmul_innards!(v,u,model,mesh::Mesh{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}},blocks,D,C,n₋)
    tmp_blocks = (blocks[1]/Δ(mesh,1), blocks[2]/Δ(mesh,1), blocks[3]/Δ(mesh,1), blocks[4]/Δ(mesh,1))
    for k in 1:n_intervals(mesh)
        _lmul_innards_generic!(v,u,model,mesh,tmp_blocks,D,C,n₋,k)
    end
end


# function _fast_mul(u::AbstractMatrix{Float64}, B::LazyGenerator)
#     output_type = typeof(u)
    
#     sz_u_1 = size(u,1)
#     sz_u_2 = size(u,2)
#     sz_B_1 = size(B,1)
#     sz_B_2 = size(B,2)
#     !(sz_u_2 == sz_B_1) && throw(DomainError("Dimension mismatch, u*B, length(u) must be size(B,1)"))

#     if output_type <: SparseArrays.SparseMatrixCSC
#         v = SparseArrays.spzeros(sz_u_1,sz_B_2)
#     else 
#         v = zeros(sz_u_1,sz_B_2)
#     end
#     model = B.dq.model
#     mesh = B.dq.mesh
#     Kp = n_bases_per_phase(B.dq) # K = n_intervals(mesh), p = n_bases_per_cell(mesh)
#     C = rates(B.dq)
#     n₋ = N₋(B.dq)
#     n₊ = N₊(B.dq)

#     # boundaries
#     # at lower
#     _lmul_at_lwr_bndry!(v,u,model.T,model.S,n₋) 
#     # in to lower 
#     _lmul_into_lwr_bndry!(v,u,model,mesh,Kp,n₋,C,B.boundary_flux.lower.in)

#     # out of lower 
#     _lmul_out_lwr_bndry!(v,u,mesh,model,Kp,n₋,B.boundary_flux.lower.in,B.boundary_flux.lower.out)

#     # at upper
#     _lmul_at_upr_bndry!(v,u,model.T,model.S,n₊) 
#     # in to upper
#     _lmul_into_upr_bndry!(v,u,model,mesh,Kp,n₋,n₊,C,B.boundary_flux.upper.in)
    
#     # out of upper 
#     _lmul_out_upr_bndry!(v,u,mesh,model,Kp,n₋,n₊,B.boundary_flux.upper.in,B.boundary_flux.upper.out)
    
#     # innards
#     for i in phases(model), j in phases(model)
#         if i == j 
#             # mult on diagonal
#             if C[i]>0.0 
#                 _lmul_ii_pos_diag_block!(v,u,i,mesh,n₋,
#                     C[i]*B.blocks[2], model.T[i,i],
#                     C[i]*B.blocks[4],
#                     n_bases_per_cell(mesh),Kp)
#             elseif C[i]<0.0
#                 _lmul_ii_neg_diag_block!(v,u,i,mesh,n₋,
#                     abs(C[i])*B.blocks[3], model.T[i,i],
#                     abs(C[i])*B.blocks[1],
#                     n_bases_per_cell(mesh),Kp)
#             else
#                 _lmul_ii_0_diag_block!(v,u,i,n_intervals(mesh),n₋,model.T[i,i],n_bases_per_cell(mesh),Kp)
#             end
#         elseif typeof(mesh)<:FRAPMesh
#             _lmul_ij_off_diag_blocks_FRAP!(v,u,i,j,model.S,model.T[i,j],B.D,Kp,n₋,n_intervals(mesh),n_bases_per_cell(mesh))
#         else
#             i_idx = (i-1)*Kp .+ (1:Kp) .+ n₋
#             j_idx = (j-1)*Kp .+ (1:Kp) .+ n₋
#             v[:,j_idx] += (u[:,i_idx]*model.T[i,j])
#         end
#     end
#     return v
# end

function _lmul_at_lwr_bndry!(v,u,T,S,n₋)
    v[:,1:n₋]+=u[:,1:n₋]*T[_has_left_boundary.(S),_has_left_boundary.(S)]
    return nothing 
end
function _lmul_at_upr_bndry!(v,u,T,S,n₊)
    v[:,end-n₊+1:end]+=u[:,end-n₊+1:end]*T[_has_right_boundary.(S),_has_right_boundary.(S)]
    return nothing 
end
function _lmul_into_lwr_bndry!(v,u,model::BoundedFluidQueue,mesh,Kp,n₋,C,bndry_flux_in_lwr)
    idxdown = n₋ .+ ((1:n_bases_per_cell(mesh)).+Kp*(findall(negative_phases(model)) .- 1)')[:]
    v[:,1:n₋] += u[:,idxdown]*LinearAlgebra.kron(
        # LinearAlgebra.diagm(0 => abs.(C[_has_left_boundary.(model.S)])),
        abs.(C[negative_phases(model)]).*model.P_lwr[:,_has_left_boundary.(model.S)],
        bndry_flux_in_lwr/Δ(mesh,1),
    )
    return nothing
end
function _lmul_into_upr_bndry!(v,u,model::BoundedFluidQueue,mesh,Kp,n₋,n₊,C,bndry_flux_in_upr)
    idxup = n₋ .+ ((1:n_bases_per_cell(mesh)) .+ Kp*(findall(positive_phases(model)) .- 1)')[:] .+
        (n_bases_per_phase(mesh)*n_phases(model).-n_bases_per_cell(mesh)*n_phases(model))
    v[:,end-n₊+1:end] += u[:,idxup]*LinearAlgebra.kron(
        # LinearAlgebra.diagm(0 => C[_has_right_boundary.(model.S)]),
        C[positive_phases(model)].*model.P_upr[:,_has_right_boundary.(model.S)],
        bndry_flux_in_upr/Δ(mesh,n_intervals(mesh)),
    )
    return nothing
end
function _lmul_out_lwr_bndry_generic!(v,u,mesh,model,Kp,n₋,bndry_flux_out_lwr)
    idxup = n₋ .+ (Kp*(findall(positive_phases(model)).-1)' .+ (1:n_bases_per_cell(mesh)))[:]
    v[:,idxup] += u[:,1:n₋]*LinearAlgebra.kron(model.T[_has_left_boundary.(model.S),positive_phases(model)],bndry_flux_out_lwr')
    return idxup
end
function _lmul_out_lwr_bndry!(v,u,mesh,model::BoundedFluidQueue,Kp,n₋,bndry_flux_in_lwr,bndry_flux_out_lwr)
    idxup = _lmul_out_lwr_bndry_generic!(v,u,mesh,model,Kp,n₋,bndry_flux_out_lwr)
    idxdown = n₋ .+ ((1:n_bases_per_cell(mesh)).+Kp*(findall(negative_phases(model)) .- 1)')[:]
    v[:,idxup] += u[:,idxdown]*LinearAlgebra.kron(
        # LinearAlgebra.diagm(0 => abs.(C[_has_left_boundary.(model.S)])),
        abs.(rates(model)[negative_phases(model)]).*model.P_lwr[:,positive_phases(model)],
        bndry_flux_in_lwr*bndry_flux_out_lwr'/Δ(mesh,1),
    )
    return nothing 
end
function _lmul_out_upr_bndry_generic!(v,u,mesh,model,Kp,n₋,n₊,bndry_flux_out_upr)
    idxdown = n₋ .+ (Kp*(findall(negative_phases(model)).-1)' .+ (1:n_bases_per_cell(mesh)))[:] .+
        (n_bases_per_phase(mesh)*n_phases(model).-n_bases_per_cell(mesh)*n_phases(model))
    v[:,idxdown] += u[:,end-n₊+1:end]*LinearAlgebra.kron(model.T[_has_right_boundary.(model.S),negative_phases(model)],bndry_flux_out_upr')
    return idxdown
end
function _lmul_out_upr_bndry!(v,u,mesh,model::BoundedFluidQueue,Kp,n₋,n₊,bndry_flux_in_upr,bndry_flux_out_upr)
    idxdown = _lmul_out_upr_bndry_generic!(v,u,mesh,model,Kp,n₋,n₊,bndry_flux_out_upr)
    idxup = n₋ .+ ((1:n_bases_per_cell(mesh)) .+ Kp*(findall(positive_phases(model)) .- 1)')[:] .+
        (n_bases_per_phase(mesh)*n_phases(model).-n_bases_per_cell(mesh)*n_phases(model))
    v[:,idxdown] += u[:,idxup]*LinearAlgebra.kron(
        # LinearAlgebra.diagm(0 => C[_has_right_boundary.(model.S)]),
        abs.(rates(model)[positive_phases(model)]).*model.P_upr[:,negative_phases(model)],
        bndry_flux_in_upr*bndry_flux_out_upr'/Δ(mesh,n_intervals(mesh)),
    )
    return nothing 
end
# function _lmul_ii_pos_diag_block!(v,u,i,mesh::Mesh,n₋,diag_block,Tᵢᵢ,up_block,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     v[:,block_idx] += (u[:,block_idx]*diag_block)/Δ(mesh,1) + u[:,block_idx]*Tᵢᵢ
#     for k in 2:n_intervals(mesh)
#         block_idx = block_idx .+ n_bases_per_cell_mesh

#         ℓ_idx = block_idx .- n_bases_per_cell_mesh
#         v[:,block_idx] += (u[:,ℓ_idx]*up_block)/Δ(mesh,k-1)

#         v[:,block_idx] += (u[:,block_idx]*diag_block)/Δ(mesh,k) + u[:,block_idx]*Tᵢᵢ
#     end
#     return nothing 
# end
# function _lmul_ii_neg_diag_block!(v,u,i,mesh::Mesh,n₋,diag_block,Tᵢᵢ,down_block,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     for k in 1:n_intervals(mesh)-1
#         ℓ_idx = block_idx .+ n_bases_per_cell_mesh
#         v[:,block_idx] += (u[:,ℓ_idx]*down_block)/Δ(mesh,k+1)

#         v[:,block_idx] += (u[:,block_idx]*diag_block)/Δ(mesh,k) + u[:,block_idx]*Tᵢᵢ

#         block_idx = block_idx .+ n_bases_per_cell_mesh
#     end
#     v[:,block_idx] += (u[:,block_idx]*diag_block)/Δ(mesh,n_intervals(mesh)) + u[:,block_idx]*Tᵢᵢ
#     return nothing 
# end
# function _lmul_ii_0_diag_block!(v,u,i,n_intervals_mesh,n₋,Tᵢᵢ,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     for k in 1:n_intervals_mesh
#         v[:,block_idx] += u[:,block_idx]*Tᵢᵢ
#         block_idx = block_idx .+ n_bases_per_cell_mesh
#     end
#     return nothing 
# end
# function _lmul_ii_pos_diag_block!(v,u,i,mesh::Mesh{StepRangeLen{Float64}},n₋,diag_block,Tᵢᵢ,up_block,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     diag_block /= Δ(mesh,1)
#     diag_block += Tᵢᵢ*LinearAlgebra.I(n_bases_per_cell_mesh)
#     up_block /= Δ(mesh,1)
#     v[:,block_idx] += u[:,block_idx]*diag_block
#     for k in 2:n_intervals(mesh)
#         block_idx = block_idx .+ n_bases_per_cell_mesh

#         ℓ_idx = block_idx .- n_bases_per_cell_mesh
#         v[:,block_idx] += u[:,ℓ_idx]*up_block

#         v[:,block_idx] += u[:,block_idx]*diag_block
#     end
#     return nothing 
# end
# function _lmul_ii_neg_diag_block!(v,u,i,mesh::Mesh{StepRangeLen{Float64}},n₋,diag_block,Tᵢᵢ,down_block,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     diag_block /= Δ(mesh,1)
#     diag_block += Tᵢᵢ*LinearAlgebra.I(n_bases_per_cell_mesh)
#     down_block /= Δ(mesh,1)
#     for k in 1:n_intervals(mesh)-1
#         ℓ_idx = block_idx .+ n_bases_per_cell_mesh
#         v[:,block_idx] += u[:,ℓ_idx]*down_block

#         v[:,block_idx] += u[:,block_idx]*diag_block

#         block_idx = block_idx .+ n_bases_per_cell_mesh
#     end
#     v[:,block_idx] += u[:,block_idx]*diag_block
#     return nothing 
# end
# function _lmul_ij_off_diag_blocks_FRAP!(v,u,i,j,S,Tᵢⱼ,D,Kp,n₋,n_intervals_mesh,n_bases_per_cell_mesh)
#     if membership(S,i)!=membership(S,j) 
#         i_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#         j_idx = (j-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#         # changes from S₊ to S₋ etc.
#         v[:,j_idx] += Tᵢⱼ*(u[:,i_idx]*D)
#         for k in 2:n_intervals_mesh
#             i_idx = i_idx .+ n_bases_per_cell_mesh
#             j_idx = j_idx .+ n_bases_per_cell_mesh
#             v[:,j_idx] += Tᵢⱼ*(u[:,i_idx]*D)
#         end
#     else
#         i_idx = (i-1)*Kp .+ (1:Kp) .+ n₋
#         j_idx = (j-1)*Kp .+ (1:Kp) .+ n₋
#         v[:,j_idx] += u[:,i_idx]*Tᵢⱼ
#     end
#     return nothing 
# end 


function fast_mul(B::LazyGenerator,u::AbstractMatrix{Float64})
    output_type = typeof(u)
    
    sz_u_1 = size(u,1)
    sz_u_2 = size(u,2)
    sz_B_1 = size(B,1)
    sz_B_2 = size(B,2)
    !(sz_B_2 == sz_u_1) && throw(DomainError("Dimension mismatch, u*B, length(u) must be size(B,1)"))

    if output_type <: SparseArrays.SparseMatrixCSC
        v = SparseArrays.spzeros(sz_B_1,sz_u_2)
    else 
        v = zeros(sz_B_1,sz_u_2)
    end
    model = B.dq.model
    mesh = B.dq.mesh
    Kp = n_bases_per_cell(B.dq) # K = n_intervals(mesh), p = n_bases_per_cell(mesh)
    C = rates(B.dq)
    n₋ = N₋(B.dq)
    n₊ = N₊(B.dq)

    # boundaries
    # at lower
    _rmul_at_lwr_bndry!(v,u,model.T,model.S,n₋) 
    # in to lower 
    _rmul_into_lwr_bndry!(v,u,model,mesh,Kp,n₋,C,B.boundary_flux.lower.in)
    
    # out of lower 
    _rmul_out_lwr_bndry!(v,u,mesh,model,Kp,n₋,B.boundary_flux.lower.in,B.boundary_flux.lower.out)
    
    # at upper
    _rmul_at_upr_bndry!(v,u,model.T,model.S,n₊) # v[:,end-n₊+1:end] += u[:,end-n₊+1:end]*model.T[_has_right_boundary.(model.S),_has_right_boundary.(model.S)]
    # in to upper
    _rmul_into_upr_bndry!(v,u,model,mesh,Kp,n₋,n₊,C,B.boundary_flux.upper.in)
    
    # out of upper 
    _rmul_out_upr_bndry!(v,u,mesh,model,Kp,n₋,n₊,B.boundary_flux.upper.in,B.boundary_flux.upper.out)
    
    # innards
    _rmul_innards!(v,u,B.dq.model,B.dq.mesh,B.blocks,B.D,C,n₋)
    return v
end
function _rmul_innards_generic!(v,u,model,mesh,tmp_blocks,D,C,n₋,k)
    for j in 1:n_phases(model)
        i_idx = (k-1)*n_phases(model)*n_bases_per_cell(mesh) + n₋ .+ (1:n_bases_per_cell(mesh)) 
        j_idx = i_idx .+ (j-1)*n_bases_per_cell(mesh) 
        (k>1) && ((C[j]>0.0) && (v[j_idx.-n_bases_per_cell(mesh)*n_phases(model),:] += C[j]*tmp_blocks[4]*u[j_idx,:]))
        (k<n_intervals(mesh)) && ((C[j]<0.0) && (v[j_idx.+n_bases_per_cell(mesh)*n_phases(model),:] += abs(C[j])*tmp_blocks[1]*u[j_idx,:]))
        for i in 1:n_phases(model)
            if (typeof(mesh)<:FRAPMesh)&&(membership(model.S,i)!=membership(model.S,j)) # + to - change
                v[i_idx,:] += D*u[j_idx,:]*model.T[i,j]
            elseif (i!=j)||(C[i]==0.0)
                v[i_idx,:] += model.T[i,j]*u[j_idx,:]
            elseif C[i]>0.0 # i==j
                v[i_idx,:] += C[i]*tmp_blocks[2]*u[j_idx,:] + model.T[i,j]*u[j_idx,:]
            elseif C[i]<0.0 # i==j
                v[i_idx,:] += abs(C[i])*tmp_blocks[3]*u[j_idx,:] + model.T[i,j]*u[j_idx,:]
            end
            i_idx = i_idx .+ n_bases_per_cell(mesh)
        end
    end
end
function _rmul_innards!(v,u,model,mesh::Mesh{Vector{Float64}},blocks,D,C,n₋)
    for k in 1:n_intervals(mesh)
        if 1<k<n_intervals(mesh) 
            tmp_blocks = (blocks[1]/Δ(mesh,k+1), blocks[2]/Δ(mesh,k), blocks[3]/Δ(mesh,k), blocks[4]/Δ(mesh,k-1))
        elseif k==1
            tmp_blocks = (blocks[1]/Δ(mesh,k+1), blocks[2]/Δ(mesh,k), blocks[3]/Δ(mesh,k), zeros(0,0))
        else
            tmp_blocks = (zeros(0,0), blocks[2]/Δ(mesh,k), blocks[3]/Δ(mesh,k), blocks[4]/Δ(mesh,k-1))
        end
        _lmul_innards_generic!(v,u,model,mesh,tmp_blocks,D,C,n₋,k)
    end
end
function _rmul_innards!(v,u,model,mesh::Mesh{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}},blocks,D,C,n₋)
    tmp_blocks = (blocks[1]/Δ(mesh,1), blocks[2]/Δ(mesh,1), blocks[3]/Δ(mesh,1), blocks[4]/Δ(mesh,1))
    for k in 1:n_intervals(mesh)
        _rmul_innards_generic!(v,u,model,mesh,tmp_blocks,D,C,n₋,k)
    end
end


# function _fast_mul(B::LazyGenerator,u::AbstractMatrix{Float64})
#     output_type = typeof(u)
    
#     sz_u_1 = size(u,1)
#     sz_u_2 = size(u,2)
#     sz_B_1 = size(B,1)
#     sz_B_2 = size(B,2)
#     !(sz_B_2 == sz_u_1) && throw(DomainError("Dimension mismatch, u*B, length(u) must be size(B,1)"))

#     if output_type <: SparseArrays.SparseMatrixCSC
#         v = SparseArrays.spzeros(sz_B_1,sz_u_2)
#     else 
#         v = zeros(sz_B_1,sz_u_2)
#     end
#     model = B.dq.model
#     mesh = B.dq.mesh
#     Kp = n_bases_per_phase(B.dq) # K = n_intervals(mesh), p = n_bases_per_cell(mesh)
#     C = rates(B.dq)
#     n₋ = N₋(B.dq)
#     n₊ = N₊(B.dq)

#     # boundaries
#     # at lower
#     _rmul_at_lwr_bndry!(v,u,model.T,model.S,n₋) 
#     # in to lower 
#     _rmul_into_lwr_bndry!(v,u,model,mesh,Kp,n₋,C,B.boundary_flux.lower.in)
    
#     # out of lower 
#     _rmul_out_lwr_bndry!(v,u,mesh,model,Kp,n₋,B.boundary_flux.lower.in,B.boundary_flux.lower.out)
    
#     # at upper
#     _rmul_at_upr_bndry!(v,u,model.T,model.S,n₊) # v[:,end-n₊+1:end] += u[:,end-n₊+1:end]*model.T[_has_right_boundary.(model.S),_has_right_boundary.(model.S)]
#     # in to upper
#     _rmul_into_upr_bndry!(v,u,model,mesh,Kp,n₋,n₊,C,B.boundary_flux.upper.in)
    
#     # out of upper 
#     _rmul_out_upr_bndry!(v,u,mesh,model,Kp,n₋,n₊,B.boundary_flux.upper.in,B.boundary_flux.upper.out)
    
#     # innards
#     for i in phases(model), j in phases(model)
#         if i == j 
#             # mult on diagonal
#             if C[i]>0.0 
#                 _rmul_ii_pos_diag_block!(v,u,i,mesh,n₋,
#                     C[i]*B.blocks[2], model.T[i,i],
#                     C[i]*B.blocks[4],
#                     n_bases_per_cell(mesh),Kp)
#             elseif C[i]<0.0
#                 _rmul_ii_neg_diag_block!(v,u,i,mesh,n₋,
#                     abs(C[i])*B.blocks[3], model.T[i,i],
#                     abs(C[i])*B.blocks[1],
#                     n_bases_per_cell(mesh),Kp)
#             else
#                 _rmul_ii_0_diag_block!(v,u,i,n_intervals(mesh),n₋,model.T[i,i],n_bases_per_cell(mesh),Kp)
#             end
#         elseif typeof(mesh)<:FRAPMesh
#             _rmul_ij_off_diag_blocks_FRAP!(v,u,i,j,model.S,model.T[i,j],B.D,Kp,n₋,n_intervals(mesh),n_bases_per_cell(mesh))
#         else
#             i_idx = (i-1)*Kp .+ (1:Kp) .+ n₋
#             j_idx = (j-1)*Kp .+ (1:Kp) .+ n₋
#             v[i_idx,:] += model.T[i,j]*u[j_idx,:]
#         end
#     end
#     return v
# end

function _rmul_at_lwr_bndry!(v,u,T,S,n₋)
    v[1:n₋,:] += T[_has_left_boundary.(S),_has_left_boundary.(S)]*u[1:n₋,:]
    return nothing 
end
function _rmul_at_upr_bndry!(v,u,T,S,n₊)
    v[end-n₊+1:end,:] += T[_has_right_boundary.(S),_has_right_boundary.(S)]*u[end-n₊+1:end,:]
    return nothing 
end
function _rmul_into_lwr_bndry!(v,u,model::BoundedFluidQueue,mesh,Kp,n₋,C,bndry_flux_in_lwr)
    idxdown = n₋ .+ ((1:n_bases_per_cell(mesh)).+Kp*(findall(negative_phases(model)) .- 1)')[:]
    v[idxdown,:] += LinearAlgebra.kron(
        abs.(C[negative_phases(model)]).*model.P_lwr[:,_has_left_boundary.(model.S)],
        bndry_flux_in_lwr/Δ(mesh,1),
    )*u[1:n₋,:]
    return nothing
end
function _rmul_into_upr_bndry!(v,u,model::BoundedFluidQueue,mesh,Kp,n₋,n₊,C,bndry_flux_in_upr)
    idxup = n₋ .+ ((1:n_bases_per_cell(mesh)).+Kp*(findall(positive_phases(model)) .- 1)')[:] .+
        (n_bases_per_phase(mesh)*n_phases(model).-n_bases_per_cell(mesh)*n_phases(model))
    v[idxup,:] += LinearAlgebra.kron(
        C[positive_phases(model)].*model.P_upr[:,_has_right_boundary.(model.S)],
        bndry_flux_in_upr/Δ(mesh,n_intervals(mesh)),
    )*u[end-n₊+1:end,:]
    return nothing
end
function _rmul_out_lwr_bndry_generic!(v,u,mesh,model,Kp,n₋,bndry_flux_out_lwr)
    idxup = n₋ .+ (Kp*(findall(positive_phases(model)).-1)' .+ (1:n_bases_per_cell(mesh)))[:]
    v[1:n₋,:] += LinearAlgebra.kron(model.T[_has_left_boundary.(model.S),positive_phases(model)],bndry_flux_out_lwr')*u[idxup,:]
    return idxup 
end
function _rmul_out_lwr_bndry!(v,u,mesh,model::BoundedFluidQueue,Kp,n₋,bndry_flux_in_lwr,bndry_flux_out_lwr)
    idxup = _rmul_out_lwr_bndry_generic!(v,u,mesh,model,Kp,n₋,bndry_flux_out_lwr)
    idxdown = n₋ .+ ((1:n_bases_per_cell(mesh)).+Kp*(findall(negative_phases(model)) .- 1)')[:]
    v[idxdown,:] += LinearAlgebra.kron(
        # LinearAlgebra.diagm(0 => abs.(C[_has_left_boundary.(model.S)])),
        abs.(rates(model)[negative_phases(model)]).*model.P_lwr[:,positive_phases(model)],
        bndry_flux_in_lwr*bndry_flux_out_lwr'/Δ(mesh,1),
    )*u[idxup,:]
    return nothing 
end
function _rmul_out_upr_bndry_generic!(v,u,mesh,model,Kp,n₋,n₊,bndry_flux_out_upr)
    idxdown = n₋ .+ (Kp*(findall(negative_phases(model)).-1)' .+ (1:n_bases_per_cell(mesh)))[:] .+
        (n_bases_per_phase(mesh)*n_phases(model).-n_bases_per_cell(mesh)*n_phases(model))
    v[end-n₊+1:end,:] += LinearAlgebra.kron(model.T[_has_right_boundary.(model.S),negative_phases(model)],bndry_flux_out_upr')*u[idxdown,:]
    return idxdown 
end
function _rmul_out_upr_bndry!(v,u,mesh,model::BoundedFluidQueue,Kp,n₋,n₊,bndry_flux_in_upr,bndry_flux_out_upr)
    idxdown = _rmul_out_upr_bndry_generic!(v,u,mesh,model,Kp,n₋,n₊,bndry_flux_out_upr)
    idxup = n₋ .+ ((1:n_bases_per_cell(mesh)) .+ Kp*(findall(positive_phases(model)) .- 1)')[:] .+
        (n_bases_per_phase(mesh)*n_phases(model).-n_bases_per_cell(mesh)*n_phases(model))
    v[idxup,:] += LinearAlgebra.kron(
        # LinearAlgebra.diagm(0 => C[_has_right_boundary.(model.S)]),
        abs.(rates(model)[positive_phases(model)]).*model.P_upr[:,negative_phases(model)],
        bndry_flux_in_upr*bndry_flux_out_upr'/Δ(mesh,n_intervals(mesh)),
    )*u[idxdown,:]
    return nothing 
end
# function _rmul_ii_pos_diag_block!(v,u,i,mesh::Mesh,n₋,diag_block,Tᵢᵢ,up_block,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     v[block_idx,:] += (diag_block*u[block_idx,:])/Δ(mesh,1) + u[block_idx,:]*Tᵢᵢ
#     for k in 2:n_intervals(mesh)
#         block_idx = block_idx .+ n_bases_per_cell_mesh

#         ℓ_idx = block_idx .- n_bases_per_cell_mesh
#         v[ℓ_idx,:] += (up_block*u[block_idx,:])/Δ(mesh,k-1)

#         v[block_idx,:] += (diag_block*u[block_idx,:])/Δ(mesh,k) + u[block_idx,:]*Tᵢᵢ
#     end
#     return nothing 
# end
# function _rmul_ii_neg_diag_block!(v,u,i,mesh::Mesh,n₋,diag_block,Tᵢᵢ,down_block,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     for k in 1:n_intervals(mesh)-1
#         ℓ_idx = block_idx .+ n_bases_per_cell_mesh
#         v[ℓ_idx,:] += (down_block*u[block_idx,:])/Δ(mesh,k+1)

#         v[block_idx,:] += (diag_block*u[block_idx,:])/Δ(mesh,k) + u[block_idx,:]*Tᵢᵢ

#         block_idx = block_idx .+ n_bases_per_cell_mesh
#     end
#     v[block_idx,:] += (diag_block*u[block_idx,:])/Δ(mesh,n_intervals(mesh)) + u[block_idx,:]*Tᵢᵢ
#     return nothing 
# end
# function _rmul_ii_0_diag_block!(v,u,i,n_intervals_mesh,n₋,Tᵢᵢ,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     for k in 1:n_intervals_mesh
#         v[block_idx,:] += u[block_idx,:]*Tᵢᵢ
#         block_idx = block_idx .+ n_bases_per_cell_mesh
#     end
#     return nothing 
# end
# function _rmul_ii_pos_diag_block!(v,u,i,mesh::Mesh{StepRangeLen{Float64}},n₋,diag_block,Tᵢᵢ,up_block,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     diag_block /= Δ(mesh,1)
#     diag_block += Tᵢᵢ*LinearAlgebra.I(n_bases_per_cell_mesh)
#     up_block /= Δ(mesh,1)
#     v[block_idx,:] += diag_block*u[block_idx,:]
#     for k in 2:n_intervals(mesh)
#         block_idx = block_idx .+ n_bases_per_cell_mesh

#         ℓ_idx = block_idx .- n_bases_per_cell_mesh
#         v[ℓ_idx,:] += up_block*u[block_idx,:]

#         v[block_idx,:] += diag_block*u[block_idx,:]
#     end
#     return nothing 
# end
# function _rmul_ii_neg_diag_block!(v,u,i,mesh::Mesh{StepRangeLen{Float64}},n₋,diag_block,Tᵢᵢ,down_block,n_bases_per_cell_mesh,Kp)
#     block_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#     diag_block /= Δ(mesh,1)
#     diag_block += Tᵢᵢ*LinearAlgebra.I(n_bases_per_cell_mesh)
#     down_block /= Δ(mesh,1)
#     for k in 1:n_intervals(mesh)-1
#         ℓ_idx = block_idx .+ n_bases_per_cell_mesh
#         v[ℓ_idx,:] += down_block*u[block_idx,:]

#         v[block_idx,:] += diag_block*u[block_idx,:]

#         block_idx = block_idx .+ n_bases_per_cell_mesh
#     end
#     v[block_idx,:] += diag_block*u[block_idx,:]
#     return nothing 
# end
# function _rmul_ij_off_diag_blocks_FRAP!(v,u,i,j,S,Tᵢⱼ,D,Kp,n₋,n_intervals_mesh,n_bases_per_cell_mesh)
#     if membership(S,i)!=membership(S,j)# B.pmidx[i,j]
#         i_idx = (i-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#         j_idx = (j-1)*Kp .+ (1:n_bases_per_cell_mesh) .+ n₋
#         # changes from S₊ to S₋ etc.
#         v[i_idx,:] += Tᵢⱼ*(D*u[j_idx,:])
#         for k in 2:n_intervals_mesh
#             i_idx = i_idx .+ n_bases_per_cell_mesh
#             j_idx = j_idx .+ n_bases_per_cell_mesh
#             v[i_idx,:] += Tᵢⱼ*(D*u[j_idx,:])
#         end
#     else
#         i_idx = (i-1)*Kp .+ (1:Kp) .+ n₋
#         j_idx = (j-1)*Kp .+ (1:Kp) .+ n₋
#         v[i_idx,:] += u[j_idx,:]*Tᵢⱼ
#     end
#     return nothing 
# end 

fast_mul(A::LazyGenerator, B::LazyGenerator) = fast_mul(build_full_generator(A).B,B)
function fast_mul(A::LazyGenerator,x::Real) 
    blocks = (x*A.blocks[i] for i in 1:4)
    boundary_flux = BoundaryFlux(
        OneBoundaryFlux(A.boundary_flux.upper.in*x,A.boundary_flux.upper.out*x),# upper 
        OneBoundaryFlux(A.boundary_flux.lower.in*x,A.boundary_flux.lower.out*x) # lower
    )
    D = x*A.D
    return LazyGenerator(A.dq,blocks,boundary_flux,D)
end
fast_mul(x::Real,A::LazyGenerator) = fast_mul(A,x)

# for f in (:+,:-), t in (Matrix{Float64},SparseArrays.SparseMatrixCSC{Float64,Int})
#     @eval $f(B::LazyGenerator,A::$t) = [$f(B[i,j],A[i,j]) for i in 1:size(B,1), j in 1:size(B,2)]
#     @eval $f(A::$t,B::LazyGenerator) = $f(B,A)
# end

function show(io::IO, mime::MIME"text/plain", B::LazyGenerator)
    if VERSION >= v"1.6"
        show(io, mime, fast_mul(SparseArrays.SparseMatrixCSC{Float64,Int}(LinearAlgebra.I(size(B,1))),B))
    else
        show(io, mime, fast_mul(Matrix{Float64}(LinearAlgebra.I(size(B,1))),B))
    end
end
# show(B::LazyGenerator) = show(stdout, B)

function getindex_interior(B::LazyGenerator,row::Int,col::Int)
    
    if abs(row-col) > n_bases_per_level(B.dq) + n_bases_per_cell(B.dq)
        return 0.0
    else 
        k, i, p = _map_from_index_interior(row,B)
        l, j, q = _map_from_index_interior(col,B)
        C = rates(B.dq.model)
        if abs(k-l)>1 # cant move more than one level, skip free
            return 0.0
        elseif (k!=l)&&((i!=j)||(C[i]==0.0)) # can't move a level unless i=j
            return 0.0
        elseif (k==l+1)&&(C[i]<0.0) # lower diag block
            return abs(C[i])*B.blocks[1][p,q]/Δ(B.dq,k)
        elseif (k==l-1)&&(C[i]>0.0) # upper diag block
            return C[i]*B.blocks[4][p,q]/Δ(B.dq,k)
        elseif (k==l) # on diagonal
            if (i==j)
                if (C[i]<0.0) 
                    (p==q) ? (return abs(C[i])*B.blocks[3][p,q]/Δ(B.dq,k)+B.dq.model.T[i,i]) : (return abs(C[i])*B.blocks[3][p,q]/Δ(B.dq,k))
                elseif (C[i]>0.0)
                    (p==q) ? (return C[i]*B.blocks[2][p,q]/Δ(B.dq,k)+B.dq.model.T[i,i]) : (return C[i]*B.blocks[2][p,q]/Δ(B.dq,k))
                elseif (p==q)
                    return B.dq.model.T[i,i]
                else 
                    return 0.0
                end
            elseif (l==1)&&(k==1)&&(C[i]<0.0)&&(C[j]>0.0) # lwr boundary condition
                i_idx = i - sum(rates(B.dq.model.S[1:i-1]).>=0.0)
                return abs(C[i])*B.dq.model.P_lwr[i_idx,j]*B.boundary_flux.lower.in[p]*B.boundary_flux.lower.out[q]/Δ(B.dq,k) + B.dq.model.T[i,j]*B.D[p,q]
            elseif (l==n_intervals(B.dq))&&(k==n_intervals(B.dq))&&(C[i]>0.0)&&(C[j]<0.0) # upr boundary condition
                i_idx = i - sum(rates(B.dq.model.S[1:i-1]).<=0.0)
                return C[i]*B.dq.model.P_upr[i_idx,j]*B.boundary_flux.upper.in[p]*B.boundary_flux.upper.out[q]/Δ(B.dq,k) + B.dq.model.T[i,j]*B.D[p,q]
            elseif (typeof(B.dq.mesh)<:FRAPMesh)&&(membership(B.dq.model.S,i)!=membership(B.dq.model.S,j)) # + to - change
                return B.dq.model.T[i,j]*B.D[p,q]
            elseif (p==q)
                return B.dq.model.T[i,j]
            else 
                return 0.0
            end     
        else 
            return 0.0  
        end
    end
end

# function _getindex_interior(B::LazyGenerator,row::Int,col::Int)
#     i, k, p = _map_from_index_interior(row,B)
#     j, l, q = _map_from_index_interior(col,B)
    
#     model = B.dq.model
#     C = rates(model)

#     v=0.0
#     if i==j
#         if k==l
#             v=abs(C[i])*B.blocks[2 + (C[i].<0)][p,q]/Δ(B.dq,k) + model.T[i,j]*(p==q)
#         elseif k+1==l# upper diagonal blocks
#             (C[i]>0) && (v=C[i]*B.blocks[4][p,q]/Δ(B.dq,k))
#         elseif k-1==l
#             (C[i]<0) && (v=abs(C[i])*B.blocks[1][p,q]/Δ(B.dq,k))
#         end
#     elseif membership(model.S,i)!=membership(model.S,j)
#         (k==l) && (v=model.T[i,j]*B.D[p,q])
#     else
#         ((p==q)&&(k==l)) && (v=model.T[i,j])
#     end

#     if (C[i]<0.0)&&(C[j]>0.0)&&(l==1)&&(k==1)
#         i_idx = i - sum(rates(B.dq.model.S[1:i-1]).>=0.0)
#         v+=abs(C[i])*B.dq.model.P_lwr[i_idx,j]*B.boundary_flux.lower.in[p]*B.boundary_flux.lower.out[q]/Δ(B.dq,k)
#     elseif (C[i]>0.0)&&(C[j]<0.0)&&(l==n_intervals(B.dq))&&(k==n_intervals(B.dq))
#         i_idx = i - sum(rates(B.dq.model.S[1:i-1]).<=0.0)
#         v+=abs(C[i])*B.dq.model.P_upr[i_idx,j]*B.boundary_flux.upper.in[p]*B.boundary_flux.upper.out[q]/Δ(B.dq,k)
#     end
#     return v
# end

function getindex_out_boundary(B::LazyGenerator,row::Int,col::Int)
    (!_is_boundary_index(row,B))&&throw(DomainError(row,"row index does not correspond to a boundary"))
    l, j, q = _map_from_index_interior(col,B)
    
    model = B.dq.model
    C = rates(model)

    i = _map_from_index_boundary(row,B)
    if (l==1)&&(C[j]>0.0)&&_has_left_boundary(model.S,i)&&(row∈(1:N₊(B.dq)))
        v = model.T[i,j]*B.boundary_flux.lower.out[q]
    elseif (l==n_intervals(B.dq))&&(C[j]<0.0)&&_has_right_boundary(model.S,i)&&(row∈(size(B,2).-(N₋(B.dq)-1:-1:0)))
        v = model.T[i,j]*B.boundary_flux.upper.out[q]
    else 
        v = 0.0
    end
    
    return v
end
# function _getindex_in_boundary(B::LazyGenerator,row::Int,col::Int)
#     (!_is_boundary_index(col,B))&&throw(DomainError(col,"col index does not correspond to a boundary"))
#     k, i, p = _map_from_index_interior(row,B)
    
#     C = rates(B.dq)
    
#     j = _map_from_index_boundary(col,B)
#     if (k==1)&&(C[i]<=0.0)&&(i==j)
#         v = abs(C[i])*B.boundary_flux.lower.in[p]/Δ(B.dq,1)
#     elseif (k==n_intervals(B.dq))&&(C[i]>0.0)&&(i==j)
#         v = abs(C[i])*B.boundary_flux.upper.in[p]/Δ(B.dq,n_intervals(B.dq))
#     else 
#         v = 0.0
#     end
    
#     return v
# end
function getindex_in_boundary(B::LazyGenerator,row::Int,col::Int)
    (!_is_boundary_index(col,B))&&throw(DomainError(col,"col index does not correspond to a boundary"))
    k, i, p = _map_from_index_interior(row,B)
    
    C = rates(B.dq)
    
    j = _map_from_index_boundary(col,B)
    if (k==1)&&(C[i]<0.0)&&(C[j]<=0.0)&&(col∈(1:N₋(B.dq)))
        i_idx = i - sum(rates(B.dq.model.S[1:i-1]).>=0.0)
        v = (B.dq.model.P_lwr[i_idx,j])*abs(C[i])*B.boundary_flux.lower.in[p]/Δ(B.dq,1)
    elseif (k==n_intervals(B.dq))&&(C[i]>0.0)&&(C[j]>=0.0)&&(col∈(size(B,2).-(N₊(B.dq)-1:-1:0)))
        i_idx = i - sum(rates(B.dq.model.S[1:i-1]).<=0.0)
        v = (B.dq.model.P_upr[i_idx,j])*abs(C[i])*B.boundary_flux.upper.in[p]/Δ(B.dq,k)
    else 
        v = 0.0
    end
    
    return v
end

function getindex(B::LazyGenerator,row::Int,col::Int)
    sz = size(B)
    !((0<row<=sz[1])&&(0<col<=sz[2]))&&throw(BoundsError(B,(row,col)))
    
    if _is_boundary_index(row,B) && _is_boundary_index(col,B)
        i = _map_from_index_boundary(row,B)
        j = _map_from_index_boundary(col,B)
        if (row∈(1:N₋(B.dq)))&&(col∈(1:N₋(B.dq)))
            v = B.dq.model.T[i,j]
        elseif (row∈(size(B,1).-(N₊(B.dq)-1:-1:0)))&&(col∈(size(B,2).-(N₊(B.dq)-1:-1:0)))
            v = B.dq.model.T[i,j]
        else 
            v = 0.0
        end
    elseif _is_boundary_index(col,B)
        v = getindex_in_boundary(B,row,col)
    elseif _is_boundary_index(row,B)
        v = getindex_out_boundary(B,row,col)
    else
        v = getindex_interior(B,row,col)
    end
    return v
end

function getindex(B::LazyGenerator,i::Int) 
    !(0<i<=length(B))&&throw(BoundsError(B,i))
    sz = size(B)
    col = (i-1)÷sz[1] 
    row = i-col*sz[1]
    col += 1
    return B[row,col]
end
