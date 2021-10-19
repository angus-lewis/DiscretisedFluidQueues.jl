"""
    Generator

Abstract type representing a discretised infinitesimal generator of a FLuidQueue. 
Behaves much like a square matrix. 
"""
abstract type Generator <: AbstractMatrix{Float64} end 
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
    LazyGenerator <: Generator

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
struct LazyGenerator  <: Generator
    dq::DiscretisedFluidQueue
    blocks::NTuple{4,AbstractMatrix{Float64}}
    boundary_flux::BoundaryFlux
    D::AbstractMatrix{Float64}
    function LazyGenerator(
        dq::DiscretisedFluidQueue,
        blocks::NTuple{4,AbstractMatrix{Float64}},
        boundary_flux::BoundaryFlux,
        D::AbstractMatrix{Float64},
    )
        s = size(blocks[1])
        for b in 1:4
            checksquare(blocks[b]) 
            !(s == size(blocks[b])) && throw(DomainError("blocks must be the same size"))
        end
        checksquare(D)
        !(s == size(D)) && throw(DomainError("blocks must be the same size as D"))
        
        return new(dq,blocks,boundary_flux,D)
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

function _map_to_index_interior(i::Int,k::Int,p::Int,dq::DiscretisedFluidQueue)
    # i phase
    # k cell
    # p basis
    
    _check_phase_index(i,dq.model)
    _check_mesh_index(k,dq.mesh)
    _check_basis_index(p,dq.mesh)

    P = n_bases_per_cell(dq)
    KP = n_bases_per_phase(dq)

    idx = (i-1)*KP + (k-1)*P + p
    return N₋(dq) + idx 
end
function _map_to_index_boundary(i::Int,dq::DiscretisedFluidQueue)
    # i phase
    _check_phase_index(i,dq.model)
    if _has_left_boundary(dq.model.S,i) 
        idx = N₋(dq.model.S[1:i])
    else _has_right_boundary(dq.model.S,i)
        N = n_phases(dq)
        KP = n_bases_per_phase(dq)
        idx = N₊(dq.model.S[1:i]) + KP*N + N₋(dq)
    end
    return idx
end

_is_boundary_index(n::Int,B::LazyGenerator) = (n∈1:N₋(B.dq))||(n∈(size(B,1).-(0:N₊(B.dq)-1)))
function _map_from_index_interior(n::Int,B::LazyGenerator)
    # n matrix index to map to phase, cell, basis
    (!(1<=n<=size(B,1))||_is_boundary_index(n,B))&&throw(DomainError(n,"not a valid interior index"))
    
    n -= (N₋(B.dq)+1)
    N = n_phases(B.dq)
    KP = n_bases_per_phase(B.dq)
    P = n_bases_per_cell(B.dq)

    i = (n÷KP) + 1
    k = mod(n,KP)÷P + 1 #(n-1 - (i-1)*KP)÷P + 1
    p = mod(n,P) + 1
    return i, k, p
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
    model = B.dq.model
    mesh = B.dq.mesh
    Kp = n_bases_per_phase(B.dq) # K = n_intervals(mesh), p = n_bases_per_cell(mesh)
    C = rates(B.dq)
    n₋ = N₋(B.dq)
    n₊ = N₊(B.dq)

    # boundaries
    # at lower
    v[:,1:n₋] += u[:,1:n₋]*model.T[_has_left_boundary.(model.S),_has_left_boundary.(model.S)]
    # in to lower 
    idxdown = n₋ .+ ((1:n_bases_per_cell(mesh)).+Kp*(findall(_has_left_boundary.(model.S)) .- 1)')[:]
    v[:,1:n₋] += u[:,idxdown]*LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => abs.(C[_has_left_boundary.(model.S)])),
        B.boundary_flux.lower.in/Δ(mesh,1),
    )
    # out of lower 
    idxup = n₋ .+ (Kp*(findall(C .> 0).-1)' .+ (1:n_bases_per_cell(mesh)))[:]
    v[:,idxup] += u[:,1:n₋]*kron(model.T[_has_left_boundary.(model.S),C.>0],B.boundary_flux.lower.out')

    # at upper
    v[:,end-n₊+1:end] += u[:,end-n₊+1:end]*model.T[_has_right_boundary.(model.S),_has_right_boundary.(model.S)]
    # in to upper
    idxup = n₋ .+ ((1:n_bases_per_cell(mesh)) .+ Kp*(findall(_has_right_boundary.(model.S)) .- 1)')[:] .+
        (Kp - n_bases_per_cell(mesh))
    v[:,end-n₊+1:end] += u[:,idxup]*LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => C[_has_right_boundary.(model.S)]),
        B.boundary_flux.upper.in/Δ(mesh,n_intervals(mesh)),
    )
    # out of upper 
    idxdown = n₋ .+ (Kp*(findall(C .< 0).-1)' .+ (1:n_bases_per_cell(mesh)))[:] .+
        (Kp - n_bases_per_cell(mesh))
    v[:,idxdown] += u[:,end-n₊+1:end]*kron(model.T[_has_right_boundary.(model.S),C.<0],B.boundary_flux.upper.out')

    # innards
    for i in phases(model), j in phases(model)
        if i == j 
            # mult on diagonal
            for k in 1:n_intervals(mesh)
                k_idx = (i-1)*Kp .+ (k-1)*n_bases_per_cell(mesh) .+ (1:n_bases_per_cell(mesh)) .+ n₋
                for ℓ in 1:n_intervals(mesh)
                    if (k == ℓ+1) && (C[i] > 0)
                        ℓ_idx = k_idx .- n_bases_per_cell(mesh) 
                        v[:,k_idx] += C[i]*(u[:,ℓ_idx]*B.blocks[4])/Δ(mesh,ℓ)
                    elseif k == ℓ
                        v[:,k_idx] += (u[:,k_idx]*(abs(C[i])*B.blocks[2 + (C[i].<0)]/Δ(mesh,ℓ) + model.T[i,j]*LinearAlgebra.I))
                    elseif (k == ℓ-1) && (C[i] < 0)
                        ℓ_idx = k_idx .+ n_bases_per_cell(mesh) 
                        v[:,k_idx] += abs(C[i])*(u[:,ℓ_idx]*B.blocks[1])/Δ(mesh,ℓ)
                    end
                end
            end
        elseif membership(model.S,i)!=membership(model.S,j)# B.pmidx[i,j]
            # changes from S₊ to S₋ etc.
            for k in 1:n_intervals(mesh)
                for ℓ in 1:n_intervals(mesh)
                    if k == ℓ
                        i_idx = (i-1)*Kp .+ (k-1)*n_bases_per_cell(mesh) .+ (1:n_bases_per_cell(mesh)) .+ n₋
                        j_idx = (j-1)*Kp .+ (k-1)*n_bases_per_cell(mesh) .+ (1:n_bases_per_cell(mesh)) .+ n₋
                        v[:,j_idx] += (u[:,i_idx]*(model.T[i,j]*B.D))
                    end
                end
            end
        else
            i_idx = (i-1)*Kp .+ (1:Kp) .+ n₋
            j_idx = (j-1)*Kp .+ (1:Kp) .+ n₋
            v[:,j_idx] += (u[:,i_idx]*model.T[i,j])
        end
    end
    return v
end

function fast_mul(B::LazyGenerator, u::AbstractMatrix{Float64})
    output_type = typeof(u)

    sz_u_1 = size(u,1)
    sz_B_2 = size(B,2)

    !(sz_u_1 == sz_B_2) && throw(DomainError("Dimension mismatch, u*B, length(u) must be size(B,2)"))

    if output_type <: SparseArrays.SparseMatrixCSC
        v = SparseArrays.spzeros(sz_u_1,sz_B_2)
    else 
        v = zeros(sz_u_1,sz_B_2)
    end

    model = B.dq.model

    Kp = n_bases_per_phase(B.dq) # K = n_intervals, p = n_bases

    C = rates(model)
    n₋ = N₋(B.dq)
    n₊ = N₊(B.dq)
    # boundaries
    # at lower
    v[1:n₋,:] += model.T[_has_left_boundary.(model.S),_has_left_boundary.(model.S)]*u[1:n₋,:]
    # in to lower 
    idxdown = n₋ .+ ((1:n_bases_per_cell(B.dq)).+Kp*(findall(_has_left_boundary.(model.S)) .- 1)')[:]
    v[idxdown,:] += LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => abs.(C[_has_left_boundary.(model.S)])),
        B.boundary_flux.lower.in/Δ(B.dq,1),
    )*u[1:n₋,:]
    # out of lower 
    idxup = n₋ .+ (Kp*(findall(C .> 0).-1)' .+ (1:n_bases_per_cell(B.dq)))[:]
    v[1:n₋,:] += kron(model.T[_has_left_boundary.(model.S),C.>0],B.boundary_flux.lower.out')*u[idxup,:]

    # at upper
    v[end-n₊+1:end,:] += model.T[_has_right_boundary.(model.S),_has_right_boundary.(model.S)]*u[end-n₊+1:end,:]
    # in to upper
    idxup = n₋ .+ ((1:n_bases_per_cell(B.dq)).+Kp*(findall(_has_right_boundary.(model.S)) .- 1)')[:] .+
        (Kp - n_bases_per_cell(B.dq))
    v[idxup,:] += LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => C[_has_right_boundary.(model.S)]),
        B.boundary_flux.upper.in/Δ(B.dq,n_intervals(B.dq)),
    )*u[end-n₊+1:end,:]
    # out of upper 
    idxdown = n₋ .+ (Kp*(findall(C .< 0).-1)' .+ (1:n_bases_per_cell(B.dq)))[:] .+
        (Kp - n_bases_per_cell(B.dq))
    v[end-n₊+1:end,:] += kron(model.T[_has_right_boundary.(model.S),C.<0],B.boundary_flux.upper.out')*u[idxdown,:]

    # innards
    for i in phases(model), j in phases(model)
        if i == j 
            # mult on diagonal
            for k in 1:n_intervals(B.dq)
                k_idx = (i-1)*Kp .+ (k-1)*n_bases_per_cell(B.dq) .+ (1:n_bases_per_cell(B.dq)) .+ n₋
                for ℓ in 1:n_intervals(B.dq)
                    if (k == ℓ+1) && (C[i] > 0) # upper diagonal block
                        ℓ_idx = k_idx .- n_bases_per_cell(B.dq) 
                        v[ℓ_idx,:] += C[i]*(B.blocks[4]*u[k_idx,:])/Δ(B.dq,ℓ)
                    elseif k == ℓ # diagonal 
                        v[k_idx,:] += ((abs(C[i])*B.blocks[2 + (C[i].<0)]/Δ(B.dq,ℓ) + model.T[i,j]*LinearAlgebra.I)*u[k_idx,:])
                    elseif (k == ℓ-1) && (C[i] < 0) # lower diagonal
                        ℓ_idx = k_idx .+ n_bases_per_cell(B.dq) 
                        v[ℓ_idx,:] += abs(C[i])*(B.blocks[1]*u[k_idx,:])/Δ(B.dq,ℓ)
                    end
                end
            end
        elseif membership(model.S,i)!=membership(model.S,j) # B.pmidx[i,j]
            # changes from S₊ to S₋ etc.
            for k in 1:n_intervals(B.dq)
                for ℓ in 1:n_intervals(B.dq)
                    if k == ℓ
                        i_idx = (i-1)*Kp .+ (k-1)*n_bases_per_cell(B.dq) .+ (1:n_bases_per_cell(B.dq)) .+ n₋
                        j_idx = (j-1)*Kp .+ (k-1)*n_bases_per_cell(B.dq) .+ (1:n_bases_per_cell(B.dq)) .+ n₋
                        v[i_idx,:] += (model.T[i,j]*B.D)*u[j_idx,:]
                    end
                end
            end
        else
            i_idx = (i-1)*Kp .+ (1:Kp) .+ n₋
            j_idx = (j-1)*Kp .+ (1:Kp) .+ n₋
            v[i_idx,:] += model.T[i,j]*u[j_idx,:]
        end
    end
    return v
end

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
    i, k, p = _map_from_index_interior(row,B)
    j, l, q = _map_from_index_interior(col,B)
    
    model = B.dq.model
    C = rates(model)

    v=0.0
    if i==j
        if k==l
            v=abs(C[i])*B.blocks[2 + (C[i].<0)][p,q]/Δ(B.dq,k) + model.T[i,j]*(p==q)
        elseif k+1==l# upper diagonal blocks
            (C[i]>0) && (v=C[i]*B.blocks[4][p,q]/Δ(B.dq,k))
        elseif k-1==l
            (C[i]<0) && (v=abs(C[i])*B.blocks[1][p,q]/Δ(B.dq,k))
        end
    elseif membership(model.S,i)!=membership(model.S,j)
        (k==l) && (v=model.T[i,j]*B.D[p,q])
    else
        ((p==q)&&(k==l)) && (v=model.T[i,j])
    end
    return v
end
function getindex_out_boundary(B::LazyGenerator,row::Int,col::Int)
    (!_is_boundary_index(row,B))&&throw(DomainError(row,"row index does not correspond to a boundary"))
    j, l, q = _map_from_index_interior(col,B)
    
    model = B.dq.model
    C = rates(model)

    i = _map_from_index_boundary(row,B)
    if (l==1)&&(C[j]>0)&&_has_left_boundary(model.S,i)
        v = model.T[i,j]*B.boundary_flux.lower.out[q]
    elseif (l==n_intervals(B.dq))&&(C[j]<0)&&_has_right_boundary(model.S,i)
        v = model.T[i,j]*B.boundary_flux.upper.out[q]
    else 
        v = 0.0
    end
    
    return v
end
function getindex_in_boundary(B::LazyGenerator,row::Int,col::Int)
    (!_is_boundary_index(col,B))&&throw(DomainError(col,"col index does not correspond to a boundary"))
    i, k, p = _map_from_index_interior(row,B)
    
    C = rates(B.dq)
    
    j = _map_from_index_boundary(col,B)
    if (k==1)&&(C[i]<0)&&(i==j)
        v = abs(C[i])*B.boundary_flux.lower.in[p]/Δ(B.dq,1)
    elseif (k==n_intervals(B.dq))&&(C[i]>0)&&(i==j)
        v = abs(C[i])*B.boundary_flux.upper.in[p]/Δ(B.dq,n_intervals(B.dq))
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
        (_has_left_boundary(B.dq.model.S,i)==_has_left_boundary(B.dq.model.S,j)) ? (v = B.dq.model.T[i,j]) : (v=0.0)
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

# function getindex(B::LazyGenerator,c::Colon) 
#     return [B[i] for i in 1:length(B)]
# end