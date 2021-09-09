abstract type Generator <: AbstractArray{Real,2} end 
const BoundaryFluxTupleType = NamedTuple{(:upper,:lower),Tuple{NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}},NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}}}}
struct LazyGenerator  <: Generator
    model::FluidQueue
    mesh::Mesh
    blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}}
    boundary_flux::BoundaryFluxTupleType
    D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}}
    # pmidx::Union{Array{Bool,2},BitArray{2}}
    # Fil::IndexDict
    function LazyGenerator(
        model::FluidQueue,
        mesh::Mesh,
        blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}},
        boundary_flux::NamedTuple{(:upper,:lower),Tuple{NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}},NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}}}},
        # T::Array{<:Real,2},
        # C::PhaseSet,
        # Δ::Array{<:Real,1},
        D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
        # pmidx::Union{Array{Bool,2},BitArray{2}},
        # Fil::IndexDict,
    )
        s = size(blocks[1])
        for b in 1:4
            checksquare(blocks[b]) 
            !(s == size(blocks[b])) && throw(DomainError("blocks must be the same size"))
        end
        # checksquare(T)
        checksquare(D)
        !(s == size(D)) && throw(DomainError("blocks must be the same size as D"))
        # checksquare(pmidx) 
        # !(size(T) == size(pmidx)) && throw(DomainError(pmidx, "must be the same size as T"))
        # !(length(C) == size(T,1)) && throw(DomainError(C, "must be the same length as T"))
        
        return new(model,mesh,blocks,boundary_flux,D)#,Fil)
    end
end
function LazyGenerator(
    model::Model,
    mesh::Mesh,
    blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}},
    boundary_flux::NamedTuple{(:in, :out),Tuple{Array{Float64,1},Array{Float64,1}}},
    # T::Array{<:Real,2},
    # C::PhaseSet,
    # Δ::Array{<:Real,1},
    D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
    # pmidx::Union{Array{Bool,2},BitArray{2}},
    # Fil::IndexDict,
)
    blocks = (blocks[1],blocks[2],blocks[2],blocks[3])
    boundary_flux = (upper = boundary_flux, lower = boundary_flux)
    return LazyGenerator(model,mesh,blocks,boundary_flux,D)
end
function LazyGenerator(
    blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}},
    boundary_flux::NamedTuple{(:in, :out),Tuple{Array{Float64,1},Array{Float64,1}}},
    # T::Array{<:Real,2},
    # C::Array{<:Real,1},
    # Δ::Array{<:Real,1},
    D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
    # pmidx::Union{Array{Bool,2},BitArray{2}},
    # Fil::IndexDict,
)
    blocks = (blocks[1],blocks[2],blocks[2],blocks[3])
    boundary_flux = (upper = boundary_flux, lower = boundary_flux)
    return LazyGenerator(blocks, boundary_flux, D,# Fil,
    )
end
function MakeLazyGenerator(model::Model, mesh::Mesh; v::Bool=false)
    throw(DomainError("Can construct LazyGenerator for DGMesh, FRAPMesh, only"))
end

function size(B::LazyGenerator)
    sz = n_phases(B.model)*total_n_bases(B.mesh) + N₋(B.model.S) + N₊(B.model.S)
    return (sz,sz)
end
size(B::LazyGenerator, n::Int) = size(B)[n]

function *(u::AbstractArray{<:Real,2}, B::LazyGenerator)
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
    
    model = B.model
    mesh = B.mesh

    Kp = total_n_bases(mesh) # K = n_intervals(mesh), p = n_bases(mesh)
    m = membership(model.S)
    C = rates(model)
    n₋ = N₋(model.S)
    n₊ = N₊(model.S)

    # boundaries
    # at lower
    v[:,1:n₋] += u[:,1:n₋]*model.T[m.<=0,m.<=0]
    # in to lower 
    idxdown = n₋ .+ ((1:n_bases(mesh)).+Kp*(findall(m .<= 0) .- 1)')[:]
    v[:,1:n₋] += u[:,idxdown]*LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => abs.(C[m.<=0])),
        B.boundary_flux.lower.in/Δ(mesh,1),
    )
    # out of lower 
    idxup = n₋ .+ (Kp*(findall(C .> 0).-1)' .+ (1:n_bases(mesh)))[:]
    v[:,idxup] += u[:,1:n₋]*kron(model.T[m.<=0,C.>0],B.boundary_flux.lower.out')

    # at upper
    v[:,end-n₊+1:end] += u[:,end-n₊+1:end]*model.T[m.>=0,m.>=0]
    # in to upper
    idxup = n₋ .+ ((1:n_bases(mesh)) .+ Kp*(findall(m .>= 0) .- 1)')[:] .+
        (Kp - n_bases(mesh))
    v[:,end-n₊+1:end] += u[:,idxup]*LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => C[m.>=0]),
        B.boundary_flux.upper.in/Δ(mesh,n_intervals(mesh)),
    )
    # out of upper 
    idxdown = n₋ .+ (Kp*(findall(C .< 0).-1)' .+ (1:n_bases(mesh)))[:] .+
        (Kp - n_bases(mesh))
    v[:,idxdown] += u[:,end-n₊+1:end]*kron(model.T[m.>=0,C.<0],B.boundary_flux.upper.out')

    # innards
    for i in phases(model), j in phases(model)
        if i == j 
            # mult on diagonal
            for k in 1:n_intervals(mesh)
                k_idx = (i-1)*Kp .+ (k-1)*n_bases(mesh) .+ (1:n_bases(mesh)) .+ n₋
                for ℓ in 1:n_intervals(mesh)
                    if (k == ℓ+1) && (C[i] > 0)
                        ℓ_idx = k_idx .- n_bases(mesh) 
                        v[:,k_idx] += C[i]*(u[:,ℓ_idx]*B.blocks[4])/Δ(mesh,ℓ)
                    elseif k == ℓ
                        v[:,k_idx] += (u[:,k_idx]*(abs(C[i])*B.blocks[2 + (C[i].<0)]/Δ(mesh,ℓ) + model.T[i,j]*LinearAlgebra.I))
                    elseif (k == ℓ-1) && (C[i] < 0)
                        ℓ_idx = k_idx .+ n_bases(mesh) 
                        v[:,k_idx] += abs(C[i])*(u[:,ℓ_idx]*B.blocks[1])/Δ(mesh,ℓ)
                    end
                end
            end
        elseif m[i]!=m[j]# B.pmidx[i,j]
            # changes from S₊ to S₋ etc.
            for k in 1:n_intervals(mesh)
                for ℓ in 1:n_intervals(mesh)
                    if k == ℓ
                        i_idx = (i-1)*Kp .+ (k-1)*n_bases(mesh) .+ (1:n_bases(mesh)) .+ n₋
                        j_idx = (j-1)*Kp .+ (k-1)*n_bases(mesh) .+ (1:n_bases(mesh)) .+ n₋
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

function *(B::LazyGenerator, u::AbstractArray{<:Real,2})
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

    model = B.model
    mesh = B.mesh
    Kp = total_n_bases(mesh) # K = n_intervals, p = n_bases
    m = membership(model.S)
    C = rates(model)
    n₋ = N₋(model.S)
    n₊ = N₊(model.S)
    # boundaries
    # at lower
    v[1:n₋,:] += model.T[m.<=0,m.<=0]*u[1:n₋,:]
    # in to lower 
    idxdown = n₋ .+ ((1:n_bases(mesh)).+Kp*(findall(m .<= 0) .- 1)')[:]
    v[idxdown,:] += LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => abs.(C[m.<=0])),
        B.boundary_flux.lower.in/Δ(mesh,1),
    )*u[1:n₋,:]
    # out of lower 
    idxup = n₋ .+ (Kp*(findall(C .> 0).-1)' .+ (1:n_bases(mesh)))[:]
    v[1:n₋,:] += kron(model.T[m.<=0,C.>0],B.boundary_flux.lower.out')*u[idxup,:]

    # at upper
    v[end-n₊+1:end,:] += model.T[m.>=0,m.>=0]*u[end-n₊+1:end,:]
    # in to upper
    idxup = n₋ .+ ((1:n_bases(mesh)).+Kp*(findall(m .>= 0) .- 1)')[:] .+
        (Kp - n_bases(mesh))
    v[idxup,:] += LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => C[m.>=0]),
        B.boundary_flux.upper.in/Δ(mesh,n_intervals(mesh)),
    )*u[end-n₊+1:end,:]
    # out of upper 
    idxdown = n₋ .+ (Kp*(findall(C .< 0).-1)' .+ (1:n_bases(mesh)))[:] .+
        (Kp - n_bases(mesh))
    v[end-n₊+1:end,:] += kron(model.T[m.>=0,C.<0],B.boundary_flux.upper.out')*u[idxdown,:]

    # innards
    for i in phases(model), j in phases(model)
        if i == j 
            # mult on diagonal
            for k in 1:n_intervals(mesh)
                k_idx = (i-1)*Kp .+ (k-1)*n_bases(mesh) .+ (1:n_bases(mesh)) .+ n₋
                for ℓ in 1:n_intervals(mesh)
                    if (k == ℓ+1) && (C[i] > 0) # upper diagonal block
                        ℓ_idx = k_idx .- n_bases(mesh) 
                        v[ℓ_idx,:] += C[i]*(B.blocks[4]*u[k_idx,:])/Δ(mesh,ℓ)
                    elseif k == ℓ # diagonal 
                        v[k_idx,:] += ((abs(C[i])*B.blocks[2 + (C[i].<0)]/Δ(mesh,ℓ) + model.T[i,j]*LinearAlgebra.I)*u[k_idx,:])
                    elseif (k == ℓ-1) && (C[i] < 0) # lower diagonal
                        ℓ_idx = k_idx .+ n_bases(mesh) 
                        v[ℓ_idx,:] += abs(C[i])*(B.blocks[1]*u[k_idx,:])/Δ(mesh,ℓ)
                    end
                end
            end
        elseif m[i]!=m[j] # B.pmidx[i,j]
            # changes from S₊ to S₋ etc.
            for k in 1:n_intervals(mesh)
                for ℓ in 1:n_intervals(mesh)
                    if k == ℓ
                        i_idx = (i-1)*Kp .+ (k-1)*n_bases(mesh) .+ (1:n_bases(mesh)) .+ n₋
                        j_idx = (j-1)*Kp .+ (k-1)*n_bases(mesh) .+ (1:n_bases(mesh)) .+ n₋
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

*(B::LazyGenerator, u::LazyGenerator) = SparseArrays.SparseMatrixCSC{Float64,Int}(B)*u

function show(io::IO, mime::MIME"text/plain", B::LazyGenerator)
    if VERSION >= v"1.6"
        show(io, mime, SparseArrays.SparseMatrixCSC(Matrix(LinearAlgebra.I(size(B,1)))*B))
    else
        show(io, mime, Matrix(SparseArrays.SparseMatrixCSC(Matrix(LinearAlgebra.I(size(B,1)))*B)))
    end
end
# show(B::LazyGenerator) = show(stdout, B)

function getindex(B::LazyGenerator,row::Int,col::Int)
    checkbounds(B,row,col)

    sz_B_1 = size(B,1)

    model = B.model
    mesh = B.mesh

    Kp = total_n_bases(mesh) # K = n_intervals(mesh), p = n_bases(mesh)
    m = membership(model.S)
    C = rates(model)
    n₋ = N₋(model.S)
    n₊ = N₊(model.S)

    v = 0.0

    if (row ∈ 1:n₋) && (col ∈ 1:n₋)
        v = model.T[m.<=0,m.<=0][row,col]
    elseif (row ∉ 1:n₋) && (row ∉ (sz_B_1 .+ 1) .- (1:n₊)) && (col ∈ 1:n₋) # in to lower 
        idxdown = n₋ .+ ((1:n_bases(mesh)).+Kp*(findall(m .<= 0) .- 1)')[:]
        idx = findfirst(x -> x==row, idxdown)
        if (nothing!=idx) 
            v = LinearAlgebra.kron(
                LinearAlgebra.diagm(0 => abs.(C[m.<=0])),
                B.boundary_flux.lower.in/Δ(mesh,1),
            )[idx,col]
        end
    elseif (row ∈ 1:n₋) && (col ∉ 1:n₋) && (col ∉ sz_B_1 .+ 1 .- (1:n₊)) # out of lower 
        idxup = n₋ .+ (Kp*(findall(C .> 0).-1)' .+ (1:n_bases(mesh)))[:]
        idx = findfirst(x -> x==col,idxup)
        (nothing!=idx) && (v = kron(model.T[m.<=0,C.>0],B.boundary_flux.lower.out')[row,idx])
    elseif (row ∈ (sz_B_1 .+ 1) .- (1:n₊)) && (col ∈ (sz_B_1 .+ 1) .- (1:n₊)) # at upper
        v = model.T[m.>=0,m.>=0][row-sz_B_1+n₊, col-sz_B_1+n₊]
    elseif (row ∉ (sz_B_1 .+ 1) .- (1:n₊)) && (col ∈ (sz_B_1 .+ 1) .- (1:n₊)) # in to upper
        idxup = n₋ .+ ((1:n_bases(mesh)).+Kp*(findall(m .>= 0) .- 1)')[:] .+
            (Kp - n_bases(mesh))
        idx = findfirst(x -> x==row, idxup)
        if (nothing!=idx)
            v = LinearAlgebra.kron(
                LinearAlgebra.diagm(0 => C[m.>=0]),
                B.boundary_flux.upper.in/Δ(mesh,n_intervals(mesh)),
            )[idx, col-sz_B_1+n₊]
        end
    elseif (row ∈ sz_B_1 .+ 1 .- (1:n₊)) && (col ∉ sz_B_1 .+ 1 .- (1:n₊)) # out of upper 
        idxdown = n₋ .+ (Kp*(findall(C .< 0).-1)' .+ (1:n_bases(mesh)))[:] .+
            (Kp - n_bases(mesh))
        idx = findfirst(x -> x==col, idxdown)
        (nothing!=idx) && (v = kron(model.T[m.>=0,C.<0],B.boundary_flux.upper.out')[row-sz_B_1+n₊, idx])
    else
        # innards
        # find if (T⊗I)[row,col] != 0
        row_shift = row-n₋
        col_shift = col-n₋
        # if mod(row_shift,Kp) == mod(col_shift,Kp) 
        #     display(7)
        #     i = (row_shift-1)÷(Kp) + 1 
        #     j = (col_shift-1)÷(Kp) + 1
        #     (m[i]==m[j]) && (v = model.T[i,j])
        # end
        # for i in 1:size_T, j in 1:size_T
        # phase block
        i = (row_shift-1)÷(Kp) + 1
        j = (col_shift-1)÷(Kp) + 1
        if i == j 
            # on diagonal
            # find block position [b_r,b_c] in the remaining tridiagonal matrix
            b_r = (row_shift-1 - (i-1)*(Kp))÷n_bases(mesh) + 1
            b_c = (col_shift-1 - (j-1)*(Kp))÷n_bases(mesh) + 1
            if (b_c == b_r+1) && (C[i] > 0) && (b_r+1 <= n_intervals(mesh))
                # find position [r,c] in remaining block 
                r = row_shift - (i-1)*(Kp) - (b_r-1)*n_bases(mesh)
                c = col_shift - (j-1)*(Kp) - (b_c-1)*n_bases(mesh)
                v = C[i]*B.blocks[4][r,c]/Δ(mesh,b_r)
            elseif (b_c == b_r) 
                r = row_shift - (i-1)*(Kp) - (b_r-1)*n_bases(mesh)
                c = col_shift - (j-1)*(Kp) - (b_c-1)*n_bases(mesh)
                (C[i]!=0) && (v = abs(C[i])*B.blocks[2 + (C[i].<0)][r,c]/Δ(mesh,b_r))
                (r==c) && (v += model.T[i,j])
            elseif (b_c == b_r-1) && (C[i] < 0) && (b_r-1 > 0)
                r = row_shift - (i-1)*(Kp) - (b_r-1)*n_bases(mesh)
                c = col_shift - (j-1)*(Kp) - (b_c-1)*n_bases(mesh)
                v = abs(C[i])*B.blocks[1][r,c]/Δ(mesh,b_r)
            end
        elseif m[i]!=m[j]# B.pmidx[i,j]
            # changes from S₊ to S₋ etc.
            b_r = (row_shift-1 - (i-1)*(Kp))÷n_bases(mesh) + 1
            b_c = (col_shift-1 - (j-1)*(Kp))÷n_bases(mesh) + 1
            if b_r == b_c
                r = row_shift - (i-1)*(Kp) - (b_r-1)*n_bases(mesh)
                c = col_shift - (j-1)*(Kp) - (b_c-1)*n_bases(mesh)
                v = model.T[i,j]*B.D[r,c]
            end
        elseif m[i]==m[j]# B.pmidx[i,j]
            # changes from S₊ to S₋ etc.
            b_r = (row_shift-1 - (i-1)*(Kp))÷n_bases(mesh) + 1
            b_c = (col_shift-1 - (j-1)*(Kp))÷n_bases(mesh) + 1
            if b_r == b_c
                r = row_shift - (i-1)*(Kp) - (b_r-1)*n_bases(mesh)
                c = col_shift - (j-1)*(Kp) - (b_c-1)*n_bases(mesh)
                (r==c) && (v = model.T[i,j])
            end
        end
    end
    return v
end

export getindex, *

