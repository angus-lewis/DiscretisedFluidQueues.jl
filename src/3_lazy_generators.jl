abstract type Generator <: AbstractArray{Real,2} end 

struct LazyGenerator  <: Generator
    blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}}
    boundary_flux::NamedTuple{(:upper,:lower),Tuple{NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}},NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}}}}
    T::Array{<:Real,2}
    C::PhaseSet
    Δ::Array{<:Real,1}
    D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}}
    pmidx::Union{Array{Bool,2},BitArray{2}}
    # Fil::IndexDict
    function LazyGenerator(
        blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}},
        boundary_flux::NamedTuple{(:upper,:lower),Tuple{NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}},NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}}}},
        T::Array{<:Real,2},
        C::PhaseSet,
        Δ::Array{<:Real,1},
        D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
        pmidx::Union{Array{Bool,2},BitArray{2}},
        # Fil::IndexDict,
    )
        s = size(blocks[1])
        for b in 1:4
            checksquare(blocks[b]) 
            !(s == size(blocks[b])) && throw(DomainError("blocks must be the same size"))
        end
        checksquare(T)
        checksquare(D)
        !(s == size(D)) && throw(DomainError("blocks must be the same size as D"))
        checksquare(pmidx) 
        !(size(T) == size(pmidx)) && throw(DomainError(pmidx, "must be the same size as T"))
        !(length(C) == size(T,1)) && throw(DomainError(C, "must be the same length as T"))
        
        return new(blocks,boundary_flux,T,C,Δ,D,pmidx)#,Fil)
    end
end
function LazyGenerator(
    blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}},
    boundary_flux::NamedTuple{(:in, :out),Tuple{Array{Float64,1},Array{Float64,1}}},
    T::Array{<:Real,2},
    C::PhaseSet,
    Δ::Array{<:Real,1},
    D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
    pmidx::Union{Array{Bool,2},BitArray{2}},
    # Fil::IndexDict,
)
    blocks = (blocks[1],blocks[2],blocks[2],blocks[3])
    boundary_flux = (upper = boundary_flux, lower = boundary_flux)
    return LazyGenerator(
        blocks,
        boundary_flux,
        T,
        C,
        Δ,
        D,
        pmidx,
        # Fil,
    )
end
function LazyGenerator(
    blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}},
    boundary_flux::NamedTuple{(:in, :out),Tuple{Array{Float64,1},Array{Float64,1}}},
    T::Array{<:Real,2},
    C::Array{<:Real,1},
    Δ::Array{<:Real,1},
    D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
    pmidx::Union{Array{Bool,2},BitArray{2}},
    # Fil::IndexDict,
)
    blocks = (blocks[1],blocks[2],blocks[2],blocks[3])
    boundary_flux = (upper = boundary_flux, lower = boundary_flux)
    return LazyGenerator(blocks, boundary_flux, T,
        PhaseSet(C), Δ, D, pmidx,# Fil,
    )
end
function MakeLazyGenerator(model::Model, mesh::Mesh; v::Bool=false)
    throw(DomainError("Can construct LazyGenerator for DGMesh, FRAPMesh, only"))
end

function size(B::LazyGenerator)
    sz = size(B.T,1)*size(B.blocks[1],1)*length(B.Δ) + sum(B.C.<=0) + sum(B.C.>=0)
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
    # N₋ = sum(B.C.<=0)
    # N₊ = sum(B.C.>=0)
    if output_type <: SparseArrays.SparseMatrixCSC
        v = SparseArrays.spzeros(sz_u_1,sz_B_2)
    else 
        v = zeros(sz_u_1,sz_B_2)
    end
    size_delta = length(B.Δ)
    size_blocks = size(B.blocks[1],1)
    size_T = size(B.T,1)

    # boundaries
    # at lower
    v[:,1:N₋(B.C)] += u[:,1:N₋(B.C)]*B.T[B.C.<=0,B.C.<=0]
    # in to lower 
    idxdown = N₋(B.C) .+ ((1:size_blocks).+size_blocks*size_delta*(findall(B.C .<= 0) .- 1)')[:]
    v[:,1:N₋(B.C)] += u[:,idxdown]*LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => abs.(B.C[B.C.<=0])),
        B.boundary_flux.lower.in/B.Δ[1],
    )
    # out of lower 
    idxup = N₋(B.C) .+ (size_blocks*size_delta*(findall(B.C .> 0).-1)' .+ (1:size_blocks))[:]
    v[:,idxup] += u[:,1:N₋(B.C)]*kron(B.T[B.C.<=0,B.C.>0],B.boundary_flux.lower.out')

    # at upper
    v[:,end-N₊(B.C)+1:end] += u[:,end-N₊(B.C)+1:end]*B.T[B.C.>=0,B.C.>=0]
    # in to upper
    idxup = N₋(B.C) .+ ((1:size_blocks).+size_blocks*size_delta*(findall(B.C .> 0) .- 1)')[:] .+
        (size_blocks*size_delta - size_blocks)
    v[:,end-N₊(B.C)+1:end] += u[:,idxup]*LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => B.C[B.C.>0]),
        B.boundary_flux.upper.in/B.Δ[end],
    )
    # out of upper 
    idxdown = N₋(B.C) .+ (size_blocks*size_delta*(findall(B.C .< 0).-1)' .+ (1:size_blocks))[:] .+
        (size_blocks*size_delta - size_blocks)
    v[:,idxdown] += u[:,end-N₊(B.C)+1:end]*kron(B.T[B.C.>=0,B.C.<0],B.boundary_flux.upper.out')

    # innards
    for i in 1:size_T, j in 1:size_T
        if i == j 
            # mult on diagonal
            for k in 1:size_delta
                k_idx = (i-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks) .+ N₋(B.C)
                for ℓ in 1:size_delta
                    if (k == ℓ+1) && (B.C[i] > 0)
                        ℓ_idx = k_idx .- size_blocks 
                        v[:,k_idx] += B.C[i]*(u[:,ℓ_idx]*B.blocks[4])/B.Δ[ℓ]
                    elseif k == ℓ
                        v[:,k_idx] += (u[:,k_idx]*(abs(B.C[i])*B.blocks[2 + (B.C[i].<0)]/B.Δ[ℓ] + B.T[i,j]*LinearAlgebra.I))
                    elseif (k == ℓ-1) && (B.C[i] < 0)
                        ℓ_idx = k_idx .+ size_blocks 
                        v[:,k_idx] += abs(B.C[i])*(u[:,ℓ_idx]*B.blocks[1])/B.Δ[ℓ]
                    end
                end
            end
        elseif B.pmidx[i,j]
            # changes from S₊ to S₋ etc.
            for k in 1:size_delta
                for ℓ in 1:size_delta
                    if k == ℓ
                        i_idx = (i-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks) .+ N₋(B.C)
                        j_idx = (j-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks) .+ N₋(B.C)
                        v[:,j_idx] += (u[:,i_idx]*(B.T[i,j]*B.D))
                    end
                end
            end
        else
            i_idx = (i-1)*size_blocks*size_delta .+ (1:size_blocks*size_delta) .+ N₋(B.C)
            j_idx = (j-1)*size_blocks*size_delta .+ (1:size_blocks*size_delta) .+ N₋(B.C)
            v[:,j_idx] += (u[:,i_idx]*B.T[i,j])
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
    # N₋ = sum(B.C.<=0)
    # N₊ = sum(B.C.>=0)
    if output_type <: SparseArrays.SparseMatrixCSC
        v = SparseArrays.spzeros(sz_u_1,sz_B_2)
    else 
        v = zeros(sz_u_1,sz_B_2)
    end
    size_delta = length(B.Δ)
    size_blocks = size(B.blocks[1],1)
    size_T = size(B.T,1)

    # boundaries
    # at lower
    v[1:N₋(B.C),:] += B.T[B.C.<=0,B.C.<=0]*u[1:N₋,:]
    # in to lower 
    idxdown = N₋ .+ ((1:size_blocks).+size_blocks*size_delta*(findall(B.C .<= 0) .- 1)')[:]
    v[idxdown,:] += LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => abs.(B.C[B.C.<=0])),
        B.boundary_flux.lower.in/B.Δ[1],
    )*u[1:N₋,:]
    # out of lower 
    idxup = N₋ .+ (size_blocks*size_delta*(findall(B.C .> 0).-1)' .+ (1:size_blocks))[:]
    v[1:N₋,:] += kron(B.T[B.C.<=0,B.C.>0],B.boundary_flux.lower.out')*u[idxup,:]

    # at upper
    v[end-N₊+1:end,:] += B.T[B.C.>=0,B.C.>=0]*u[end-N₊+1:end,:]
    # in to upper
    idxup = N₋ .+ ((1:size_blocks).+size_blocks*size_delta*(findall(B.C .> 0) .- 1)')[:] .+
        (size_blocks*size_delta - size_blocks)
    v[idxup,:] += LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => B.C[B.C.>0]),
        B.boundary_flux.upper.in/B.Δ[end],
    )*u[end-N₊+1:end,:]
    # out of upper 
    idxdown = N₋ .+ (size_blocks*size_delta*(findall(B.C .< 0).-1)' .+ (1:size_blocks))[:] .+
        (size_blocks*size_delta - size_blocks)
    v[end-N₊+1:end,:] += kron(B.T[B.C.>=0,B.C.<0],B.boundary_flux.upper.out')*u[idxdown,:]

    # innards
    for i in 1:size_T, j in 1:size_T
        if i == j 
            # mult on diagonal
            for k in 1:size_delta
                k_idx = (i-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks) .+ N₋
                for ℓ in 1:size_delta
                    if (k == ℓ+1) && (B.C[i] > 0)
                        ℓ_idx = k_idx .- size_blocks 
                        v[ℓ_idx,:] += B.C[i]*(B.blocks[4]*u[k_idx,:])/B.Δ[ℓ]
                    elseif k == ℓ
                        v[k_idx,:] += ((abs(B.C[i])*B.blocks[2 + (B.C[i].<0)]/B.Δ[ℓ] + B.T[i,j]*LinearAlgebra.I)*u[k_idx,:])
                    elseif (k == ℓ-1) && (B.C[i] < 0)
                        ℓ_idx = k_idx .+ size_blocks 
                        v[ℓ_idx,:] += abs(B.C[i])*(B.blocks[1]*u[k_idx,:])/B.Δ[ℓ]
                    end
                end
            end
        elseif B.pmidx[i,j]
            # changes from S₊ to S₋ etc.
            for k in 1:size_delta
                for ℓ in 1:size_delta
                    if k == ℓ
                        i_idx = (i-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks) .+ N₋
                        j_idx = (j-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks) .+ N₋
                        v[i_idx,:] += (B.T[i,j]*B.D)*u[j_idx,:]
                    end
                end
            end
        else
            i_idx = (i-1)*size_blocks*size_delta .+ (1:size_blocks*size_delta) .+ N₋
            j_idx = (j-1)*size_blocks*size_delta .+ (1:size_blocks*size_delta) .+ N₋
            v[i_idx,:] += B.T[i,j]*u[j_idx,:]
        end
    end
    return v
end

*(B::LazyGenerator, u::LazyGenerator) = SparseArrays.SparseMatrixCSC(B)*u

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

    N₋ = sum(B.C.<=0)
    N₊ = sum(B.C.>=0)

    v = 0.0

    size_delta = length(B.Δ)
    size_blocks = size(B.blocks[1],1)

    if (row ∈ 1:N₋) && (col ∈ 1:N₋)
        v += B.T[B.C.<=0,B.C.<=0][row,col]
    elseif (row ∉ 1:N₋) && (row ∉ (sz_B_1 .+ 1) .- (1:N₊)) && (col ∈ 1:N₋) # in to lower 
        idxdown = N₋ .+ ((1:size_blocks).+size_blocks*size_delta*(findall(B.C .<= 0) .- 1)')[:]
        idx = findfirst(x -> x==row, idxdown)
        if (nothing!=idx) 
            v += LinearAlgebra.kron(
                LinearAlgebra.diagm(0 => abs.(B.C[B.C.<=0])),
                B.boundary_flux.lower.in/B.Δ[1],
            )[idx,col]
        end
    elseif (row ∈ 1:N₋) && (col ∉ 1:N₋) && (col ∉ sz_B_1 .+ 1 .- (1:N₊)) # out of lower 
        idxup = N₋ .+ (size_blocks*size_delta*(findall(B.C .> 0).-1)' .+ (1:size_blocks))[:]
        idx = findfirst(x -> x==col,idxup)
        (nothing!=idx) && (v += kron(B.T[B.C.<=0,B.C.>0],B.boundary_flux.lower.out')[row,idx])
    elseif (row ∈ (sz_B_1 .+ 1) .- (1:N₊)) && (col ∈ (sz_B_1 .+ 1) .- (1:N₊)) # at upper
        v += B.T[B.C.>=0,B.C.>=0][row-sz_B_1+N₊, col-sz_B_1+N₊]
    elseif (row ∉ (sz_B_1 .+ 1) .- (1:N₊)) && (col ∈ (sz_B_1 .+ 1) .- (1:N₊)) # in to upper
        idxup = N₋ .+ ((1:size_blocks).+size_blocks*size_delta*(findall(B.C .> 0) .- 1)')[:] .+
            (size_blocks*size_delta - size_blocks)
        idx = findfirst(x -> x==row, idxup)
        if (nothing!=idx)
            v += LinearAlgebra.kron(
                LinearAlgebra.diagm(0 => B.C[B.C.>0]),
                B.boundary_flux.upper.in/B.Δ[end],
            )[idx, col-sz_B_1+N₊]
        end
    elseif (row ∈ sz_B_1 .+ 1 .- (1:N₊)) && (col ∉ sz_B_1 .+ 1 .- (1:N₊)) # out of upper 
        idxdown = N₋ .+ (size_blocks*size_delta*(findall(B.C .< 0).-1)' .+ (1:size_blocks))[:] .+
            (size_blocks*size_delta - size_blocks)
        idx = findfirst(x -> x==col, idxdown)
        (nothing!=idx) && (v += kron(B.T[B.C.>=0,B.C.<0],B.boundary_flux.upper.out')[row-sz_B_1+N₊, idx])
    else
        # innards
        # find if (T⊗I)[row,col] != 0
        row_shift = row-N₋
        col_shift = col-N₋
        if mod(row_shift,size_delta*size_blocks) == mod(col_shift,size_delta*size_blocks) 
            i = (row_shift-1)÷(size_delta*size_blocks) + 1 
            j = (col_shift-1)÷(size_delta*size_blocks) + 1
            !(B.pmidx[i,j]) && (v += B.T[i,j])
        end
        # for i in 1:size_T, j in 1:size_T
        # phase block
        i = (row_shift-1)÷(size_delta*size_blocks) + 1
        j = (col_shift-1)÷(size_delta*size_blocks) + 1
        if i == j 
            # on diagonal
            # find block position [b_r,b_c] in the remaining tridiagonal matrix
            b_r = (row_shift-1 - (i-1)*(size_delta*size_blocks))÷size_blocks + 1
            b_c = (col_shift-1 - (j-1)*(size_delta*size_blocks))÷size_blocks + 1
            if (b_c == b_r+1) && (B.C[i] > 0) && (b_r+1 <= size_delta)
                # find position [r,c] in remaining block 
                r = row_shift - (i-1)*(size_delta*size_blocks) - (b_r-1)*size_blocks
                c = col_shift - (j-1)*(size_delta*size_blocks) - (b_c-1)*size_blocks
                v += B.C[i]*B.blocks[4][r,c]/B.Δ[r]
            elseif b_c == b_r
                r = row_shift - (i-1)*(size_delta*size_blocks) - (b_r-1)*size_blocks
                c = col_shift - (j-1)*(size_delta*size_blocks) - (b_c-1)*size_blocks
                v += abs(B.C[i])*B.blocks[2 + (B.C[i].<0)][r,c]/B.Δ[r] 
            elseif (b_c == b_r-1) && (B.C[i] < 0) && (b_r-1 > 0)
                r = row_shift - (i-1)*(size_delta*size_blocks) - (b_r-1)*size_blocks
                c = col_shift - (j-1)*(size_delta*size_blocks) - (b_c-1)*size_blocks
                v += abs(B.C[i])*B.blocks[1][r,c]/B.Δ[r]
            end
        elseif B.pmidx[i,j]
            # changes from S₊ to S₋ etc.
            # find block position [b_r,b_c] in the remaining tridiagonal matrix
            b_r = (row_shift-1 - (i-1)*(size_delta*size_blocks))÷size_blocks + 1
            b_c = (col_shift-1 - (j-1)*(size_delta*size_blocks))÷size_blocks + 1
            if b_r == b_c
                r = row_shift - (i-1)*(size_delta*size_blocks) - (b_r-1)*size_blocks
                c = col_shift - (j-1)*(size_delta*size_blocks) - (b_c-1)*size_blocks
                v += B.T[i,j]*B.D[r,c]
            end
        end
    end
    return v
end

