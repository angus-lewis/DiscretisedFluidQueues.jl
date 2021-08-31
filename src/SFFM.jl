module SFFM
import Base: *, size, show, getindex, +, -, setindex!
import Jacobi, LinearAlgebra, SparseArrays
import Plots, StatsBase, KernelDensity

abstract type Model end

"""
Construct a SFFM model object.

    FluidFluidQueue(
        T::Array{Float64,2},
        C::Array{Float64,1},
        r::NamedTuple{(:r, :R)};
        Bounds::Array{<:Real,2} = [-Inf Inf; -Inf Inf],
    )

# Arguments
- `T::Array{Float64,2}`: generator matrix for the CTMC ``φ(t)``
- `C::Array{Float64,1}`: vector of rates ``d/dt X(t)=C[i]`` for ``i=φ(t)``.
- `r::NamedTuple{(:r, :R)}`: rates for the second fluid.
    - `:r(x::Array{Real})`, a function  which takes arrays of x-values and
        returns a row vector of values for each x-value. i.e. `:r([0;1])`
        returns a `2×NPhases` array where the first row contains all the
        ``rᵢ(0)`` and row 2 all the ``rᵢ(1)`` values.
    - `:R(x::Array{Real})`: has the same structure/behaviour as ``:r`` but
        returns the integral of ``:r``. i.e. `Rᵢ(x)=∫ˣrᵢ(y)dy`.
- `Bounds::Array{<:Real,2}`: contains the bounds for the model. The first row
    are the L and R bounds for ``X(t)`` and the second row the bounds for
    ``Y(t)`` (although the bounds for ``Y(t)`` don't actually do anything yet).

# Outputs
- a model object which is a tuple with fields
    - `:T`: as input
    - `:C`: as input
    - `:r`: a named tuple with fields `(:r, :R, :a)`, `:r` and `:R` are as input
        and `:a = abs.(:r)` returns the absolute values of the rates.
    - `Bounds`: as input
)
"""
struct PhaseSet 
    C::Array{<:Real,1}
end
get_rates(S::PhaseSet) = S.C
get_rates(S::PhaseSet,i::Int) = S.C[i]
n_phases(S::PhaseSet) = length(S.C)
phases(S::PhaseSet) = 1:n_phases(S::PhaseSet)
N₋(S::PhaseSet) = sum(S.C.<=0)
N₊(S::PhaseSet) = sum(S.C.>=0)

struct FluidQueue <: Model
    T::Array{<:Real,2}
    S::PhaseSet
    bounds::Array{<:Real,1}
end 
get_rates(m::FluidQueue) = get_rates(m.S)
get_rates(m::FluidQueue,i::Int) = get_rates(m.S,i)
n_phases(m::FluidQueue) = n_phases(m.S)
phases(m::FluidQueue) = 1:n_phases(m.S)

struct FluidFluidQueue <: Model
    T::Array{<:Real,2}
    S::PhaseSet
    r::NamedTuple{(:r, :R, :a)}
    Bounds::Array{<:Real}
end 
get_rates(m::FluidFluidQueue) = get_rates(m.S)
get_rates(m::FluidFluidQueue,i::Int) = get_rates(m.S,i)
n_phases(m::FluidFluidQueue) = n_phases(m.S)
phases(m::FluidFluidQueue) = 1:n_phases(m.S)


function FluidFluidQueue(
    T::Array{<:Real},
    C::Array{<:Real,1},
    r::NamedTuple{(:r, :R)};
    Bounds::Array{<:Real,2} = [-Inf Inf; -Inf Inf],
    v::Bool = false,
)
    a(x) = abs.(r.r(x))
    r = (r = r.r, R = r.R, a = a)

    v && println("UPDATE: FluidFluidQueue object created with fields ", fieldnames(SFFM.Model))
    return FluidFluidQueue(
        T,
        PhaseSet(C),
        r,
        Bounds,
    )
end
function FluidFluidQueue(
    T::Array{<:Real},
    S::PhaseSet,
    r::NamedTuple{(:r, :R)};
    Bounds::Array{<:Real,2} = [-Inf Inf; -Inf Inf],
    v::Bool = false,
)
    a(x) = abs.(r.r(x))
    r = (r = r.r, R = r.R, a = a)

    v && println("UPDATE: FluidFluidQueue object created with fields ", fieldnames(SFFM.Model))
    return FluidFluidQueue(
        T,
        S,
        r,
        Bounds,
    )
end
FluidFluidQueue() = FluidFluidQueue([0],PhaseSet([0]),(r=0, R=0, a=0),[0])

function _duplicate_zero_states(T::Array{<:Real,2},C::Array{<:Real,1}, r::NamedTuple{(:r, :R)})
    n_0 = sum(C.==0)
    
    # assign rates for the augmented model
    C_aug = Array{Float64,1}(undef,length(C)+n_0)
    c_zero = 0
    plus_idx = falses(length(C_aug)) # zero states associated with +
    neg_idx = falses(length(C_aug)) # zero states associated with -
    for i in 1:length(C)
        C_aug[i+c_zero] = C[i]
        if C[i] == 0
            C_aug[i+c_zero+1] = C[i]
            plus_idx[i+c_zero] = true
            neg_idx[i+c_zero+1] = true
            c_zero += 1
        end
    end

    # assign second fluid rates
    function r_aug_inner(x)
        out = zeros(length(C_aug))
        out[(C_aug.!=0).|(plus_idx)] = r.r(x)
        out[neg_idx] = r.r(x)[C.==0]
        return out
    end
    function R_aug(x)
        out = zeros(length(C_aug))
        out[(C_aug.!=0).|(plus_idx)] = r.R(x)
        out[neg_idx] = r.R(x)[C.==0]
        return out
    end
    r_aug = (r = r_aug_inner, R = R_aug)

    # assign augmented generator
    c_zero = 0
    T_aug = zeros(length(C_aug),length(C_aug))
    for i in 1:length(C)
        if C[i] == 0
            # duplicate 
            T_aug[i+c_zero,(C_aug.!=0).|(plus_idx)] = T[i,:]
            T_aug[i+c_zero+1,(C_aug.!=0).|(neg_idx)] = T[i,:]
            c_zero += 1
        elseif C[i] < 0
            T_aug[i+c_zero,(C_aug.!=0).|(neg_idx)] = T[i,:]
        elseif C[i] > 0 
            T_aug[i+c_zero,(C_aug.!=0).|(plus_idx)] = T[i,:]
        end
    end
    return T_aug, C_aug, r_aug
end

function augment_model(model::FluidFluidQueue)
    if (any(model.C.==0))
        T_aug, C_aug, r_aug = _duplicate_zero_states(model.T,model.C,model.r)
        return FluidFluidQueue(T_aug,C_aug,r_aug,model.Bounds)
    else # no zero states, no augmentation needed
       return model 
    end
end

# """

#     NPhases(model::Model)

# the number of states in the state space
# """
# NPhases(model::Model) = length(model.C)

# phases(model::Model) = 1:NPhases(model)

# """

#     modelDicts(model::Model) 

# input: a Model object

# outputs:
#      - SDict: a dictionary with keys `"+","-","0","bullet"`
#     and corresponding values `findall(model.C .> 0)`, `findall(model.C .< 0)`,
#     `findall(model.C .== 0)`, `findall(model.C .!= 0)`, respectively.

#      - TDict: a dictionary of submatrices of `T` with keys
#     `"ℓm"` with ``ℓ,m∈{+,-,0,bullet}`` and corresponding values
#     `model.T[S[ℓ],S[m]]`.
# """
# function modelDicts(model::Model) 
#     nPhases = NPhases(model)
#     SDict = Dict{String,Array}("S" => 1:nPhases)
#     SDict["+"] = findall(model.C .> 0)
#     SDict["-"] = findall(model.C .< 0)
#     SDict["0"] = findall(model.C .== 0)
#     SDict["bullet"] = findall(model.C .!= 0)

#     TDict = Dict{String,Array}("T" => model.T)
#     for ℓ in ["+" "-" "0" "bullet"], m in ["+" "-" "0" "bullet"]
#         TDict[ℓ*m] = model.T[SDict[ℓ], SDict[m]]
#     end

#     return SDict, TDict
# end

# """

#     TDict(model::Model) 


# """
# function TDict(model::Model) 
    
# end


"""

    Mesh 

Abstract type representing a mesh for a numerical scheme. 
"""
abstract type Mesh end 

""" First index of the generator to index the generator with ("+","-") etc, elements are strings """
const PlusMinusIndex = Union{String,Tuple{String,String}}

""" Second index of the generator to index the generator with (i,j) etc, elements are Int or :"""
const PhaseIndex = Union{Tuple{Union{Int64,Colon},Union{Int64,Colon}},Int64,Colon}

""" Index of the generator e.g. ("+","-"),(i,j) etc, elements are (String,String),(Int/Colon,Int/Colon) """
const GeneratorIndex = Union{PlusMinusIndex,Tuple{PlusMinusIndex,PhaseIndex}}

""" The dictionary Fil which indicate which cells correspond to +, -, or 0 fluid-fluid rates """
const IndexDict = Dict{Tuple{String,Union{Int64,Colon}},BitArray{1}}

""" The dictionary which returns various partitions of the generator i.e. ("+","-"),(i,j) """
const PartitionedGenerator = Dict{GeneratorIndex,SparseArrays.SparseMatrixCSC{Float64,Int64}}

"""

    MakeFil(
        model::FluidFluidQueue,
        Nodes::Array{<:Real,1},
        )

Construct dict with entries indexing which cells belong to Fᵢᵐ. 
"""
function MakeFil(
    model::FluidFluidQueue,
    Nodes::Array{<:Real,1},
    )
    meshNIntervals = length(Nodes) - 1
    Δtemp = Nodes[2:end] - Nodes[1:end-1]

    Fil = IndexDict()
    
    ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
    idxPlus = model.r.r(Nodes[1:end-1].+Δtemp[:]/2).>0
    idxZero = model.r.r(Nodes[1:end-1].+Δtemp[:]/2).==0
    idxMinus = model.r.r(Nodes[1:end-1].+Δtemp[:]/2).<0
    for i in 1:NPhases(model)
        Fil[("+",i)] = idxPlus[:,i]
        Fil[("0",i)] = idxZero[:,i]
        Fil[("-",i)] = idxMinus[:,i]
        if model.C[i] .<= 0
            Fil[("p+",i)] = [model.r.r(model.Bounds[1,1])[i]].>0
            Fil[("p0",i)] = [model.r.r(model.Bounds[1,1])[i]].==0
            Fil[("p-",i)] = [model.r.r(model.Bounds[1,1])[i]].<0
        end
        if model.C[i] .>= 0
            Fil[("q+",i)] = [model.r.r(model.Bounds[1,end])[i]].>0
            Fil[("q0",i)] = [model.r.r(model.Bounds[1,end])[i]].==0
            Fil[("q-",i)] = [model.r.r(model.Bounds[1,end])[i]].<0
        end
    end
    currKeys = keys(Fil)
    for ℓ in ["+", "-", "0"], i = 1:NPhases(model)
        if ((ℓ,i) ∉ currKeys)
            Fil[(ℓ,i)] = falses(meshNIntervals)
        end
        if (("p"*ℓ,i) ∉ currKeys) && (model.C[i] <= 0)
            Fil["p"*ℓ,i] = falses(1)
        end
        if (("p"*ℓ,i) ∉ currKeys) && (model.C[i] > 0)
            Fil["p"*ℓ,i] = falses(0)
        end
        if (("q"*ℓ,i) ∉ currKeys) && (model.C[i] >= 0)
            Fil["q"*ℓ,i] = falses(1)
        end
        if (("q"*ℓ,i) ∉ currKeys) && (model.C[i] < 0)
            Fil["q"*ℓ,i] = falses(0)
        end
    end
    for ℓ in ["+", "-", "0"]
        Fil[(ℓ,:)] = falses(meshNIntervals * NPhases(model))
        Fil[("p"*ℓ,:)] = trues(0)
        Fil[("q"*ℓ,:)] = trues(0)
        for i = 1:NPhases(model)
            idx = findall(Fil[(ℓ,i)]) .+ (i - 1) * meshNIntervals
            Fil[(ℓ,:)][idx] .= true
            Fil[("p"*ℓ,:)] = [Fil[("p"*ℓ,:)]; Fil[("p"*ℓ,i)]]
            Fil[("q"*ℓ,:)] = [Fil[("q"*ℓ,:)]; Fil[("q"*ℓ,i)]]
        end
    end
    return Fil
end

function MakeDict(
    B::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{<:Real,Int64}},
    model::Model, 
    mesh::Mesh,
    Fil::IndexDict,
    )

    ## Make a Dictionary so that the blocks of B are easy to access
    # N₋ = sum(model.C.<=0)
    # N₊ = sum(model.C.>=0)

    BDict = PartitionedGenerator()

    ppositions = cumsum(model.C .<= 0)
    qpositions = cumsum(model.C .>= 0)
    for ℓ in ["+", "-", "0"], m in ["+", "-", "0"]
        for i = 1:NPhases(model), j = 1:NPhases(model)
            FilBases = repeat(Fil[(ℓ,i)]', NBases(mesh), 1)[:]
            pitemp = falses(N₋(model.S))
            qitemp = falses(N₊(model.S))
            pjtemp = falses(N₋(model.S))
            qjtemp = falses(N₊(model.S))
            if model.C[i] <= 0
                if length(pitemp) > 0 
                    pitemp[ppositions[i]] = Fil[("p"*ℓ,i)][1]
                end
            end
            if model.C[j] <= 0
                if length(pjtemp) > 0
                    pjtemp[ppositions[j]] = Fil[("p"*m,j)][1]
                end
            end
            if model.C[i] >= 0
                if length(qitemp) > 0
                    qitemp[qpositions[i]] = Fil[("q"*ℓ,i)][1]
                end
            end
            if model.C[j] >= 0
                if length(qjtemp) > 0
                    qjtemp[qpositions[j]] = Fil[("q"*m,j)][1]
                end
            end
            i_idx = [
                pitemp
                falses((i - 1) * TotalNBases(mesh))
                FilBases
                falses(NPhases(model) * TotalNBases(mesh) - i * TotalNBases(mesh))
                qitemp
            ]
            FjmBases = repeat(Fil[(m, j)]', NBases(mesh), 1)[:]
            j_idx = [
                pjtemp
                falses((j - 1) * TotalNBases(mesh))
                FjmBases
                falses(NPhases(model) * TotalNBases(mesh) - j * TotalNBases(mesh))
                qjtemp
            ]
            BDict[((ℓ, m),(i, j))] = B[i_idx, j_idx]
        end
        # below we need to use repeat(Fil[ℓ]', NBases(mesh), 1)[:] to
        # expand the index Fil[ℓ] from cells to all basis function
        FlBases =
            [Fil["p"*ℓ,:]; repeat(Fil[ℓ,:]', NBases(mesh), 1)[:]; Fil["q"*ℓ,:]]
        FmBases =
            [Fil["p"*m,:]; repeat(Fil[m,:]', NBases(mesh), 1)[:]; Fil["q"*m,:]]
        BDict[((ℓ,m),(:,:))] = B[FlBases, FmBases]
    end
    return BDict
end

abstract type Generator <: AbstractArray{Real,2} end 

checksquare(A::AbstractArray{<:Any,2}, str::String) = if !(size(A,1)==size(A,2)); throw(DomainError(str*" must be square")); end 
checksquare(A::AbstractArray{<:Any,2}) = checksquare(A, "")

struct LazyGenerator  <: Generator
    blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}}
    boundary_flux::NamedTuple{(:upper,:lower),Tuple{NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}},NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}}}}
    T::Array{<:Real,2}
    C::PhaseSet
    Δ::Array{<:Real,1}
    D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}}
    pmidx::Union{Array{Bool,2},BitArray{2}}
    Fil::IndexDict
    function LazyGenerator(
        blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}},
        boundary_flux::NamedTuple{(:upper,:lower),Tuple{NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}},NamedTuple{(:in,:out),Tuple{Vector{Float64},Vector{Float64}}}}},
        T::Array{<:Real,2},
        C::PhaseSet,
        Δ::Array{<:Real,1},
        D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
        pmidx::Union{Array{Bool,2},BitArray{2}},
        Fil::IndexDict,
    )
        s = size(blocks[1])
        for b in 1:4
            checksquare(blocks[b],"blocks") 
            !(s == size(blocks[b])) && throw(DomainError("blocks must be the same size"))
        end
        checksquare(T,"T")
        checksquare(D,"D")
        !(s == size(D)) && throw(DomainError("blocks must be the same size as D"))
        checksquare(pmidx,"pmidx") 
        !(size(T) == size(pmidx)) && throw(DomainError("pmidx must be the same size as T"))
        !(length(C) == size(T,1)) && throw(DomainError("C must be the same length as T"))
        
        return new(blocks,boundary_flux,T,C,Δ,D,pmidx,Fil)
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
    Fil::IndexDict,
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
        Fil,
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
    Fil::IndexDict,
)
    blocks = (blocks[1],blocks[2],blocks[2],blocks[3])
    boundary_flux = (upper = boundary_flux, lower = boundary_flux)
    return LazyGenerator(
        blocks,
        boundary_flux,
        T,
        PhaseSet(C),
        Δ,
        D,
        pmidx,
        Fil,
    )
end

function size(B::SFFM.LazyGenerator)
    sz = size(B.T,1)*size(B.blocks[1],1)*length(B.Δ) + sum(B.C.<=0) + sum(B.C.>=0)
    return (sz,sz)
end
size(B::SFFM.LazyGenerator, n::Int) = size(B)[n]

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
    v[:,end-N₊(B.C)+1:end] += u[:,end-N₊+1:end]*B.T[B.C.>=0,B.C.>=0]
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
    N₋ = sum(B.C.<=0)
    N₊ = sum(B.C.>=0)
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
    v[1:N₋,:] += B.T[B.C.<=0,B.C.<=0]*u[1:N₋,:]
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

function show(io::IO, mime::MIME"text/plain", B::SFFM.LazyGenerator)
    if VERSION >= v"1.6"
        show(io, mime, SparseArrays.SparseMatrixCSC(Matrix(LinearAlgebra.I(size(B,1)))*B))
    else
        show(io, mime, Matrix(SparseArrays.SparseMatrixCSC(Matrix(LinearAlgebra.I(size(B,1)))*B)))
    end
end
# show(B::SFFM.LazyGenerator) = show(stdout, B)

function getindex(B::SFFM.LazyGenerator,row::Int,col::Int)
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

# function getindex_correct(B::SFFM.LazyGenerator,row::Int,col::Int)
#     checkbounds(B,row,col)

#     ei = zeros(1,size(B,1))
#     ei[row] = 1
#     return (ei*B)[col]
# end

function getindex(B::SFFM.LazyGenerator,plus_minus_index::PlusMinusIndex,phase_index::PhaseIndex)
    N₋ = sum(B.C.<=0)
    N₊ = sum(B.C.<=0)

    out = []

    ppositions = cumsum(B.C .<= 0)
    qpositions = cumsum(B.C .>= 0)

    ℓ,m = plus_minus_index
    if typeof(phase_index[1])==Colon
        i_range = 1:length(B.C)
    else
        i_range = phase_index[1]
    end
    if typeof(phase_index[2])==Colon
        j_range = 1:length(B.C)
    else
        j_range = phase_index[2]
    end

    for i = i_range, j = j_range
        FilBases = repeat(B.Fil[ℓ,i]', size(B.D,1), 1)[:]
        pitemp = falses(N₋)
        qitemp = falses(N₊)
        pjtemp = falses(N₋)
        qjtemp = falses(N₊)
        if B.C[i] <= 0
            if length(pitemp) > 0 
                pitemp[ppositions[i]] = B.Fil["p"*ℓ,i][1]
            end
        end
        if B.C[j] <= 0
            if length(pjtemp) > 0
                pjtemp[ppositions[j]] = B.Fil["p"*m,j][1]
            end
        end
        if B.C[i] >= 0
            if length(qitemp) > 0
                qitemp[qpositions[i]] = B.Fil["q"*ℓ,i][1]
            end
        end
        if B.C[j] >= 0
            if length(qjtemp) > 0
                qjtemp[qpositions[j]] = B.Fil["q"*m,j][1]
            end
        end
        i_idx = [
            pitemp
            falses((i - 1) * size(B.D,1)*length(B.Δ))
            FilBases
            falses(length(B.C) * size(B.D,1)*length(B.Δ) - i * size(B.D,1)*length(B.Δ))
            qitemp
        ]
        FjmBases = repeat(B.Fil[m,j]', size(B.D,1), 1)[:]
        j_idx = [
            pjtemp
            falses((j - 1) * size(B.D,1)*length(B.Δ))
            FjmBases
            falses(length(B.C) * size(B.D,1)*length(B.Δ) - j * size(B.D,1)*length(B.Δ))
            qjtemp
        ]
        if (typeof(phase_index[1]) != Colon) && (typeof(phase_index[2]) != Colon)
            out = B[i_idx, j_idx]
        end
    end
    # we use repeat(mesh.Fil[ℓ]', NBases(mesh), 1)[:] to
    # expand the index mesh.Fil[ℓ] from cells to all basis function
    if (typeof(phase_index[1]) == Colon) && (typeof(phase_index[2]) == Colon)
        FlBases =
            [B.Fil["p"*ℓ,:]; repeat(B.Fil[ℓ,:]', size(B.D,1), 1)[:]; B.Fil["q"*ℓ,:]]
        FmBases =
            [B.Fil["p"*m,:]; repeat(B.Fil[m,:]', size(B.D,1), 1)[:]; B.Fil["q"*m,:]]
        out = B[FlBases, FmBases]
    end

    return out
end

getindex(B::SFFM.LazyGenerator,idx::Tuple{PlusMinusIndex,PhaseIndex}) = getindex(B,idx[1],idx[2])

# this is a bit slow...
function MakeDict(B::SFFM.LazyGenerator)
    BDict = SFFM.PartitionedGenerator()
    for i in 1:length(B.C), j in 1:length(B.C)
        for k in ["+","-","0"], ℓ in ["+","-","0"]
            plus_minus_index = (k,ℓ)
            phase_index = (i,j)
            BDict[plus_minus_index, phase_index] = B[plus_minus_index, phase_index]
        end
    end
    i = :;
    j = :;
    for k in ["+","-","0"], ℓ in ["+","-","0"] 
        plus_minus_index = (k,ℓ)
        phase_index = (i,j)
        BDict[plus_minus_index, phase_index] = B[plus_minus_index, phase_index]
    end
    return BDict
end

struct FullGenerator <: Generator 
    BDict::PartitionedGenerator
    B::Union{Array{Float64,Int64}, SparseArrays.SparseMatrixCSC{Float64,Int}}
    Fil::IndexDict
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

function MakeDict(
    B::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{<:Real,Int64}},
    C::Array{<:Real,1},
    order::Int,
    Fil::IndexDict,
    )

    ## Make a Dictionary so that the blocks of B are easy to access
    N₋ = sum(C.<=0)
    N₊ = sum(C.<=0)

    BDict = PartitionedGenerator()

    n_phases = length(C)
    total_n_bases = Int((size(B,1)-N₋-N₊)/n_phases)    

    ppositions = cumsum(C .<= 0)
    qpositions = cumsum(C .>= 0)
    for ℓ in ["+", "-", "0"], m in ["+", "-", "0"]
        for i = 1:n_phases, j = 1:n_phases
            FilBases = repeat(Fil[(ℓ,i)]', order, 1)[:]
            pitemp = falses(N₋)
            qitemp = falses(N₊)
            pjtemp = falses(N₋)
            qjtemp = falses(N₊)
            if C[i] <= 0
                if length(pitemp) > 0 
                    pitemp[ppositions[i]] = Fil[("p"*ℓ,i)][1]
                end
            end
            if C[j] <= 0
                if length(pjtemp) > 0
                    pjtemp[ppositions[j]] = Fil[("p"*m,j)][1]
                end
            end
            if C[i] >= 0
                if length(qitemp) > 0
                    qitemp[qpositions[i]] = Fil[("q"*ℓ,i)][1]
                end
            end
            if C[j] >= 0
                if length(qjtemp) > 0
                    qjtemp[qpositions[j]] = Fil[("q"*m,j)][1]
                end
            end
            i_idx = [
                pitemp
                falses((i - 1) * total_n_bases)
                FilBases
                falses(n_phases * total_n_bases - i * total_n_bases)
                qitemp
            ]
            FjmBases = repeat(Fil[(m, j)]', order, 1)[:]
            j_idx = [
                pjtemp
                falses((j - 1) * order)
                FjmBases
                falses(n_phases * order - j * order)
                qjtemp
            ]
            BDict[((ℓ, m),(i, j))] = B[i_idx, j_idx]
        end
        # below we need to use repeat(Fil[ℓ]', order, 1)[:] to
        # expand the index Fil[ℓ] from cells to all basis function
        FlBases =
            [Fil["p"*ℓ,:]; repeat(Fil[ℓ,:]', order, 1)[:]; Fil["q"*ℓ,:]]
        FmBases =
            [Fil["p"*m,:]; repeat(Fil[m,:]', order, 1)[:]; Fil["q"*m,:]]
        BDict[((ℓ,m),(:,:))] = B[FlBases, FmBases]
    end
    return BDict
end

function materialise(lzB::LazyGenerator)
    B = SparseArrays.SparseMatrixCSC{Float64,Int}(LinearAlgebra.I(size(lzB,1)))*lzB
    BDict = MakeDict(B,lzB.C,size(lzB.D,1),lzB.Fil)
    # BDict = MakeDict(lzB) # in the interest of speed use the above
    return FullGenerator(BDict,B,lzB.Fil)
end

function MakeQBDidx(model::Model,mesh::Mesh)
    ## Make QBD index
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)

    c = N₋
    QBDidx = zeros(Int, NPhases(model) * TotalNBases(mesh) + N₊ + N₋)
    for k = 1:NIntervals(mesh), i = 1:NPhases(model), n = 1:NBases(mesh)
        c += 1
        QBDidx[c] = (i - 1) * TotalNBases(mesh) + (k - 1) * NBases(mesh) + n + N₋
    end
    QBDidx[1:N₋] = 1:N₋
    QBDidx[(end-N₊+1):end] = (NPhases(model) * TotalNBases(mesh) + N₋) .+ (1:N₊)

    return QBDidx
end

include("METools.jl") 
include("polynomials.jl")
include("DGBase.jl")
include("Operators.jl")
include("FVM.jl")
include("FRAPApproximation.jl")
include("Distributions.jl")
include("SimulateSFFM.jl")
include("SFM.jl")
include("Plots.jl")

"""
Construct all the DG operators.

    MakeAll(
        model::SFFM.Model,
        mesh::DGMesh;
        approxType::String = "projection"
    )

# Arguments
- `model`: a model object as output from Model
- `mesh`: a Mesh object
- `approxType::String`: (optional) argument specifying how to approximate R (in
    `MakeR()`)


# Output
- a tuple with keys
    - `Matrices`: see `MakeMatrices`
    - `MatricesR`: see `MakeMatricesR`
    - `B`: see `MakeFullGenerator`
    - `D`: see `MakeD`
    - `DR`: see `MakeDR`
"""
function MakeAll(
    model::SFFM.Model,
    mesh::DGMesh;
    approxType::String = "projection"
)

    # Matrices = MakeMatrices(model, mesh)
    
    B = MakeFullGenerator(model, mesh)
    R = MakeR(model, mesh, approxType = approxType)
    D = MakeD(mesh, B, R)
    return (
        B = B,
        R = R,
        D = D,
    )
end

end


# for k in keys(lzB.Fil), ℓ in keys(lzB.Fil)
#     plus_minus_index = (k[1],ℓ[1])
#     phase_index = (k[2],ℓ[2])
#     display(plus_minus_index); display( phase_index)
# end