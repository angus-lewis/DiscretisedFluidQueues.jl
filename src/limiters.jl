
"""
minmod(a::Float64,b::Float64,c::Float64)

If the signs of `a, b, c` are the same return the smallest (closest to 0.0), 
otherwise, return 0.0.
"""
function minmod(a::Float64,b::Float64,c::Float64)
    s = (Int.(sign(a))+Int.(sign(b))+Int.(sign(c)))÷3

    if (abs(s)===1)
        return sign(a)*min(abs(a),abs(b),abs(c))
    else 
        return 0.0
    end
end

function minmodB(a::Float64,b::Float64,c::Float64,M,h)
    b = b+M*h^2*sign(b)
    c = c+M*h^2*sign(c)
    
    return minmod(a,b,c)
end

"""
linear(cell_coeffs::Vector{Float64}, V::Matrix{Float64}, Vinv::Matrix{Float64},
w::Float64, Δ::Float64)

Take coefficients legendre basis polynomials and project them down to a linear polynomial.
"""
function linear(cell_coeffs::Vector{Float64}, V::Matrix{Float64}, Vinv::Matrix{Float64},
    D::Matrix{Float64}, w::Vector{Float64}, Δ::Float64)

    # reweight to canonical legendre basis
    # convert to lagrange basis
    cell_coeffs = Vinv*(cell_coeffs*2.0./(Δ*w))
    # project to order 1 
    if length(cell_coeffs)>2
        cell_coeffs[3:end] .= 0.0
    end
    # while we're here, also get the slope of the linear function
    slope = (D*cell_coeffs)[1]*2.0/Δ
    # convert back to original basis
    cell_coeffs = (V*cell_coeffs)./(2.0./(Δ*w))

    return cell_coeffs, slope
end

function limit(coeffs::Vector{Float64}, V::Matrix{Float64}, Vinv::Matrix{Float64},
    D::Matrix{Float64}, w::Vector{Float64}, Δvec::AbstractVector{Float64}, 
    cell_nodes_matrix::Matrix{Float64}, num_phases::Int, n₋::Int, n₊::Int)

    # V, Vinv, D, w = vandermonde(n_bases_per_cell(d.dq))

    limited_coeffs = copy(coeffs)

    limited_coeffs = limited_coeffs[n₋+1:end-n₊]

    limited_coeffs = reshape(limited_coeffs,length(w),num_phases,length(Δvec))

    cell_averages = sum(limited_coeffs,dims=1)
    cell_averages = reshape(cell_averages,num_phases,length(Δvec))./transpose(Δvec)

    poly_reconstruct_left = limited_coeffs[1,:,:]*2.0./(Δvec[:]'*w[1])
    poly_reconstruct_right = limited_coeffs[end,:,:]*2.0./(Δvec[:]'*w[end])

    
    for phase in 1:num_phases
        for cell in 1:length(Δvec)
            a = cell_averages[phase,cell] - poly_reconstruct_left[phase,cell]
            b = cell_averages[phase,cell] - cell_averages[phase,max(1,cell-1)]
            c = cell_averages[phase,min(length(Δvec),cell+1)] - cell_averages[phase,cell]

            left_limited_flux = cell_averages[phase,cell] - minmod(a,b,c)

            a = poly_reconstruct_right[phase,cell] - cell_averages[phase,cell]

            right_limited_flux = cell_averages[phase,cell] + minmod(a,b,c)

            needs_limiting = !(isapprox(left_limited_flux,poly_reconstruct_left[phase,cell];atol=1e-8) &&
                isapprox(right_limited_flux,poly_reconstruct_right[phase,cell];atol=1e-8))
            if needs_limiting
                cell_coeffs = limited_coeffs[:,phase,cell]
                cell_coeffs, slope = linear(cell_coeffs,V,Vinv,D,w,Δvec[cell])

                nodes = cell_nodes_matrix[:,cell] .- (cell_nodes_matrix[1,cell]+cell_nodes_matrix[end,cell])/2.0

                limited_coeffs[:,phase,cell] = (cell_averages[phase,cell] .+ 
                    (nodes*minmod(slope,b/Δvec[cell],c/Δvec[cell])))./(2.0./(Δvec[cell]*w))
            end
        end
    end
    return [coeffs[1:n₋]; limited_coeffs[:]; coeffs[end-n₊+1:end]]
end

function limit(d::SFMDistribution{DGMesh{T}}) where T
    V, Vinv, D, w = vandermonde(n_bases_per_cell(d.dq))
    
    limited_coeffs = limit(d.coeffs[:],V,Vinv,D,w,Δ(d.dq),cell_nodes(d.dq),n_phases(d.dq),N₋(d.dq),N₊(d.dq))
    return SFMDistribution(limited_coeffs,d.dq) 
end

# abstract type AbstractLimiter end 

struct Limiter #<: AbstractLimiter
    fun::Function               # the limiter function
    generate_params::Function   # a function to map a DiscretisedFluidQueue to extra parameters for the limiter
end

muscl_params(dq::DiscretisedFluidQueue) =
    (vandermonde(n_bases_per_cell(dq))..., Δ(dq), cell_nodes(dq),n_phases(dq),N₋(dq),N₊(dq))

GeneralisedMUSCL = Limiter(limit,muscl_params)

NoLimiter = Limiter(identity,x->())
