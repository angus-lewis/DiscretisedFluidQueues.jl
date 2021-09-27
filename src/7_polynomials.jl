"""
    lagrange_polynomials(nodes::Array{Float64, 1}, evalPt::Float64)

Evaluate the lagrange polynomials defied by `nodes` at the point `evalPt`.
"""
function lagrange_polynomials(nodes::Array{Float64,1}, evalPt::Float64)
    order = length(nodes)
    poly_coefs = zeros(order)
    for n in 1:order
        notn = [1:n-1;n+1:order]
        poly_coefs[n] = prod(evalPt.-nodes[notn])./prod(nodes[n].-nodes[notn])
    end
    return poly_coefs
end

"""
    gauss_lobatto_points(a::Float64, b::Float64, n_evals::Int)

Return array containing `n_evals` gauss lobatto points in the interval `[a,b]`.
"""
function gauss_lobatto_points(a::Float64,b::Float64,n_evals::Int)
    (b<a)&&throw(DomainError("must have a<b"))
    (n_evals<1)&&throw(DomainError("n_evals must be > 0"))
    if n_evals > 1
        # the GL nodes
        # nodes in [-1,1]
        nodes = Jacobi.zglj(n_evals, 0, 0)
        # shift nodes to [a,b]
        nodes *= 0.5*(b-a)
        nodes .+= 0.5*(a+b)
    else
        nodes = [0.5*(a+b)]
    end
    return nodes
end

"""
    gauss_lobatto_weights(a::Float64, b::Float64, n_evals::Int)

Return array containing `n_evals` weights of the polynomials associated with the
gauss lobatto points in the interval `[a,b]`. i.e. the itegrals over the interval `[a,b]` 
of the lagrange polynomials defined by the nodes given by `nodes = gauss_lobatto_points(a,b,n_evals)`
"""
function gauss_lobatto_weights(a::Float64,b::Float64,n_evals::Int)
    (b<a)&&throw(DomainError("must have a<b"))
    (n_evals<1)&&throw(DomainError("n_evals must be > 0"))
    if n_evals > 1
        nodes = Jacobi.zglj(n_evals, 0, 0)
        weights = Jacobi.wglj(nodes,0,0)*(b-a)/2
    else
        weights = [(b-a)]
    end
    return weights
end

"""
    lagrange_interpolation(fun::Function, a::Float64, b::Float64, n_evals::Int)

Return a polynomial approximation to `fun` of order `n_evals-1` on the interval `[a,b]` using 
the `gauss_lobatto_points` as the nodes of the lagrange polynomials.
"""
function lagrange_interpolation(fun::Function,a::Float64,b::Float64,n_evals::Int)
    nodes = gauss_lobatto_points(a,b,n_evals)
    
    fun_vals = fun.(nodes)
    interpolant(x) = LinearAlgebra.dot(fun_vals,lagrange_polynomials(nodes,x))
    return interpolant
end

"""
    gauss_lobatto_quadrature(fun::Function, a::Float64, b::Float64, n_evals::Int)

Compute a quadrature approximation of `fun` on the interval `[a,b]` with `n_evals` function 
evaluations using the `gauss_lobatto_points`
"""
function gauss_lobatto_quadrature(fun::Function,a::Float64,b::Float64,n_evals::Int)
    nodes = gauss_lobatto_points(a,b,n_evals)
    weights = gauss_lobatto_weights(a,b,n_evals)
    fun_vals = fun.(nodes)
    quad = LinearAlgebra.dot(fun_vals,weights)
    
    return quad
end

"""
Construct a generalised vandermonde matrix.

    vandermonde( nBases::Int)

Note: requires Jacobi package Pkg.add("Jacobi")

# Arguments
- `nBases::Int`: the degree of the basis

# Output
- a tuple with keys
    - `:V::Array{Float64,2}`: where `:V[:,i]` contains the values of the `i`th
        legendre polynomial evaluate at the GLL nodes.
    - `:inv`: the inverse of :V
    - `:D::Array{Float64,2}`: where `V.D[:,i]` contains the values of the derivative
        of the `i`th legendre polynomial evaluate at the GLL nodes.
"""
function vandermonde(nBases::Int)
    if nBases > 1
        z = Jacobi.zglj(nBases, 0, 0) # the LGL nodes
    elseif nBases == 1
        z = 0.0
    end
    V = zeros(Float64, nBases, nBases)
    DV = zeros(Float64, nBases, nBases)
    if nBases > 1
        for j = 1:nBases
            # compute the polynomials at gauss-labotto quadrature points
            V[:, j] = Jacobi.legendre.(z, j - 1) .* sqrt((2 * (j - 1) + 1) / 2)
            DV[:, j] = Jacobi.dlegendre.(z, j - 1) .* sqrt((2 * (j - 1) + 1) / 2)
        end
        # Compute the Gauss-Lobatto weights for numerical quadrature
        w =
            2.0 ./ (
                nBases *
                (nBases - 1) *
                Jacobi.legendre.(Jacobi.zglj(nBases, 0, 0), nBases - 1) .^ 2
            )
    elseif nBases == 1
        V .= [1/sqrt(2)]
        DV .= [0]
        w = [2]
    end
    return (V = V, inv = inv(V), D = DV, w = w)
end

function legendre_to_lagrange(coeffs)
    order = length(coeffs)
    V = vandermonde(order)
    return V.V*coeffs
end