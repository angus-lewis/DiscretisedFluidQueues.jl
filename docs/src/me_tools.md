# ME Tools 

### Types
```@docs 
AbstractMatrixExponential
ConcentratedMatrixExponential
MatrixExponential 
cme_params
```
### Methods
```@docs 
build_me
pdf(a::Array{Float64,2}, me::AbstractMatrixExponential)
ccdf(a::Array{Float64,2}, me::AbstractMatrixExponential)
cdf(a::Array{Float64,2}, me::AbstractMatrixExponential)
```