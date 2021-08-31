using SymPy, Plots
include("../SFFM.jl")

tempCMEParams = Dict()
open("src/CMEParamsData/iltcme.json", "r") do f
    global tempCMEParams
    tempCMEParams=JSON.parse(f)  # parse and transform data
end

s = SymPy.Sym("s")
c = SymPy.Sym("c")
a = SymPy.Sym("a")
b = SymPy.Sym("b")
k = SymPy.Sym("k")
omega = SymPy.Sym("omega")

term0 = c/(1+s)
termk = (a*(1+s) + b*k*omega) / ((1+s)^2 + (k*omega)^2)

d1term0 = -SymPy.diff(term0,s,1)
d1termk = -SymPy.diff(termk,s,1)
d5term0 = -SymPy.diff(term0,s,5)
d5termk = -SymPy.diff(termk,s,5)

keysArray = []
m0Array = []
m1Array = []
m5Array = []
for key in keys(SFFM.CMEParams)
    if key < 200
        print(key); print(",")
        cVal = SFFM.CMEParams[key]["c"]
        omegaVal = SFFM.CMEParams[key]["omega"]
        aVec = SFFM.CMEParams[key]["a"]
        bVec = SFFM.CMEParams[key]["b"]
        n = length(aVec)

        m0 = N(term0(s=>0, c=>cVal, omega=>omegaVal))
        m1 = N(d1term0(s=>0, c=>cVal, omega=>omegaVal))
        m5 = N(d5term0(s=>0, c=>cVal, omega=>omegaVal))
        push!(keysArray, key)
        for kVal in 1:n
            m0 += N(termk(
                s=>0, 
                c=>cVal, omega=>omegaVal, 
                a=>aVec[kVal], b=>bVec[kVal],
                k=>kVal
            ))
            m1 += N(d1termk(
                s=>0, 
                c=>cVal, omega=>omegaVal, 
                a=>aVec[kVal], b=>bVec[kVal],
                k=>kVal
            ))
            m5 += N(d5termk(
                s=>0, 
                c=>cVal, omega=>omegaVal, 
                a=>aVec[kVal], b=>bVec[kVal],
                k=>kVal
            ))
        end
        push!(m0Array, m0)
        push!(m1Array, m1)
        push!(m5Array, m5)
    end
end

display(scatter(keysArray, m0Array))
display(scatter(keysArray, m1Array))
display(scatter(keysArray, log.((m5Array).^(1/5)-m1Array)))