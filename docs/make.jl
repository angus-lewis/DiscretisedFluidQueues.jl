using Documenter, DiscretisedFluidQueues

makedocs(
    modules = DiscretisedFluidQueues,
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "angus-lewis",
    sitename = "DiscretisedFluidQueues.jl",
    pages = Any["index.md", 
        "API" => [
        "Examples" => "examples.md", 
        "Models" => "models.md", 
        "Mesh (Discretisations)" => "mesh.md", 
        "Distributions" => "dist.md",
        "Time Integration" => "time_integration.md",
        "Generators" => "generators.md",
        "Simulate" => "sim.md",
        "Tools" => [
        "ME Tools" => "me_tools.md",
        "Polynomial tools" => "poly.md"]]],
)
    # strict = true,
    # clean = true,
    # checkdocs = :exports,

deploydocs(
    repo = "github.com/angus-lewis/DiscretisedFluidQueues.jl.git",
    push_preview = true
)
