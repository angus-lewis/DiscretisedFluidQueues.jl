using Documenter

makedocs(
    modules = DiscretisedFluidQueues,
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "angus-lewis",
    sitename = "DiscretisedFluidQueues.jl",
    pages = Any["index.md"],
)
    # strict = true,
    # clean = true,
    # checkdocs = :exports,

deploydocs(
    repo = "github.com/angus-lewis/DiscretisedFluidQueues.jl.git",
    push_preview = true
)
