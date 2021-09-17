# """
# Add to a figure a plot of a SFM distribution.

#     plot!(p,
#         model::Model,
#         mesh::Mesh,
#         Dist::SFFMDistribution;
#         jitter = 0,
#         kwargs...,
#     )

# # Arguments
# - `p::Plots.Plot{Plots.GRBackend}`: a plot object as initialised by `plot()`
# - `Model`: A Model object
# - `mesh`: A Mesh object from `MakeMesh`
# - `Dist`: A SFFMDistribution object as output from `Coeff2Dist` or `Sims2Dist`
# - `jitter`: the amount to jitter the points (in the x-direction)
# - `kwargs`: keyword args to pass to `plot()`
# """
# function plot!(
#     model::Model,
#     mesh::Mesh,
#     Dist::SFMDistribution;
#     jitter = 0,
#     kwargs...
# )
#     pc = 0
#     qc = 0
#     # yLimValues = (0.0, 0.0)
#     for i = 1:NPhases(model)
#         p = Plots.plot!(
#             Dist.x,
#             Dist.distribution[:, :, i];
#             subplot = i,
#             title = "φ=" * string(i),
#             ylabel = string(typeof(d))[15:end],
#             kwargs...,
#         )
#         if model.C[i] <= 0
#             pc = pc + 1
#             x = [model.Bounds[1,1]]
#             y = [Dist.pm[pc]]
#             p = Plots.scatter!(
#                 x .- jitter/2 .+ jitter*rand(),
#                 y;
#                 subplot = i,
#                 kwargs...,
#             )
#         end
#         if model.C[i] >= 0
#             qc = qc + 1
#             x = [model.Bounds[1,end]]
#             y = [Dist.pm[sum(model.C .<= 0) + qc]]
#             p = Plots.scatter!(
#                 x .- jitter/2 .+ jitter*rand(),
#                 y;
#                 subplot = i,
#                 kwargs...,
#             )
#         end

#     end

#     return p
# end # end plot!

# function plot!(
#     p,
#     model::Model,
#     mesh::Mesh,
#     Dist::SFMDistribution;
#     jitter = 0,
#     kwargs...
# )
#     pc = 0
#     qc = 0
#     # yLimValues = (0.0, 0.0)
#     for i = 1:NPhases(model)
#         p = Plots.bar!(
#             Dist.x[:],
#             Dist.distribution[:, :, i][:];
#             bar_width = Δ(mesh),
#             subplot = i,
#             title = "φ=" * string(i),
#             ylabel = string(typeof(Dist))[15:end],
#             kwargs...,
#         )
#         if model.C[i] <= 0
#             pc = pc + 1
#             x = [model.Bounds[1,1]]
#             y = [Dist.pm[pc]]
#             p = Plots.scatter!(
#                 x .- jitter/2 .+ jitter*rand(),
#                 y;
#                 subplot = i,
#                 kwargs...,
#             )
#         end
#         if model.C[i] >= 0
#             qc = qc + 1
#             x = [model.Bounds[1,end]]
#             y = [Dist.pm[sum(model.C .<= 0) + qc]]
#             p = Plots.scatter!(
#                 x .- jitter/2 .+ jitter*rand(),
#                 y;
#                 subplot = i,
#                 kwargs...,
#             )
#         end

#     end

#     return p
# end # end plot!

# """
# Initialise and plot a SFM distribution.

#     plot(
#         model::Model;
#         mesh::Mesh,
#         dist::NamedTuple{(:pm, :distribution, :x, :type)};
#         jitter = 0,
#         kwargs...,
#     )

# # Arguments
# - `Model`: A Model object
# - `mesh`: A Mesh object from `MakeMesh`
# - `Dist`: A distribution object as output from `Coeff2Dist` or `Sims2Dist`
# - `color`: (optional) a colour specifier for the plot
# - `jitter`: the amount to jitter the points (in the x-direction)
# - `kwargs`: keyword args to pass to `plot()`

# # Output
# - a plot object of type `Plots.Plot{Plots.GRBackend}` with `NPhases` subplots
#     containing a plot of the distribution for each phase.
# """
# function plot(
#     model::Model,
#     mesh::Mesh,
#     dist::SFMDistribution;
#     jitter = 0,
#     kwargs...,
# )
#     p = Plots.plot(layout = Plots.@layout(Plots.grid((NPhases(model)+1)÷2, 2)))
#     p = plot!(
#         p,
#         model,
#         mesh,
#         dist;
#         jitter = jitter,
#         kwargs...,
#     )

#     return p
# end # end plot


# """

#     plot(model::Model; kwargs...,)

# Blank initialiser for a plot of a SFFM model. 

# input:
#  - `model::Model` a object of SFFM model type
#  -  `kwargs`: optional keyword args you want to also pass to plot
# """
# function plot(
#     model::Model;
#     kwargs...,
# )
#     p = Plots.plot(layout = Plots.@layout(Plots.grid((NPhases(model)+1)÷2, 2)))
#     p = Plots.plot!(;kwargs)

#     return p
# end # end plot


