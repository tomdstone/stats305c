using Pkg
Pkg.activate("Stats305c", shared=true)

using LinearAlgebra
using StateSpaceDynamics
using Pickle
using StatsPlots
using Distributions
using MultivariateStats
using Random
using ProgressBars
using JLD2
using Dates

function assemble_data(; p, Δt = 20, T = Float64)
    data = Dict(
        "spike" => Dict(),
        "force" => Dict()
    )

    for i in 1:8
        logics = p["condition"] .== i
        datasize = (size(p["spikes"][findfirst(logics)])..., sum(logics))

        N = datasize[2] ÷ Δt

        data["spike"][i] = Array{T}(undef, datasize[1], N, datasize[3])
        data["force"][i] = Array{T}(undef, 1,           N, datasize[3])

        for (j, ind) in enumerate(findall(logics))
            mat1 = p["spikes"][ind]
            mat2 = p["force"][ind]

            for k in 0:N-1
                data["spike"][i][:, k+1, j] .= Matrix{T}(sum( mat1[:, (Δt * k + 1):(Δt * (k+1))], dims=2))
                data["force"][i][:, k+1, j] .= Matrix{T}(mean(mat2[:, (Δt * k + 1):(Δt * (k+1))], dims=2))
            end
        end
    end

    return data
end
