## Packages

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

## Loading data

p = Pickle.npyload("gdrive/mc_pacman.pkl")

## Data processing

# TODO: need to split into training and test sets

function assemble_data(; p = p, Δt = 20, T = Float64)
    data = Dict()

    for i in 1:8
        logics = p["condition"] .== i
        datasize = (size(p["spikes"][findfirst(logics)])..., sum(logics))

        N = datasize[2] ÷ Δt

        data[i] = Array{T}(undef, datasize[1], N, datasize[3])

        for (j, ind) in enumerate(findall(logics))
            mat = p["spikes"][ind]

            for k in 0:N-1
                data[i][:, k+1, j] .= Matrix{T}(sum(mat[:, (Δt * k + 1):(Δt * (k+1))], dims=2))
            end
        end
    end

    return data
end

##
