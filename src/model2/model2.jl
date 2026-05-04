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


## Loading data

p = Pickle.npyload("gdrive/mc_pacman.pkl")

## Data processing

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

## Utilities for making dynamical systems

function assemble_poglds(; nneuron = 128, nstate = 10, T = Float64)
    pom = PoissonObservationModel(
        randn(T, nneuron, nstate) / nneuron,
        -ones(T, nneuron)
    )

    gsm = GaussianStateModel(
        randn(T, nstate, nstate),
        Matrix{T}(I(nstate)),
        zeros(T, nstate),
        zeros(T, nstate),
        Matrix{T}(I(nstate))
    )

    return LinearDynamicalSystem(gsm, pom, fit_bool = fill(true, 5))
end

## Fitting models

models = Dict(i => assemble_poglds() for i in 1:8)

for i in ProgressBar(1:8)
    fit!(models[i], data[i])
end

## Evaluating model performance
