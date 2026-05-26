## Setup

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
using Dates

default(fontfamily = "Computer Modern")

## Functions

softplus(t) = log1p(exp(t))
positive_diag(v) = diagm(softplus.(v) .+ 1e-4)

img_folder = "gdrive/images/tom/poisson/pooled_input"
folder = "gdrive/models/models"

const BIN_SIZE = 50
const N_NEURON = 128

## Test loading

_x = Pickle.npyload("gdrive/models/pooled_input_lds_poisson_bin50_state8_steps100_seed7.pkl")
_y = Pickle.npyload("gdrive/models/models/pooled_lds_poisson_bin50_state8_steps100_seed7.pkl")

## Functions for generating data

function gen_poisson_with_input(params, input)
    T, ntrials = size(input)

    Σ = positive_diag(params["log_q"])

    state = stack(rand(MvNormal(Σ), T, ntrials))

    A = params["A"]
    B = params["B"]

    state[:, 1, :] .= rand(MvNormal(params["m0"], positive_diag(params["log_s0"])), ntrials)
    state[:, 1, :] .+= B * input[[1], :]

    for t in 2:T
        state[:, t, :] .+= A * state[:, t-1, :] + B * input[[t], :]
    end

    return rand.(Poisson.(softplus.(stack([params["C"]] .* eachslice(state, dims=3)) .+ params["d"]) .+ 1e-4))
end

## Generating data per condition

Random.seed!(1)

data = Dict()

for state in [2,4,6,8,18]
    fitted = Pickle.npyload("gdrive/models/pooled_input_lds_poisson_bin50_state$(state)_steps100_seed7.pkl")["fit"]

    for cond in 1:8
        data[[cond, state]] = gen_poisson_with_input(fitted["trainable_params"], reduce(hcat, fitted["u_test"][fitted["condition_test"] .== cond]))
    end
end

## Visualizing firing rates

for cond in 1:8,state in [2,4,6,8,18]
    mat = mean(data[[cond, state]], dims=3)[:,:,1]

    q = heatmap(
        BIN_SIZE * axes(mat, 2),
        1:N_NEURON,
        mat,
        title = "Sim Condition $cond Nstate $state",
        xlabel = "Time (ms)",
        ylabel = "Neuron",
        clims = (0,10)
    )
    savefig(q, joinpath(img_folder, "pooled-cond$cond-state$state-rate.pdf"))
end
