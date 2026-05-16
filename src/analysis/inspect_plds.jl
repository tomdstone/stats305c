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
using JLD2
using Dates

default(fontfamily = "Computer Modern")

## Functions

softplus(t) = log1p(exp(t))
positive_diag(v) = diagm(softplus.(v) .+ 1e-4)

folder = "gdrive/models/models"

## Example data

x = Pickle.npyload("gdrive/models/models/condition_lds_gaussian_cond1_bin50_state8_iters100_seed7.pkl")
y = Pickle.npyload("gdrive/models/models/condition_lds_poisson_cond1_bin50_state8_steps100_seed7.pkl")

y["fit"]["trainable_params"]

## Generating data

function generate_poisson_data(params; T::Int64, ntrials)
    Σ = positive_diag(params["log_q"])
    x0 = rand(MvNormal(params["m0"], positive_diag(params["log_s0"])))

    state = stack(rand(MvNormal(Σ), T, ntrials))
    state[:,1,:] .= rand(MvNormal(params["m0"], positive_diag(params["log_s0"])), ntrials)

    A = params["A"]
    for t in 2:T
        state[:, t, :] .+= A * state[:, t-1, :]
    end

    return rand.(Poisson.(softplus.(stack([params["C"]] .* eachslice(state, dims=3)) .+ params["d"]) .+ 1e-4))
end

data = Dict()

for cond in 1:8, state in [8,12,18]
    file = joinpath(folder, "condition_lds_poisson_cond$(cond)_bin50_state$(state)_steps100_seed7.pkl")

    pkl = Pickle.npyload(file)
    params = pkl["fit"]["trainable_params"]

    T = size(pkl["fit"]["y_test"][1],1)
    ntrials = length(pkl["fit"]["y_test"])
    data[[cond, state]] = generate_poisson_data(params; T, ntrials)
end

## Visualizing firing rates

img_folder = "gdrive/images"

for cond in 1:8
    file = joinpath(folder, "condition_lds_poisson_cond$(cond)_bin50_state8_steps100_seed7.pkl")
    pkl = Pickle.npyload(file)

    mean_observed_rates = mean(stack(pkl["fit"]["y_test"]), dims=3)[:,:,1]
    p = heatmap(
        mean_observed_rates',
        title = "Mean rate Condition $cond",
        xlabel = "Bin",
        ylabel = "Neuron",
        clims = (0,10)
    )
    savefig(p, joinpath(img_folder, "cond$cond-rate-truth.pdf"))

    for state in [8,12,18]
        mat = mean(data[[cond, state]], dims=3)[:,:,1]
        q = heatmap(
            mat,
            title = "Sim Condition $cond Nstate $state",
            xlabel = "Bin",
            ylabel = "Neuron",
            clims = (0,10)
        )
        savefig(q, joinpath(img_folder, "cond$cond-state$state-rate.pdf"))
    end

end

for cond in 1:8
    file = joinpath(folder, "condition_lds_poisson_cond$(cond)_bin50_state8_steps100_seed7.pkl")
    pkl = Pickle.npyload(file)

    rate_by_neuron = mean(stack(pkl["fit"]["y_test"]), dims=(1,3))[:]

    r = histogram(
        rate_by_neuron,
        alpha = 0.5,
        label = "truth",
        title = "Neuron firing rate distribution cond $cond"
    )
    for state in [8,12,18]
        histogram!(
            mean(data[[cond, state]], dims=(2,3))[:],
            label = state,
            alpha = 0.5
        )
    end

    savefig(r, joinpath(img_folder, "cond$cond-rate-histogram.pdf"))
end

## Visualizing the transmission matrices

for cond in 1:8, state in [8,12,18]
    file = joinpath(folder, "condition_lds_poisson_cond$(cond)_bin50_state$(state)_steps100_seed7.pkl")

    A = Pickle.npyload(file)["fit"]["trainable_params"]["A"]

    ev = eigen(A)

    s = scatter(
        real(ev.values),
        imag(ev.values),
        label = nothing,
        title = "Eigenvalues Cond $cond state $state",
    )
    plot!(
        s,
        cos.(range(0,2π, length=1000)),
        sin.(range(0,2π, length=1000)),
        label = nothing,
        aspectratio=1
    )
    savefig(s, joinpath(img_folder, "cond$cond-state$state-evals.pdf"))
end
