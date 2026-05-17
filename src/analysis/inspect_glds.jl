## Environment and packages

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

ENV["JULIA_PYTHONCALL_EXE"] = "/Users/tomstone/miniforge3/envs/dynamax_env/bin/python"
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using PythonCall

pkl = pyimport("pickle")

## Generating data

const BIN_SIZE = 50

p2m(x) = pyconvert(Matrix{Float64}, x)
p2v(x) = pyconvert(Vector{Float64}, x)

function params_from_gaussian(x)
    params = x["fit"]["params"]
    Σ0 = p2m(params.initial.cov)
    x0 = p2v(params.initial.mean)

    Σ = p2m(params.dynamics.cov)
    b = p2v(params.dynamics.bias)
    A = p2m(params.dynamics.weights)

    C = p2m(params.emissions.weights)
    d = p2v(params.emissions.bias)
    R = p2m(params.emissions.cov)

    return (; Σ0, x0, Σ, A, b, C, d, R)
end

data_from_gaussian_pickle(p) = permutedims(stack(pyconvert(Vector{Matrix{Float64}}, p)), (2,1,3))

function generate_gaussian_data(x; T::Int64, ntrials::Int64)
    Σ0, x0, Σ, A, b, C, d, R = params_from_gaussian(x)

    state = stack(rand(MvNormal(b, Σ), T, ntrials))
    state[:, 1, :] .= rand(MvNormal(x0, Σ0), ntrials)
    obs = stack(rand(MvNormal(d, R), T, ntrials))

    for t in 2:T
        state[:, t, :] .+= A * state[:, t-1, :]
        obs[:, t, :] .+= C * state[:, t, :]
    end

    return obs, state
end

## Loading in Gaussian SSM objects


params = Dict()
y_train = Dict()
y_test = Dict()


for cond in 1:8, state in [8,12,18]
    file = "gdrive/models/models/condition_lds_gaussian_cond$(cond)_bin$(BIN_SIZE)_state$(state)_iters100_seed7.pkl"

    _x = pywith(pkl.load, open(file, "r"))
    params[[cond, state]] = params_from_gaussian(_x)
    y_train[[cond, state]] = data_from_gaussian_pickle(_x["fit"]["y_train"])
    y_test[[cond, state]] = data_from_gaussian_pickle(_x["fit"]["y_test"])
end

## Generating data

obsdata = Dict()
statedata = Dict()

for cond in 1:8, state in [8,12,18]
    file = "gdrive/models/models/condition_lds_gaussian_cond$(cond)_bin$(BIN_SIZE)_state$(state)_iters100_seed7.pkl"

    _x = pywith(pkl.load, open(file, "r"))

    ntrials = length(_x["fit"]["y_test"])
    T = length(_x["fit"]["y_test"][1])

    obsdata[[cond, state]], statedata[[cond, state]] = generate_gaussian_data(_x; T, ntrials)
end

## Visualizing firing rates

folder = "gdrive/models/models"
img_folder = "gdrive/images/tom/gaussian"

for cond in 1:8
    file = joinpath(folder, "condition_lds_gaussian_cond$(cond)_bin$(BIN_SIZE)_state8_iters100_seed7.pkl")
    x = pywith(pkl.load, open(file, "r"))

    mean_observed_rates = mean(permutedims(stack(pyconvert(Vector{Matrix{Float64}}, x["fit"]["y_test"])), (2,1,3)), dims=3)[:,:,1]
    p = heatmap(
        BIN_SIZE .* axes(mean_observed_rates, 2),
        1:128,
        mean_observed_rates,
        title = "Mean rate Condition $cond",
        xlabel = "Time (ms)",
        ylabel = "Neuron",
        clims = (0,2)
    )
    savefig(p, joinpath(img_folder, "cond$cond-rate-truth.pdf"))

    for state in [8,12,18]
        mat = mean(obsdata[[cond, state]], dims=3)[:,:,1]
        q = heatmap(
            BIN_SIZE * axes(mat, 2),
            1:128,
            mat,
            title = "Sim Condition $cond Nstate $state",
            xlabel = "Time (ms)",
            ylabel = "Neuron",
            clims = (0,2)
        )
        savefig(q, joinpath(img_folder, "cond$cond-state$state-rate.pdf"))
    end
end

for cond in 1:8
    file = joinpath(folder, "condition_lds_gaussian_cond$(cond)_bin$(BIN_SIZE)_state8_iters100_seed7.pkl")
    x = Pickle.npyload(file)

    rate_by_neuron = mean(stack(x["fit"]["y_test"]), dims=(1,3))[:]

    r = histogram(
        rate_by_neuron,
        alpha = 0.5,
        label = "truth",
        title = "Neuron firing rate distribution cond $cond"
    )
    for state in [8,12,18]
        histogram!(
            mean(obsdata[[cond, state]], dims=(2,3))[:],
            label = state,
            alpha = 0.5
        )
    end

    savefig(r, joinpath(img_folder, "cond$cond-rate-histogram.pdf"))
end

## Visualizing the transmission matrices

for cond in 1:8, state in [8,12,18]
    file = joinpath(folder, "condition_lds_gaussian_cond$(cond)_bin$(BIN_SIZE)_state$(state)_iters100_seed7.pkl")

    A = params_from_gaussian(pywith(pkl.load, open(file, "r"))).A

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


## Look at PCA of the simulated data and compare

for cond in 1:8
    file = joinpath(folder, "condition_lds_gaussian_cond$(cond)_bin$(BIN_SIZE)_state8_iters100_seed7.pkl")
    x = pywith(pkl.load, open(file, "r"))

    test_data = reduce(hcat, pyconvert(Vector{Matrix{Float64}}, x["fit"]["y_test"])')


    pca = fit(PCA, test_data, maxoutdim = 20)

    t = plot(
        100cumsum(pca.prinvars / pca.tvar),
        ylims = (0,100),
        xlabel = "Principal Component",
        ylabel = "Percent Variance Explained",
        label = "Ground Truth",
        title = "Proportion of Variance Explained Cond $cond"
    )

    preds = predict(pca, test_data)

    u = scatter(
        preds[1,:],
        preds[2,:],
        alpha = 0.2,
        label = "Truth",
        xlabel = "PC 1",
        ylabel = "PC 2",
        title = "Projection onto first two PCs Cond $cond"
    )

    savefig(u, joinpath(img_folder, "cond$cond-2pcs.pdf"))


    for state in [8,12,18]
        dt = reshape(obsdata[[cond, state]], (128, :))
        pca_state = fit(PCA, dt, maxoutdim=20)

        plot!(
            t,
            100cumsum(pca_state.prinvars/ pca_state.tvar),
            label = state
        )

        preds_orig = predict(pca, dt)

        scatter!(
            u,
            preds_orig[1,:],
            preds_orig[2,:],
            label = state,
            alpha = 0.2
        )

        preds_state = predict(pca_state, dt)

        v = scatter(
            preds_state[1,:],
            preds_state[2,:],
            alpha = 0.2,
            xlabel = "PC 1",
            ylabel = "PC 2",
            label = nothing,
            title = "Projection onto first two PCs Cond $cond State $state"
        )

        savefig(v, joinpath(img_folder, "cond$cond-state$state-2pcs.pdf"))
    end

    savefig(u, joinpath(img_folder, "cond$cond-2pcs-all.pdf"))
    savefig(t, joinpath(img_folder, "cond$cond-pca.pdf"))
end
