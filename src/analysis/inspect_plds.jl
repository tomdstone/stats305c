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

gen_images = false

## Functions

softplus(t) = log1p(exp(t))
positive_diag(v) = diagm(softplus.(v) .+ 1e-4)

if gen_images
    img_folder = "gdrive/images/tom/poisson"
    folder = "gdrive/models/models"
end

const BIN_SIZE = 50
const N_NEURON = 128

# ## Example data

# x = Pickle.npyload("gdrive/models/models/condition_lds_gaussian_cond1_bin$(BIN_SIZE)_state8_iters100_seed7.pkl")
# y = Pickle.npyload("gdrive/models/models/condition_lds_poisson_cond1_bin$(BIN_SIZE)_state8_steps100_seed7.pkl")

# y["fit"]["trainable_params"]

## Generating data

function generate_poisson_data(params; T::Int64, ntrials)
    Σ = positive_diag(params["log_q"])

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
    file = joinpath(folder, "condition_lds_poisson_cond$(cond)_bin$(BIN_SIZE)_state$(state)_steps100_seed7.pkl")

    _x = Pickle.npyload(file)
    params = _x["fit"]["trainable_params"]

    T = size(_x["fit"]["y_test"][1],1)
    ntrials = length(_x["fit"]["y_test"])
    data[[cond, state]] = generate_poisson_data(params; T, ntrials)
end

## Visualizing firing rates

function vis_firing_rates(cond)
    file = joinpath(folder, "condition_lds_poisson_cond$(cond)_bin$(BIN_SIZE)_state8_steps100_seed7.pkl")
    _x = Pickle.npyload(file)

    mean_observed_rates = mean(stack(_x["fit"]["y_test"]), dims=3)[:,:,1]
    p = heatmap(
        BIN_SIZE .* axes(mean_observed_rates, 1),
        1:N_NEURON,
        mean_observed_rates',
        title = "Mean rate Condition $cond",
        xlabel = "Time (ms)",
        ylabel = "Neuron",
        clims = (0,10)
    )
    # savefig(p, joinpath(img_folder, "cond$cond-rate-truth.pdf"))

    qs = Dict()

    for state in [8,12,18]
        mat = mean(data[[cond, state]], dims=3)[:,:,1]
        qs[state] = heatmap(
            BIN_SIZE * axes(mat, 2),
            1:N_NEURON,
            mat,
            title = "Sim Condition $cond Nstate $state",
            xlabel = "Time (ms)",
            ylabel = "Neuron",
            clims = (0,10)
        )

        # savefig(q, joinpath(img_folder, "cond$cond-state$state-rate.pdf"))
    end

    rate_by_neuron = mean(stack(_x["fit"]["y_test"]), dims=(1,3))[:]

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

    # savefig(r, joinpath(img_folder, "cond$cond-rate-histogram.pdf"))

    # return nothing
    return (; p, qs, r)
end

for cond in 1:8
    gen_images && vis_firing_rates(cond)
end


## Visualizing the transmission matrices

function vis_tr_mat(cond, state)
    file = joinpath(folder, "condition_lds_poisson_cond$(cond)_bin$(BIN_SIZE)_state$(state)_steps100_seed7.pkl")

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
    # savefig(s, joinpath(img_folder, "cond$cond-state$state-evals.pdf"))

    # return nothing
    return (; s)
end

for cond in 1:8, state in [8,12,18]
    gen_images && vis_tr_mat(cond, state)
end

## Look at PCA of the simulated data and compare

function vis_PCA(cond)
    file = joinpath(folder, "condition_lds_poisson_cond$(cond)_bin$(BIN_SIZE)_state8_steps100_seed7.pkl")
    _x = Pickle.npyload(file)

    test_data = reduce(hcat, _x["fit"]["y_test"]')

    background_rates = mean(test_data, dims=2)[:]

    samples = permutedims(reduce(hcat, rand.(Poisson.(background_rates'), size(test_data,2))), (2,1))

    pca = fit(PCA, test_data, maxoutdim = 20)
    pca_sampled = fit(PCA, samples, maxoutdim=20)

    t = plot(
        100cumsum(pca.prinvars / pca.tvar),
        ylims = (0,100),
        xlabel = "Principal Component",
        ylabel = "Percent Variance Explained",
        label = "Test data",
        title = "Proportion of Variance Explained Cond $cond"
    )

    plot!(t,
        100cumsum(pca_sampled.prinvars / pca_sampled.tvar),
        label = "Independent rates",
        color = :black,
        linestyle = :dash
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

    # savefig(u, joinpath(img_folder, "cond$cond-2pcs.pdf"))

    vs = Dict()
    for state in [8,12,18]
        dt = reshape(data[[cond, state]], (N_NEURON, :))
        pca_state = fit(PCA, dt, maxoutdim=20)

        plot!(
            t,
            100cumsum(pca_state.prinvars/ pca_state.tvar),
            label = "$state states"
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

        vs[state] = scatter(
            preds_state[1,:],
            preds_state[2,:],
            alpha = 0.2,
            xlabel = "PC 1",
            ylabel = "PC 2",
            label = nothing,
            title = "Projection onto first two PCs Cond $cond State $state"
        )

        # savefig(v, joinpath(img_folder, "cond$cond-state$state-2pcs.pdf"))
    end

    # savefig(u, joinpath(img_folder, "cond$cond-2pcs-all.pdf"))
    # savefig(t, joinpath(img_folder, "cond$cond-pca.pdf"))

    return (; t, u, vs)

    # return nothing
end

for cond in 1:8
    gen_images && vis_PCA(cond)
end

## Pooled LDS

function pooled_PCA()
    pooled_models = Dict(
        i => Pickle.npyload("gdrive/models/models/pooled_lds_poisson_bin50_state$(i)_steps100_seed7.pkl")
        for i in [8,12,18]
    )

    data_dict = Dict(
        i => generate_poisson_data(
            pooled_models[i]["fit"]["trainable_params"],
            T = 6000 ÷ 50,
            ntrials = 100
        ) for i in [8,12,18]
    )

    ss = Dict()

    for state in [8,12,18]
        file = joinpath(folder, "pooled_lds_poisson_bin$(BIN_SIZE)_state$(state)_steps100_seed7.pkl")

        A = Pickle.npyload(file)["fit"]["trainable_params"]["A"]

        ev = eigen(A)

        ss[state] = scatter(
            real(ev.values),
            imag(ev.values),
            label = nothing,
            title = "Eigenvalues Pooled state $state",
        )
        plot!(
            ss[state],
            cos.(range(0,2π, length=1000)),
            sin.(range(0,2π, length=1000)),
            label = nothing,
            aspectratio=1
        )
        # savefig(s, joinpath(img_folder, "pooled-state$state-evals.pdf"))
    end

    pca = fit(PCA, reduce(hcat, pooled_models[8]["fit"]["y_test"]'), maxoutdim = 20)

    t = plot(
        100cumsum(pca.prinvars / pca.tvar),
        ylims = (0,100),
        xlabel = "Principal Component",
        ylabel = "Percent Variance Explained",
        label = "Ground Truth",
        title = "Proportion of Variance Explained "
    )

    preds = predict(pca, reduce(hcat, pooled_models[8]["fit"]["y_test"]'))

    u = scatter(
        preds[1,:],
        preds[2,:],
        alpha = 0.03,
        label = "Truth",
        xlabel = "PC 1",
        ylabel = "PC 2",
        title = "Projection onto first two PCs"
    )

    savefig(u, joinpath(folder, "pooled-2pcs.pdf"))

    for state in [8,12,18]
        model = pooled_models[state]
        data = data_dict[state]

        dt = reshape(data_dict[state], (N_NEURON, :))
        pca_fit = fit(PCA, dt, maxoutdim = 20)

        plot!(
            t,
            100cumsum(pca_fit.prinvars / pca_fit.tvar),
            label = state
        )

        preds_orig = predict(pca, dt)

        scatter!(
            u,
            preds_orig[1,:],
            preds_orig[2,:],
            label = state,
            alpha = 0.03
        )
    end

    return (; ss, t, u)

    # savefig(t, joinpath(folder, "pooled-pca.pdf"))
    # savefig(u, joinpath(folder, "pooled-2pcs-all.pdf"))
    # return nothing
end

gen_images && pooled_PCA()
