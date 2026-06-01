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

p = Pickle.npyload("gdrive/mc_pacman.pkl")

raw_data = assemble_data(; p, Δt = 1)
data = assemble_data(; p, Δt = 50)


# data is {condition => neuron x time x trial}

## Z-scoring

function zscore_data(data_array)
    x = (sqrt.(data_array) .- (mean(data_array, dims=(2,3)))) ./ (std(data_array, dims=(2,3)))
    replace!(x, Inf => 0., -Inf => 0., NaN => 0.)

    return x
end

## Basic PCA plots

function view_PCA(M::AbstractMatrix)
    pca = fit(PCA, M, maxoutdim=20)

    preds = predict(pca, M)

    x = preds[1,:]
    y = preds[2,:]

    p = scatter(
        x,
        y,
        label = nothing
    )
    q = plot(
        100cumsum(pca.prinvars) / pca.tvar,
        ylims = (0,100),
        label = "Cumulative Variance"
    )
    plot!(
        q,
        1:20,
        (1:20) * (100 / minimum(size(M))),
        label = "Diagonal"
    )

    return plot(p, q, dpi = 600)
end

function view_PCA(T::Array{S,3}) where S
    return view_PCA(reshape(T, (size(T,1), :)))
end

## Plotting

i=1

p = view_PCA(zscore_data(data["spike"][i]))
q = view_PCA(data["spike"][i])

plot!(p, suptitle = "Z-scored data")
plot!(q, suptitle = "Unnormalized data")

background_rates = mean(data["spike"][i], dims=(2,3))[:]
samples = permutedims(reduce(hcat, rand.(Poisson.(background_rates'), size(reshape(data["spike"][i], (128,:)),2))), (2,1))

r = view_PCA(zscore_data(samples))
plot!(r, suptitle = "Z-scored independent sampled data")

s = plot(p, q, r, layout = (3,1), size = (1000, 1200), suptitle = i)

i += 1
s

## Fitting with SSD.jl

zscored_data = Dict(i => zscore_data(data["spike"][i]) for i in 1:8)

function init_LDS(statedim, obsdim=128)
    state_model = GaussianStateModel(;
        A=randn(statedim, statedim),
        Q = I(statedim) |> Matrix{Float64},
        b = zeros(statedim),
        x0 = zeros(statedim),
        P0 = I(statedim) |> Matrix{Float64}
    )

    obs_model = GaussianObservationModel(;
        C = randn(obsdim, statedim),
        R = I(obsdim) |> Matrix{Float64},
        d = zeros(obsdim)
    )

    model = LinearDynamicalSystem(
        state_model,
        obs_model,
        statedim,
        obsdim,
        repeat([true], 6)
    )

    return model
end

models = Dict([cond, statedim] => init_LDS(statedim) for cond in 1:8, statedim in [2,4,6,8])

# fit!(models[[1,8]], data["spike"][1], tol=1e-3)

# Trying to brute force way around the failure of Positive definiteness in Cholesky decomp of R.

# for cond in 1:8
#     for nstate in [2,4,6,8]

#         for i in 1:100
#             try
#                 fit!(models[[cond, nstate]], data["spike"][cond], max_iter = 1)
#                 println("$cond $nstate: $i ✓")
#             catch e
#                 # println(e)
#                 println("$cond $nstate: $i ×")
#                 R = models[[cond, nstate]].obs_model.R
#                 R = Symmetric(0.5 * (R + R'))
#                 re = eigen(R)
#                 models[[cond, nstate]].obs_model.R = Matrix(Eigen(max.(1e-10, re.values), re.vectors))
#             end

#         end
#     end
# end

function store_ssd(m, file)
    s = m.state_model
    o = m.obs_model
    params = Dict(
        "A" => s.A[:],
        "Q" => s.Q[:],
        "b" => s.b,
        "x0"=> s.x0,
        "P0"=> s.P0[:],
        "C" => o.C[:],
        "R" => o.R[:],
        "d" => o.d
    )
    Pickle.store(file, params)
end

function load_ssd(cond, nstate)
    file = "gdrive/models/SSD/cond$cond-state$nstate.pkl"
    params = Pickle.load(file)

    obsdim = length(params["d"])

    state_model = GaussianStateModel(
        A = reshape(params["A"], nstate, nstate) |> Matrix{Float64},
        Q = reshape(params["Q"], nstate, nstate) |> Matrix{Float64},
        b = params["b"] |> Vector{Float64},
        x0 = params["x0"] |> Vector{Float64},
        P0 = reshape(params["P0"], nstate, nstate) |> Matrix{Float64}
    )

    obs_model = GaussianObservationModel(
        C = reshape(params["C"], obsdim, nstate) |> Matrix{Float64},
        R = reshape(params["R"], obsdim, obsdim) |> Matrix{Float64},
        d = params["d"] |> Vector{Float64}
    )

    return LinearDynamicalSystem(
        state_model,
        obs_model,
        nstate,
        obsdim,
        repeat([true], 6)
    )
end

for cond in 1:8, nstate in [2,4,6,8]
    file = "gdrive/models/SSD/cond$cond-state$nstate.pkl"
    m = models[[cond, nstate]]
    store_ssd(m, file)
end

##

loaded_models = Dict([cond, state] => load_ssd(cond, state) for cond in 1:8, state in [2,4,6,8])


xsmooth, psmooth = StateSpaceDynamics.smooth(loaded_models[[1,2]], zscored_data[1])

smoothed_estimates = Dict(
    [cond, state] => StateSpaceDynamics.smooth(loaded_models[[cond, state]], zscored_data[cond]) for
    cond in 1:8, state in [2,4,6,8]
)

preds = Dict()

for cond in 1:8, state in [2,4,6,8]
    data_matrix = reshape(smoothed_estimates[[cond, state]][1], state, :)

    pca = fit(PCA, data_matrix, maxoutdim = 2)
    preds[[cond, state]] = reshape(predict(pca, data_matrix), :, size(smoothed_estimates[[cond, state]][1])[[2,3]]...)


end


function plot_curve_with_color(x,y)
    plot!(x, y, line_z = 1:length(x), color = :rainbow, colorbar = nothing, label = nothing)
end

for cond in 1:8, state in [2,4,6,8]
    plot(
        xlabel = "PC 1",
        ylabel = "PC 2",
        title = "Latents in condition $cond: $state states",
        dpi = 600
    )

    for trial in axes(preds[[cond, state]],3)
        plot_curve_with_color(preds[[cond, state]][1,:,trial], preds[[cond, state]][2,:,trial])
    end

    savefig("gdrive/images/tom/zscored/zscored_gaussian_latents_state$(state)_cond$(cond).pdf")
end
