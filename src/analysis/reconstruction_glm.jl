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
using KernelDensity
using DSP
using GLM

default(fontfamily = "Computer Modern")

## Loading data

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

## Smoothing

f = DSP.Windows.gaussian(41, 0.015)

# d = data["spike"][1]

# permutedims(filtfilt(f, permutedims(d, (2,1,3))), (2,1,3))



function reshape_for_regression(nxtxr)
    return reshape(permutedims(nxtxr, (2,1,3)), size(nxtxr,2), :)
end

function unshape_from_regression(tx_nxr, nneuron, ntrial)
    return permutedims(reshape(tx_nxr, :, nneuron, ntrial), (2,1,3))
end


function setup_target_matrix(data_tensor)
    return filtfilt(DSP.Windows.gaussian(41, 0.015), reshape_for_regression(data_tensor))
end

function setup_design_matrix(force_tensor)
    mean_force = mean(force_tensor, dims=3)[:]
    derv_force = vcat([0.], mean_force[2:end] - mean_force[1:end-1])

    mean_force .= mean_force .- mean(mean_force) / std(mean_force)
    derv_force .= derv_force .- mean(derv_force) / std(derv_force)

    return hcat(
        ones(length(mean_force)),
        mean_force,
        # vcat(mean_force[[1,1]], mean_force[1:end-2]),
        # vcat(mean_force[3:end], mean_force[[end,end]]),
        derv_force,
        # vcat(derv_force[[1,1]], derv_force[1:end-2]),
        # vcat(derv_force[3:end], derv_force[[end,end]]),
    )
end



# regress_on_matrix(setup_design_matrix(data["force"][1]), setup_target_matrix(data["spike"][1]))

function regress_on_matrix(design, target)
    fits = [
        lm(design, col) for col in eachcol(target)
    ]

    return (
        stack([coeftable(f).cols[4] for f in fits]),
        stack([coef(f) for f in fits]),
        stack([[r2(f)] for f in fits])
    )
end


pvals = Dict()
coefs = Dict()
r2s = Dict()

for cond in 1:8
    # results = nshape_from_regression(
    results = regress_on_matrix(
            setup_design_matrix(data["force"][cond]),
            setup_target_matrix(data["spike"][cond]),
        )
        # size(data["spike"][i],1),
        # size(data["spike"][i],3),
    # )

    pvals[cond] = unshape_from_regression(
        results[1],
        size(data["spike"][cond],1),
        size(data["spike"][cond],3)
    )

    coefs[cond] = unshape_from_regression(
        results[2],
        size(data["spike"][cond],1),
        size(data["spike"][cond],3)
    )

    r2s[cond] = unshape_from_regression(
        results[3],
        size(data["spike"][cond],1),
        size(data["spike"][cond],3)
    )
end


h = histogram(r2s[1][:], label = nothing, dpi = 600, title = "Force encoding R-squared values")

## Doing GLM on each force trial, not just the average across trials

r2s_dict = Dict()

f = DSP.Windows.gaussian(41, 0.015)

f ./= sum(f)

for c in 1:8
    spike_data = data["spike"][c]
    force_data = data["force"][c]

    nneuron, ntime, ntrial = size(spike_data)

    r2_vals = Array{Float64}(undef, nneuron, ntrial)

    for trial in 1:ntrial
        force = filtfilt(f, force_data[1,:,trial])

        design_matrix = hcat(ones(ntime), force, vcat([0], force[2:end]-force[1:end-1]))

        for neuron in 1:nneuron
            signal = filtfilt(f, spike_data[neuron, :, trial])

            r2_vals[neuron, trial] = r2(lm(design_matrix, signal))

        end
    end
    r2s_dict[c] = r2_vals
end

neuron_r2s = replace(mean(r2s_dict[7], dims=2)[:],
    # NaN => 1. # uncomment for argmin computation
    NaN => -1.
)

argmax(neuron_r2s)

argmin(neuron_r2s)

maximum(neuron_r2s)


u = errorline(
    50 * (1:size(data["spike"][7],2)),
    20filtfilt(f, data["spike"][7][argmax(neuron_r2s), :, :]),
    errorstyle=:plume,
    label = nothing,
    groupcolor = [:black],
    secondarycolor = [:blue],
    xlabel = "Time (ms)",
    ylabel = "Firing rate (Hz)",
    title = "Neuron with r^2 = 0.698",
    dpi = 600
)


uu = errorline(
    50 * (1:size(data["spike"][7],2)),
    filtfilt(f, data["force"][7][1, :, :]),
    errorstyle=:plume,
    secondarycolor = [:red],
    groupcolor = [:black],
    label = nothing,
    title = "Force Profile",
    xlabel = "Time (ms)",
    ylabel = "Force (N)",
    dpi = 600
)


v = errorline(
    50 * (1:size(data["spike"][7],2)),
    20filtfilt(f, data["spike"][7][1, :, :]),
    errorstyle=:plume,
    label = nothing,
    groupcolor = [:black],
    secondarycolor = [:green],
    xlabel = "Time (ms)",
    ylabel = "Firing rate (Hz)",
    title = "Neuron with r^2 = 0.0357",
    dpi = 600
)

savefig(u, "gdrive/images/tom/force_encoding/maximal_neuron.png")
savefig(uu, "gdrive/images/tom/force_encoding/force_profile.png")
savefig(v, "gdrive/images/tom/force_encoding/minimal_neuron.png")
