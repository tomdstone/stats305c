## Setup

include("utilities.jl")
include("../analysis/inspect_plds.jl")
default(fontfamily = "Computer Modern")
using KernelDensity

## Data Loading

raw_data = Pickle.npyload("gdrive/mc_pacman.pkl")

# yes doing Δt = 1 is stupid but the code is already there
raster_data = assemble_data(; p = raw_data, Δt = 1)

_data = assemble_data(; p = raw_data, Δt = 50)

## Firing rate KDE

get_x_vals(cart_ind_array) = [pt.I[1] for pt in cart_ind_array]

# kde(get_x_vals(findall(raster_data["spike"][1][9,:,:] .== 1 )), boundary = (1, 6001)) |> plot
# histogram(get_x_vals(findall(raster_data["spike"][1][1,:,:] .== 1 )), boundary = (1, 6001))

function vis_kde_rates(cond, sz=(500,400))
    dat = raster_data["spike"][cond]

    tmax = size(dat, 2)
    p = plot(
        title = "Smoothed neuron firing rates Condition $cond",
        xlims = (0, tmax),
        xlabel = "Time (ms)",
        ylims = (0, Inf),
        ylabel = "Firing rate (Hz)",
        rightmargin = 5Plots.Measures.mm,
        size = sz,
    )
    for neuron in axes(dat, 1)
        k = kde(get_x_vals(findall(dat[neuron, :, :] .== 1.)), boundary = (1,tmax))

        # coeff_ = sum(dat[neuron, :, :]) * 1000 / size(dat,3) # same as below
        coeff_ = sum(mean(dat[neuron, :, :], dims=2)) * 1000

        plot!(p, k.x, k.density * coeff_, label = nothing)
    end
    # savefig(p, "gdrive/images/tom/neuron_kdes/neuron_firing_rate_kde_cond$cond.pdf")

    return p
end

# total = 0
# for neuron in axes(dat, 1)
#     k = kde(get_x_vals(findall(dat[neuron, :, :] .== 1.)), boundary = (1,tmax))

#     coeff = sum(dat[neuron, :, :]) * 1000 / size(dat,3)
#     coeff = sum(mean(dat[neuron, :, :], dims=3)) * 1000


#     plot!(p, k.x, k.density * coeff, label = nothing)

#     total += sum(k.density * coeff * (k.x[2] - k.x[1]) / 1000)
# end
# println(total / sum(dat))




## Raster plots

function raster_plot(neuron_x_time)
    return heatmap(
        # axes(neuron_x_time, 2),
        # axes(neuron_x_time, 1),
        neuron_x_time,
        c = cgrad([:white, :black]),
        clims = (0,1),
        xlabel = "Time (ms)",
        ylabel = "Neuron",
        yticks = 0:32:128,
        # ylims = (0, Inf),
        # xtick_width = 0,
        cbar = false,
        rightmargin = 5Plots.Measures.mm
    )
end




## Firing rate noise in PCA



## Selected images

p = vis_kde_rates(7, (700, 500))
plot!(p, dpi = 600)
savefig(p, "gdrive/final/images/kde_firing_rates_cond7.png")

q = raster_plot(raster_data["spike"][7][:,:,3])
plot!(q, title = "Condition 7, Trial 3", dpi=600)
savefig(q, "gdrive/final/images/raster_plot_cond7_trial3.png")

t = vis_PCA(7).t
plot!(t, dpi =600, title = "PCA Variance Explianed, Condition 7")
savefig(t, "gdrive/final/images/pca_var_explained_cond7.png")
