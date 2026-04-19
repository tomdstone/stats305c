using Pkg
Pkg.activate("Stats305c", shared=true)

using MAT, Pickle, MultivariateStats, Plots, LinearAlgebra

# data source 1
dir = "gdrive/doi-10.7281-t1-v73vza/SMA_Monkey"
x = matread(joinpath(dir, "BothMonkey_SMA_neuron_Info.mat"))
y = matread(joinpath(dir, "MonkeyH/Multi-attribute-Hobbit-01-02-2019.mat"))

# MC Cycle data
z = Pickle.npyload("gdrive/mc_cycle.pkl")

# I just looked through the data objects in my terminal

data_matrix = reduce(hcat, z["spikes"])

w = fit(PCA, data_matrix, maxoutdim = 20)

plot(
    100*cumsum(w.prinvars / w.tvar),
    ylims = (0,100),
    xlabel = "Principal Component",
    ylabel = "Percent Variance Explained",
    label = missing
)


# MC Pacman

p = Pickle.npyload("gdrive/mc_pacman.pkl")

data_matrix2 = Matrix{Float64}(reduce(hcat, p["spikes"]) )

# PCA

w2 = fit(PCA, data_matrix2, maxoutdim=20)

plot(
    100*cumsum(w2.prinvars / w2.tvar),
    ylims = (0,100),
    xlabel = "Principal Component",
    ylabel = "Percent Variance Explained",
    label = missing
)

colors = reduce(vcat, [repeat([cond], size(data,2)) for (cond, data) in zip(p["condition"], p["spikes"]) ])

pr = predict(w2, data_matrix2)

scatter3d(pr[1,:], pr[2,:], pr[3, :], alpha=0.5, color = colors)
scatter3d(pr[11,:], pr[12,:], pr[13, :], alpha=0.5, color = colors)
scatter(pr[12,:], pr[13, :], alpha=0.5, color = colors)


# Kernel PCA



w3 = fit(KernelPCA, data_matrix2[:, 1:1000:end], maxoutdim=20, kernel = (x,y)->exp(-0.01*norm(x-y)^2.0), inverse=true)

i = 500

preds = predict(w3, data_matrix2[:, i:1000:end])

scatter3d(preds[1,:], preds[2,:], preds[3,:], alpha = 0.5, group = colors[i:1000:end], label = nothing, xlims=(-0.05,0.05), ylims=(-0.05,0.05), zlims=(-0.05,0.05))
plot!([0,1],[0,0], [0,0], color = "red", label = "x")
plot!([0,0],[0,1], [0,0], color = "blue", label = "y")
plot!([0,0],[0,0], [0,1], color = "green", label = "z")


# trying independent bernouli RVs to see what PCA looks like


k = rand(Bernoulli(mean(data_matrix2)), 128, 100000)

w2 = fit(PCA, k, maxoutdim=20)


## Fitting ICA to the data


r1 = fit(ICA, data_matrix2[:, 1:1000:end], 10)
r1 = fit(ICA, k[:, 1:100:end], 10)



## MDS

M = fit(MDS, data_matrix2[:, 1:1000:end], maxoutdim=10, distances=false)
