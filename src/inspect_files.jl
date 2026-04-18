using Pkg
Pkg.activate("Stats305c", shared=true)

using MAT, Pickle, MultivariateStats, Plots

# data source 1
dir = "gdrive/doi-10.7281-t1-v73vza/SMA_Monkey"
x = matread(joinpath(dir, "BothMonkey_SMA_neuron_Info.mat"))
y = matread(joinpath(dir, "MonkeyH/Multi-attribute-Hobbit-01-02-2019.mat"))

# MC Cycle data
z = Pickle.npyload("gdrive/mc_cycle.pkl")

# I just looked through the data objects in my terminal

data_matrix = reduce(hcat, z["spikes"])

w = fit(PCA, data_matrix[:, 1:100:end], maxoutdim = 20)

cumsum(w.prinvars)
