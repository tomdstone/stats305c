using Pkg
Pkg.activate("Stats305c", shared=true)

using LinearAlgebra
using StateSpaceDynamics
using Pickle
using StatsPlots
using Distributions
using MultivariateStats
# using HiddenMarkovModels
using Random

##  ------------

p = Pickle.npyload("gdrive/mc_pacman.pkl")

function assemble_first_k_tensor(C, kmax; T=Float64)
    u, s, v = svd(C)

    Cs = Array{T}(undef, size(C)..., kmax)

    for k in 1:kmax
        Cs[:,:,k] .= u[:, 1:k] * diagm(s[1:k]) * v'[1:k, :]
    end

    return Cs
end


function initialize_vardim_obs_models(;
    nneuron = 128,
    nstate = 256,
    kmax = 10,
    T = Float64
)

    C_init = randn(T, nneuron, nstate)
    Cs = assemble_first_k_tensor(C_init, kmax; T)

    obsmodels = EmissionModel[
        PoissonRegressionEmission(
            nstate, nneuron, Cs[:,:,k], true, 0.
        ) for k in 1:kmax
    ]

    return obsmodels
end

function initialize_vardim_hmm(;
    nneuron = 128,
    nstate = 256,
    kmax = 10,
    T = Float64
)
    # WARNING: not sure if rows or cols...?
    trans_matrix = rand(nstate, nstate)
    trans_matrix ./= sum(trans_matrix, dims=1)

    obsmodels = initialize_vardim_obs_models(; nneuron, nstate, kmax, T)

    initial_state_prior = ones(nstate) / nstate

    return HiddenMarkovModel(trans_matrix, obsmodels, initial_state_prior, kmax)
end



function bin_by_factor(M::Matrix, f)
    d, N = size(M)

    n = N ÷ f

    m = Matrix{UInt8}(undef, d, n)

    for i in 1:n
        m[:, i] .= sum(M[:, (f*(i-1)+1):(f*i)], dims=2)
    end
    return m
end


data_20ms = Matrix{Float64}.(bin_by_factor.([(x) for (i,x) in enumerate(p["spikes"]) if p["condition"][i] == 1], 20))


# fit!(hmm, data_20ms)



# With hidden states, can fit regression models!

using StateSpaceDynamics, Random, LinearAlgebra
nneuron = 16
nhidden = 32
nstate  = 10

trans_matrix = rand(nstate, nstate)
trans_matrix ./= sum(trans_matrix, dims=2)

# trans_matrix = zeros(10,10)
# for i in 1:10
#     trans_matrix[i, i:end] .= 1 / (10-i+1)
# end

obsmodels = [
    PoissonRegressionEmission(
        nhidden, nneuron, vcat(ones(1,nneuron), randn(nhidden, nneuron)), true, 0.
    ) for k in 1:nstate
]

hmm = HiddenMarkovModel(trans_matrix, obsmodels, ones(nstate) / nstate, nstate)

neuron_data = rand(Poisson(0.1), nneuron, 100) .|> Float64
hidden_data = rand(MvNormal(zeros(nhidden), 0.01I), 100)

rand(hmm, vcat(ones(1,100),hidden_data), n=100)

StateSpaceDynamics.fit!(hmm, neuron_data, hidden_data)
