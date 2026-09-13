using Turing, Distributions, StatsPlots, Random, LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

# Data: 8 observations with 6 bins each
X_data = [
    [0.0, 0.1, 0.2, 0.2, 0.3, 0.2],
    [0.1, 0.3, 0.2, 0.3, 0.1, 0.0],
    [0.3, 0.3, 0.1, 0.2, 0.1, 0.0],
    [0.2, 0.3, 0.1, 0.2, 0.1, 0.1],
    [0.2, 0.0, 0.1, 0.2, 0.3, 0.1],
    [0.2, 0.4, 0.4, 0.0, 0.0, 0.0],
    [0.2, 0.4, 0.3, 0.1, 0.0, 0.0],
    [0.2, 0.3, 0.3, 0.2, 0.0, 0.0]
]

# Binary outcome: whether the action was chosen
Y_data = [1, 1, 0, 0, 1, 0, 1, 1]

# Convert data to a matrix (each row is an observation)
X = reduce(vcat, [x' for x in X_data])
N, D = size(X)  # N observations, D bins

# Add a small constant to avoid zeros (required for Dirichlet) and renormalize each row
epsilon = 1e-6
X .= X .+ epsilon
X = X ./ sum(X, dims=2)

@model function composition_model(X, Y, N, D)
    # Global parameters:
    precision ~ Gamma(2.0, 0.1)                   # Controls dispersion of observed X around latent composition
    beta ~ filldist(Normal(0, 10), D - 1)           # Regression coefficients for ALR-transformed predictors
    intercept ~ Normal(0, 10)                       # Baseline intercept for logistic regression

    # Vectorized sampling: latent true compositions for all observations
    theta ~ filldist(Dirichlet(ones(D)), N)         # theta is an N×D matrix; each row is a latent composition

    # Likelihood for the observed compositional data
    for i in 1:N
        X[i, :] ~ Dirichlet(theta[i, :] * precision)
    end

    # Vectorized Additive Log-Ratio (ALR) transformation:
    # Use the last component as the reference. We reshape theta[:, D] to (N x 1) for correct broadcasting.
    alr = log.(theta[:, 1:(D-1)] ./ reshape(theta[:, D], N, 1))

    # Logistic regression: compute the linear predictor for each observation
    logit_p = intercept .+ alr * beta               # alr is N×(D-1) and beta is a (D-1) vector => result is N×1
    p = 1 ./(1 .+ exp.(-logit_p))                    # Apply the sigmoid function to get probabilities

    # Likelihood for the binary outcome for all observations
    Y ~ arraydist([Bernoulli(p[i]) for i in 1:N])
end

# Instantiate the model
model = composition_model(X, Y_data, N, D)

# Run MCMC sampling using the NUTS sampler
chain = sample(model, NUTS(), 1000)

# Summarize the results
println(describe(chain))

# Plot the traces and posterior distributions
plot(chain)
