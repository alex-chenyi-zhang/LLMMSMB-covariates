using Plots
#using PlotlyJS
using Random, Distributions, StatsBase, LinearAlgebra, DelimitedFiles, Optim
Random.seed!()

# get observed data and known covariates
io = open("data/input/X.txt","r")
X = readdlm(io, Float64)
close(io)
#X = X[:,400:end]

io = open("data/input/Y.txt","r")
Y = readdlm(io, Float64)
close(io)
#Y = Y[400:end,400:end]

N = length(X[1,:])
P = length(X[:,1])
K = 4     # I know that the data was generate with K = 4. In principle one should do model selection to discover it

# variational parameters for the E-step
ϕ = ones(N,N,K) #.* 1/K  # initialized as uniform distributions
for i in 1:N
    for j in 1:N
        ϕ[i,j,:] = rand(Dirichlet(K,1.0))
    end
end
λ = randn(N,K)    # mean vectors for the gaussians for every node
ν = [Matrix(1.0I, K, K) for i in 1:N] # covariance matrices for the gaussians. This is a vector of matrices

# parameters to be optimized in the M-step
#Σ = Matrix(1.0I, K, K)    # global covariance matrix
Σ = rand(Wishart(K,Matrix(.5I,K, K)))
σ_2 = rand(Gamma(1,1), K);      # prior covariance on the transformation coefficients Γ
B = zeros(K,K)
for k in 1:K
    for g in 1:k
        B[k,g] = rand()*0.04
        B[g,k] = B[k,g]
    end
end
B .+= Matrix(1.0I, K, K)*0.8
ρ = 0.1

Γ = zeros(K,P)
for k in 1:K
    Γ[k,:] .= randn(P)* sqrt(σ_2[k])
end

μ = Γ * X;
for i in 1:N
    ϕ[i,i,:] .= 0
end

B = [[0.855, 0.040, 0.024, 0.006] [0.040, 0.692, 0.035, 0.018] [0.024, 0.035, 0.885, 0.046] [0.006, 0.018, 0.046, 0.645]]

###########################################################################################################################


# approximate ELBO
function ELBO(ϕ::Array{Float64, 3}, λ::Array{Float64, 2}, ν::Vector{Matrix{Float64}},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, B::Array{Float64, 2}, ρ::Float64, μ::Array{Float64, 2})
    apprELBO = 0.0
    inv_Σ = inv(Σ)
    apprELBO = 0.5 * N * log(det(inv_Σ))
    for i in 1:N
        apprELBO -= 0.5 * (dot(λ[i,:] .- μ[:,i], inv_Σ, λ[i,:] .- μ[:,i]) + tr(inv_Σ*ν[i])) #first 2 terms
    end
    
    for i in 1:N
        for j in 1:N
            if i != j
                apprELBO += dot(ϕ[i,j,:], λ[i,:]) #third term
                for k in 1:K
                    apprELBO -= ϕ[i,j,k]*log(ϕ[i,j,k]) #last entropic term
                end
            end
        end
    end
    
    for i in 1:N
        theta = zeros(K)
        theta .= exp.(λ[i,:])
        theta /= sum(theta)
        # second line of the expression above
        apprELBO -= (N-1) * ( (log(sum(exp.(λ[i,:])))) + 0.5*tr((diagm(theta) .- theta * theta')*ν[i]) )
        #gaussian entropic term
        apprELBO += 0.5*log(det(ν[i]))
    end
    
    #likelihood term
    for i in 1:N
        for j in 1:i-1
            for k in 1:K
                for g in 1:K
                    #logP = Y[i,j] * ( log(B[k,g]*(1-ρ)) - abs(i-j)/N) - B[k,g]*(1-ρ)*exp(-abs(i-j)/N)
                    logP = Y[i,j]*log(B[k,g]*(1-ρ)) + (1-Y[i,j])*log(1-B[k,g]*(1-ρ))
                    apprELBO += ϕ[i,j,k]*ϕ[j,i,g]*logP
                end
            end
        end
    end
    return apprELBO
end

###########################################################################################################################


# define the function to optimize
function f(η_i::Array{Float64, 1}; ϕ_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i::Array{Float64},N::Int)
    f = -0.5 * dot(η_i .- μ_i, inv_Σ, η_i .- μ_i) + dot(η_i, ϕ_i) -(N-1)*log(sum(exp.(η_i)))
    return f
end


function gradf!(G, η_i::Array{Float64, 1};ϕ_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i::Array{Float64},N::Int)
    G .= -exp.(η_i)/sum(exp.(η_i))*(N-1) .+ ϕ_i - inv_Σ*(η_i .- μ_i) 
end

function hessf!(H, η_i::Array{Float64, 1}; inv_Σ::Array{Float64, 2}, μ_i::Array{Float64},N::Int)
    theta = exp.(η_i)/sum(exp.(η_i))
    H .= - (N-1)*(diagm(theta) .- theta*theta') .- inv_Σ
end

###########################################################################################################################


n_iterations = 10
elbows = zeros(n_iterations)
elbows[1] = ELBO(ϕ, λ, ν, Σ, σ_2, B, ρ, μ)
println(elbows[1])
for i_iter in 2:n_iterations
    inv_Σ = inv(Σ)
    max_steps = 50
    delta_eta = zeros(K)
    tol = 0.001
    G = zeros(K)
    H = zeros(K,K)
    r = 0.05
    #f_values = zeros(N,max_steps)
    for m in 1:6
        f_values = zeros(N,max_steps)
        Threads.@threads for i in 1:N
            G = zeros(K)
            H = zeros(K,K)
            ϕ_i = sum(ϕ[i,:,:],dims=1)[1,:]
            μ_i = μ[:,i]
            η_i = randn(K)
            i_step = 1
            rel_change = 2*tol
            gradf!(G, η_i; ϕ_i=ϕ_i, inv_Σ = inv_Σ, μ_i = μ_i, N=N)
            hessf!(H, η_i; inv_Σ = inv_Σ, μ_i = μ_i, N=N)
            gradient_length = sum(G.^2)
            f_old = f(η_i; ϕ_i=ϕ_i, inv_Σ = inv_Σ, μ_i = μ_i, N=N)
            f_values[i,1] = f_old

            while (i_step < max_steps) & (gradient_length > tol)
            #for i_step in 2:max_steps
                i_step += 1
                delta_eta .= - r*inv(H)*G
                η_i .= η_i .+ delta_eta
                f_new = f(η_i; ϕ_i=ϕ_i, inv_Σ = inv_Σ, μ_i = μ_i, N=N)
                f_values[i,i_step] = f_new
                gradf!(G, η_i; ϕ_i=ϕ_i, inv_Σ = inv_Σ, μ_i = μ_i, N=N)
                hessf!(H, η_i; inv_Σ = inv_Σ, μ_i = μ_i, N=N)
                gradient_length = sum(G.^2)
            end
            λ[i,:] .= η_i
            ν[i] .= -inv(H)
        end
        ###################################################################
        for i in 1:N
            for j in 1:N
                if i != j
                    for k in 1:K
                        logP = λ[i,k] 
                        for g in 1:K
                            logP += ϕ[j,i,g] * (Y[i,j]*log(B[k,g]*(1-ρ)) + (1-Y[i,j])*log(1-(1-ρ)*B[k,g])) 
                        end
                        ϕ[i,j,k] = exp(logP)
                    end
                    ϕ[i,j,:] ./= sum(ϕ[i,j,:])
                end
            end
        end
        ###################################################################
    end
    for m in 1:6
        for k in 1:K
            Γ[k,:] = inv(X*X' + diagm(ones(P)/σ_2[k]))*(X*λ[:,k])
            σ_2[k] = (0.5 + sum(Γ[k,:].^2))/(0.5 + P)
        end
        
    end
    
    μ .= Γ * X

    Σ .= zeros(K,K)
    for i in 1:N
        Σ .+= 1/N * (ν[i] .+ (λ[i,:] .- μ[:,i])*(λ[i,:] .- μ[:,i])')
    end

    #for k in 1:K
    #    for g in 1:K
    #        num = 0.
    #        den = 0.
    #        for i in 1:N
    #            for j in 1:N
    #                num += ϕ[i,j,k]*ϕ[j,i,g]*Y[i,j]
    #                den += ϕ[i,j,k]*ϕ[j,i,g]
    #            end
    #        end
    #        B[k,g] = num/(den*(1-ρ))
    #    end
    #end
    #num = 0.
    #den = 0.
    #for k in 1:K
    #    for g in 1:K
    #        for i in 1:N
    #            for j in 1:N
    #                num += ϕ[i,j,k]*ϕ[j,i,g]*(1-Y[i,j])
    #                den += ϕ[i,j,k]*ϕ[j,i,g]
    #            end
    #        end
    #    end
    #end
    #ρ = num/den
    
    elbows[i_iter] = ELBO(ϕ, λ, ν, Σ, σ_2, B, ρ, μ)
    println(i_iter, " ELBO  ", elbows[i_iter])
    if isnan(elbows[i_iter]) 
        break
    end
    #println(ρ)
end


thetas = zeros(N,K)
for i in 1:N
    thetas[i,:] .= exp.(λ[i,:]) ./ sum(exp.(λ[i,:]))
end


A_pred = zeros(N,N)
A_expected = zeros(N,N)
for i in 1:N
    for j in 1:i-1
        z_i = sample(Weights(thetas[i,:]))
        z_j = sample(Weights(thetas[j,:]))
        
        rate = B[z_i,z_j] *(1-ρ) #*exp(-abs(i-j)/(N))
        if rand() < rate
            A_pred[i,j] = 1
            A_pred[j,i] = A_pred[i,j]
        end
        A_expected[i,j] = rate 
        A_expected[j,i] = A_expected[i,j]
    end
end

open("theta.txt", "w") do io 
    writedlm(io, thetas)
end

open("Sigma.txt", "w") do io 
    writedlm(io, Σ)
end

open("fit_map.txt", "w") do io 
    writedlm(io, A)
end

