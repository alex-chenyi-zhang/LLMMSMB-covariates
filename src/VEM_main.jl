include("LNMMSBM_functions.jl")

function run_inference(n_iter::Int, start_node::Int, end_node::Int, n_runs::Int, covariate_file::String, map_file::String, K::Int)
    # get observed data and known covariates
    io = open(covariate_file,"r")
    X = readdlm(io, Float64; header=true)[1]
    close(io)
    X = X[:,start_node:end_node]

    io = open(map_file,"r")
    Y = readdlm(io, Float64)
    close(io)
    Y = Y[start_node:end_node,start_node:end_node]

    N = length(X[1,:])
    P = length(X[:,1])
    #K = 4     # I know that the data was generate with K = 4. In principle one should do model selection to discover it
    println(P)

    for i_run in 1:n_runs
        println("run number: ", i_run, )
        ϕ = ones(N,N,K) #.* 1/K  # initialized as uniform distributions
        for i in 1:N
            for j in 1:N
                ϕ[i,j,:] = rand(Dirichlet(K,0.5))
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

        elbows = run_VEM!(n_iter, ϕ, λ, ν, Σ, σ_2, B, ρ, μ, Y, X, Γ, K, N, P)

        # make predictions
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
        if !isdir("data/preliminary_results/")
            mkdir("data/preliminary_results/")
        end

        open("data/preliminary_results/thetas_$(N)_$(K)_$(covariate_file[13:end])", "a") do io
            writedlm(io, thetas')
        end
        open("data/preliminary_results/elbows_$(N)_$(K)_$(covariate_file[13:end])", "a") do io
            writedlm(io, elbows')
        end
        open("data/preliminary_results/pred_map_$(N)_$(K)_$(covariate_file[13:end])", "a") do io
            writedlm(io, A_pred)
        end
        open("data/preliminary_results/B_$(N)_$(K)_$(covariate_file[13:end])", "a") do io
            writedlm(io, B)
        end
        open("data/preliminary_results/Sigma_$(N)_$(K)_$(covariate_file[13:end])", "a") do io
            writedlm(io, Σ)
        end
        open("data/preliminary_results/Theta_$(N)_$(K)_$(covariate_file[13:end])", "a") do io
            writedlm(io, Γ)
        end
    end
end
