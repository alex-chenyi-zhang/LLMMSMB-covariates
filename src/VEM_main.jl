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
        λ = randn(K,N)    # mean vectors for the gaussians for every node
        #ν = [Matrix(1.0I, K, K) for i in 1:N] # covariance matrices for the gaussians. This is a vector of matrices
        ν = zeros(K,K,N)
        for i in 1:N
            ν[:,:,i] = rand(Wishart(K,Matrix(.5I,K, K)))
        end




        # parameters to be optimized in the M-step
        #Σ = Matrix(1.0I, K, K)    # global covariance matrix
        Σ = rand(Wishart(K,Matrix(.5I,K, K)))
        σ_2 = rand(InverseGamma(1,1), K);      # prior covariance on the transformation coefficients Γ

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
        thetas = zeros(K,N)
        for i in 1:N
            thetas[:,i] .= exp.(λ[:,i]) ./ sum(exp.(λ[:,i]))
        end

        A_pred = zeros(N,N)
        A_expected = zeros(N,N)
        for i in 1:N
            for j in 1:i-1
                z_i = sample(Weights(thetas[:,i]))
                z_j = sample(Weights(thetas[:,j]))

                rate = B[z_i,z_j] *(1-ρ) #*exp(-abs(i-j)/(N))
                if rand() < rate
                    A_pred[i,j] = 1
                    A_pred[j,i] = A_pred[i,j]
                end
                A_expected[i,j] = rate
                A_expected[j,i] = A_expected[i,j]
            end
        end
        nu_matrix = zeros(N,K*K)
        for i in 1:N
            nu_matrix[i,:] .= [ν[:,:,i]...]
        end

        if !isdir("data/preliminary_results/")
            mkdir("data/preliminary_results/")
        end

        open("data/preliminary_results/thetas_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, thetas)
        end
        open("data/preliminary_results/elbows_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, elbows')
        end
        open("data/preliminary_results/nu_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, nu_matrix)
        end
        open("data/preliminary_results/lambda_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, λ)
        end
        open("data/preliminary_results/pred_map_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, A_pred)
        end
        open("data/preliminary_results/B_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, B)
        end
        open("data/preliminary_results/Sigma_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, Σ)
        end
        open("data/preliminary_results/Gamma_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, Γ)
        end
        open("data/preliminary_results/sigma_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, σ_2')
        end
    end
end

## this is a slight variation of the previous function that allows for multiple regions to e considered together

function run_inference_multi(n_iter::Int, n_runs::Int, covariate_file_names::String, map_file_names::String, K::Int)
    io = open(covariate_file_names, "r")
    covariate_files = readdlm(io, String)
    close(io)

    io = open(map_file_names, "r")
    contact_map_files = readdlm(io, String)
    close(io)

    n_regions = length(contact_map_files)
    println(n_regions)

    println(contact_map_files[1], "\t" , covariate_files[1])
    # read the first files from list
    io = open(covariate_files[1],"r")
    X_i = readdlm(io, Float64; header=true)[1]
    close(io)

    io = open(contact_map_files[1],"r")
    Y_i = readdlm(io, Float64)
    close(io)

    N = length(X_i[1,:])
    P = length(X_i[:,1])



    # data structures for aggregated covariates and contact maps
    X = zeros(P,N*n_regions)
    Y = zeros(N, N*n_regions)
    X[:,1:N] .= X_i
    Y[:,1:N] .= Y_i
    for i_region in 2:n_regions
        println(contact_map_files[i_region], "\t" , covariate_files[i_region])
        io = open(covariate_files[i_region],"r")
        X_i = readdlm(io, Float64; header=true)[1]
        close(io)

        io = open(contact_map_files[i_region],"r")
        Y_i = readdlm(io, Float64)
        close(io)
        X[:,(i_region-1)*N+1:i_region*N] .= X_i
        Y[:,(i_region-1)*N+1:i_region*N] .= Y_i
    end
    println(length(X[1,:]))
    println(length(Y[1,:]))
    println(N, "\t", P)


    for i_run in 1:n_runs
        println("run number: ", i_run, )
        ϕ = [ones(N,N,K) for i_region in 1:n_regions] #.* 1/K  # initialized as uniform distributions
        for i_region in 1:n_regions
            for i in 1:N
                for j in 1:N
                    ϕ[i_region][i,j,:] = rand(Dirichlet(K,0.5))
                end
            end
        end

        #λ = randn(N*n_regions,K)    # mean vectors for the gaussians for every node
        λ = randn(K, N*n_regions)
        #ν = [Matrix(1.0I, K, K) for i in 1:N] # covariance matrices for the gaussians. This is a vector of matrices
        ν = [zeros(K,K,N) for i_region in 1:n_regions]
        for i_region in 1:n_regions
            for i in 1:N
                ν[i_region][:,:,i] = rand(Wishart(K,Matrix(.5I,K, K)))
            end
        end
#########################################################
        for i_region in 2:n_regions
            for i in 1:N
                for j in 1:N
                    ϕ[i_region][i,j,:] .= ϕ[1][i,j,:]
                end
                ν[i_region][:,:,i] = ν[1][:,:,i]
            end
            λ[:,(i_region-1)*N+1:(i_region)*N] .= λ[:,1:N]
        end
#########################################################
        # parameters to be optimized in the M-step
        #Σ = Matrix(1.0I, K, K)    # global covariance matrix
        Σ = rand(Wishart(K,Matrix(.5I,K, K)))
        σ_2 = rand(InverseGamma(1,1), K);      # prior covariance on the transformation coefficients Γ

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

        #μ = Γ * X;
        for i_region in 1:n_regions
            for i in 1:N
                ϕ[i_region][i,i,:] .= 0
            end
        end
        μ = zeros(K,N*n_regions)
        μ = Γ * X#[:,(i_region-1)*N+1:i_region*N]
        ########################################################################
        ########################################################################

        # I think there is a problem analizing all the regions together because of a
        # problem of excheancgeability of the topic labelling.
        # to try to solve this, I can try to run for a bit the VEM on one region alone
        # and use these values for the "global parameters"
        #=println("start burn in...")
        burn_in = 10
        for bi in 1:10
            μ = Γ * X
            #λ .= μ
            el_burn_in = run_VEM!(burn_in, ϕ[1], λ[:,1:N], ν[1], Σ, σ_2, B, ρ, μ[:,1:N], Y[:,1:N], X[:,1:N], Γ, K, N, P)
            μ = Γ * X
            #λ .= μ
            el_burn_in = run_VEM!(burn_in, ϕ[2], λ[:,N+1:2*N], ν[2], Σ, σ_2, B, ρ, μ[:,N+1:2*N], Y[:,N+1:2*N], X[:,N+1:2*N], Γ, K, N, P)
        end


        μ = Γ * X
        println("... burn in terminated")=#

        elbows = run_VEM!(n_iter, ϕ, λ, ν, Σ, σ_2, B, ρ, μ, Y, X, Γ, K, N, P, n_regions)
        #println(elbows)


        if !isdir("data/preliminary_results/multi_region/")
            mkdir("data/preliminary_results/multi_region/")
        end

        for i_region in 1:n_regions
            nu_matrix = zeros(N,K*K)
            for i in 1:N
                nu_matrix[i,:] .= [ν[i_region][:,:,i]...]
            end
            println((covariate_files[i_region][13:end]))
            open("data/preliminary_results/multi_region/elbows_$(N)_$(K)$(covariate_files[i_region][13:end])", "a") do io
                writedlm(io, elbows[i_region,:]')
            end
            open("data/preliminary_results/multi_region/nu_$(N)_$(K)$(covariate_files[i_region][13:end])", "a") do io
                writedlm(io, nu_matrix)
            end
            open("data/preliminary_results/multi_region/lambda_$(N)_$(K)$(covariate_files[i_region][13:end])", "a") do io
                writedlm(io, λ)
            end
        end

        open("data/preliminary_results/multi_region/B_$(N)_$(K).txt", "a") do io
            writedlm(io, B)
        end
        open("data/preliminary_results/multi_region/Sigma_$(N)_$(K).txt", "a") do io
            writedlm(io, Σ)
        end
        open("data/preliminary_results/multi_region/Gamma_$(N)_$(K).txt", "a") do io
            writedlm(io, Γ)
        end
        open("data/preliminary_results/multi_region/sigma_$(N)_$(K).txt", "a") do io
            writedlm(io, σ_2')
        end
    end
end






function prova_identificabilita(n_iter::Int, covariate_file_names::String, map_file_names::String, K::Int)
    io = open(covariate_file_names, "r")
    covariate_files = readdlm(io, String)
    close(io)

    io = open(map_file_names, "r")
    contact_map_files = readdlm(io, String)
    close(io)

    n_regions = length(contact_map_files)
    println(n_regions)

    println(contact_map_files[1], "\t" , covariate_files[1])
    # read the first files from list
    io = open(covariate_files[1],"r")
    X_i = readdlm(io, Float64; header=true)[1]
    close(io)

    io = open(contact_map_files[1],"r")
    Y_i = readdlm(io, Float64)
    close(io)

    N = length(X_i[1,:])
    P = length(X_i[:,1])



    # data structures for aggregated covariates and contact maps
    X = zeros(P,N*n_regions)
    Y = zeros(N, N*n_regions)
    X[:,1:N] .= X_i
    Y[:,1:N] .= Y_i
    for i_region in 2:n_regions
        println(contact_map_files[i_region], "\t" , covariate_files[i_region])
        io = open(covariate_files[i_region],"r")
        X_i = readdlm(io, Float64; header=true)[1]
        close(io)

        io = open(contact_map_files[i_region],"r")
        Y_i = readdlm(io, Float64)
        close(io)
        X[:,(i_region-1)*N+1:i_region*N] .= X_i
        Y[:,(i_region-1)*N+1:i_region*N] .= Y_i
    end
    println(X[:,1], "\n", X[:,N*n_regions])
    println(length(X[1,:]))
    println(length(Y[1,:]))
    println(N, "\t", P)


    ϕ = [ones(N,N,K) for i_region in 1:n_regions] #.* 1/K  # initialized as uniform distributions
    for i_region in 1:n_regions
        for i in 1:N
            for j in 1:N
                ϕ[i_region][i,j,:] = rand(Dirichlet(K,0.5))
            end
        end
    end

    λ = randn(K, N*n_regions)  # mean vectors for the gaussians for every node
    #ν = [Matrix(1.0I, K, K) for i in 1:N] # covariance matrices for the gaussians. This is a vector of matrices
    ν = [zeros(K,K,N) for i_region in 1:n_regions]
    for i_region in 1:n_regions
        for i in 1:N
            ν[i_region][:,:,i] = rand(Wishart(K,Matrix(.5I,K, K)))
        end
    end

#########################################################
    # parameters to be optimized in the M-step
    #Σ = Matrix(1.0I, K, K)    # global covariance matrix
    Σ = rand(Wishart(K,Matrix(.5I,K, K)))
    σ_2 = rand(InverseGamma(1,1), K);      # prior covariance on the transformation coefficients Γ

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

    #μ = Γ * X;
    for i_region in 1:n_regions
        for i in 1:N
            ϕ[i_region][i,i,:] .= 0
        end
    end
    μ = zeros(K,N*n_regions)
    μ = Γ * X#[:,(i_region-1)*N+1:i_region*N]
    ########################################################################
    ########################################################################

    el_burn_in = run_VEM!(n_iter, ϕ[1], λ[:,1:N], ν[1], Σ, σ_2, B, ρ, μ[:,1:N], Y[:,1:N], X[:,1:N], Γ, K, N, P)


    i_region = 2
    inv_Σ = inv(Σ)
    μ = Γ * X
    for n in 1:50
        inv_Σ = inv(Σ)
        μ = Γ * X
        for m in 1:10
            println(m, "\t", ELBO(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, σ_2, B, ρ, μ[:,(i_region-1)*N+1:i_region*N], Y[:,(i_region-1)*N+1:i_region*N], K, N))
            Estep_logitNorm!(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], inv_Σ, μ[:,(i_region-1)*N+1:i_region*N], N, K)
            Estep_multinomial!(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], B, ρ, Y[:,(i_region-1)*N+1:i_region*N], N, K)
            Estep_logitNorm!(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], inv_Σ, μ[:,(i_region-1)*N+1:i_region*N], N, K)
        end

        println("\t", ELBO(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, σ_2, B, ρ, μ[:,(i_region-1)*N+1:i_region*N], Y[:,(i_region-1)*N+1:i_region*N], K, N))
        Mstep_logitNorm!(λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, σ_2, μ[:,(i_region-1)*N+1:i_region*N], Γ, X[:,(i_region-1)*N+1:i_region*N], N, K, P)
        Mstep_blockmodel!(ϕ[i_region], B, ρ, Y[:,(i_region-1)*N+1:i_region*N], N, K)
        println("\t", ELBO(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, σ_2, B, ρ, μ[:,(i_region-1)*N+1:i_region*N], Y[:,(i_region-1)*N+1:i_region*N], K, N))
        inv_Σ = inv(Σ)
        μ = Γ * X
        for m in 1:10
            println(m, "\t", ELBO(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, σ_2, B, ρ, μ[:,(i_region-1)*N+1:i_region*N], Y[:,(i_region-1)*N+1:i_region*N], K, N))
            Estep_logitNorm!(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], inv_Σ, μ[:,(i_region-1)*N+1:i_region*N], N, K)
            Estep_multinomial!(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], B, ρ, Y[:,(i_region-1)*N+1:i_region*N], N, K)
            Estep_logitNorm!(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], inv_Σ, μ[:,(i_region-1)*N+1:i_region*N], N, K)
        end
    end

    el_burn_in = run_VEM!(150, ϕ[1], λ[:,1:N], ν[1], Σ, σ_2, B, ρ, μ[:,1:N], Y[:,1:N], X[:,1:N], Γ, K, N, P)


end


function prova_identificabilita_2(n_iter::Int, covariate_file_names::String, map_file_names::String, K::Int)
    io = open(covariate_file_names, "r")
    covariate_files = readdlm(io, String)
    close(io)

    io = open(map_file_names, "r")
    contact_map_files = readdlm(io, String)
    close(io)

    n_regions = length(contact_map_files)
    println(n_regions)

    println(contact_map_files[1], "\t" , covariate_files[1])
    # read the first files from list
    io = open(covariate_files[1],"r")
    X_i = readdlm(io, Float64; header=true)[1]
    close(io)
    X_i = X_i[:,1:500]

    io = open(contact_map_files[1],"r")
    Y_i = readdlm(io, Float64)
    close(io)
    Y_i = Y_i[1:500,1:500]

    N = length(X_i[1,:])
    P = length(X_i[:,1])



    # data structures for aggregated covariates and contact maps
    X = zeros(P,N*n_regions)
    Y = zeros(N, N*n_regions)
    X[:,1:N] .= X_i
    Y[:,1:N] .= Y_i
    for i_region in 2:n_regions
        println(contact_map_files[i_region], "\t" , covariate_files[i_region])
        io = open(covariate_files[i_region],"r")
        X_i = readdlm(io, Float64; header=true)[1]
        close(io)
        X_i = X_i[:,1:500]

        io = open(contact_map_files[i_region],"r")
        Y_i = readdlm(io, Float64)
        close(io)
        Y_i = Y_i[1:500,1:500]
        X[:,(i_region-1)*N+1:i_region*N] .= X_i
        Y[:,(i_region-1)*N+1:i_region*N] .= Y_i
    end
    println(X[:,1], "\n", X[:,N*n_regions])
    println(length(X[1,:]))
    println(length(Y[1,:]))
    println(N, "\t", P)


    ϕ = [ones(N,N,K) for i_region in 1:n_regions] #.* 1/K  # initialized as uniform distributions
    for i_region in 1:n_regions
        for i in 1:N
            for j in 1:N
                ϕ[i_region][i,j,:] = rand(Dirichlet(K,0.5))
            end
        end
    end

    λ = randn(K, N*n_regions)  # mean vectors for the gaussians for every node
    #ν = [Matrix(1.0I, K, K) for i in 1:N] # covariance matrices for the gaussians. This is a vector of matrices
    ν = [zeros(K,K,N) for i_region in 1:n_regions]
    for i_region in 1:n_regions
        for i in 1:N
            ν[i_region][:,:,i] = rand(Wishart(K,Matrix(.5I,K, K)))
        end
    end

#########################################################
    # parameters to be optimized in the M-step
    #Σ = Matrix(1.0I, K, K)    # global covariance matrix
    Σ = rand(Wishart(K,Matrix(.5I,K, K)))
    σ_2 = rand(InverseGamma(1,1), K);      # prior covariance on the transformation coefficients Γ

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

    #μ = Γ * X;
    for i_region in 1:n_regions
        for i in 1:N
            ϕ[i_region][i,i,:] .= 0
        end
    end
    μ = zeros(K,N*n_regions)
    μ = Γ * X#[:,(i_region-1)*N+1:i_region*N]
    ########################################################################
    ########################################################################

    for M in 1:5
        for i_region in 1:1#n_regions
            inv_Σ = inv(Σ)
            μ = Γ * X
            println(M, "\t", i_region)
            elbows = run_VEM!(n_iter, ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), ν[i_region], Σ, σ_2, B, ρ, @view(μ[:,(i_region-1)*N+1:i_region*N]), Y[:,(i_region-1)*N+1:i_region*N], X[:,(i_region-1)*N+1:i_region*N], Γ, K, N, P)
            println(ELBO(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, σ_2, B, ρ, μ[:,(i_region-1)*N+1:i_region*N], Y[:,(i_region-1)*N+1:i_region*N], K, N))
            elbows = run_VEM!(n_iter, ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), ν[i_region], Σ, σ_2, B, ρ, @view(μ[:,(i_region-1)*N+1:i_region*N]), Y[:,(i_region-1)*N+1:i_region*N], X[:,(i_region-1)*N+1:i_region*N], Γ, K, N, P)
            println(ELBO(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, σ_2, B, ρ, μ[:,(i_region-1)*N+1:i_region*N], Y[:,(i_region-1)*N+1:i_region*N], K, N))

            #=nu_matrix = zeros(N,K*K)
            for i in 1:N
                nu_matrix[i,:] .= [ν[i_region][:,:,i]...]
            end
            println((covariate_files[i_region][13:end]))
            open("data/preliminary_results/multi_region/elbows_$(N)_$(K)$(covariate_files[i_region][13:end])", "a") do io
                writedlm(io, elbows')
            end
            open("data/preliminary_results/multi_region/nu_$(N)_$(K)$(covariate_files[i_region][13:end])", "a") do io
                writedlm(io, nu_matrix)
            end
            open("data/preliminary_results/multi_region/lambda_$(N)_$(K)$(covariate_files[i_region][13:end])", "a") do io
                writedlm(io, λ[:,(i_region-1)*N+1:i_region*N])
            end


            open("data/preliminary_results/multi_region/B_$(N)_$(K).txt", "a") do io
                writedlm(io, B)
            end
            open("data/preliminary_results/multi_region/Sigma_$(N)_$(K).txt", "a") do io
                writedlm(io, Σ)
            end
            open("data/preliminary_results/multi_region/Gamma_$(N)_$(K).txt", "a") do io
                writedlm(io, Γ)
            end=#
        end
    end

end
