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
            #thetas[:,i] .= exp.(λ[:,i]) ./ sum(exp.(λ[:,i]))
            thetas[:,i] .= softmax(λ[:,i])
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

##########################################################################################################################
#=  what follows is the function to run the inference without taking into consideration input covariates. In this
    variant the topic distributions have a shared global logistic normal prior with only one mean vector and covariance
    matrix parameters. Whereas in the case with covariates you still have a global covariance matrix for the logistic normal
    but instead of having a single mean vector, you have a shared linear transformation (i.e. a matrix to be multiplied) that
    transforms the node specific bio-covariates into node specific mean vectors for the logistic normal
=#

function run_inference_gauss(n_iter::Int, start_node::Int, end_node::Int, n_runs::Int, map_file::String, K::Int)
    io = open(map_file,"r")
    Y = readdlm(io, Float64)
    close(io)
    Y = Y[start_node:end_node,start_node:end_node]
    N = length(Y[1,:])
    println(N)
    Threads.@threads for i_run in 1:n_runs
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

        #like_var = rand(InverseGamma(1,1), 1)
        #like_var[1]=0.2
        like_var = zeros(K,K)
        for k in 1:K
            for g  in 1:k
                like_var[k,g] = rand(InverseGamma(1,1))
                like_var[g,k] = like_var[k,g]
            end
        end

        B = zeros(K,K)
        for k in 1:K
            B[k,k] = randn()*0.2+1.0
            for g in 1:k-1
                B[k,g] = randn()*0.2
                B[g,k] = B[k,g]
            end
        end
        #B .+= Matrix(1.0I, K, K)*0.8


        μ = randn(K)

        for i in 1:N
            ϕ[i,i,:] .= 0
        end

        elbows = run_VEM_gauss!(n_iter, ϕ, λ, ν, Σ, B, like_var, μ, Y, K, N)

        # make predictions
        thetas = zeros(K,N)
        for i in 1:N
            #thetas[:,i] .= exp.(λ[:,i]) ./ sum(exp.(λ[:,i]))
            thetas[:,i] .= softmax(λ[:,i])
        end

        like_std = sqrt.(like_var)
        println(like_std)
        A_pred = zeros(N,N)
        A_expected = zeros(N,N)
        for i in 1:N
            for j in 1:i-1
                z_i = sample(Weights(thetas[:,i]))
                z_j = sample(Weights(thetas[:,j]))

                r = B[z_i,z_j]
                A_expected[i,j] = r
                A_expected[j,i] = A_expected[i,j]
                A_pred[i,j] = rand(Normal(A_expected[i,j],like_std[z_i,z_j]))
                A_pred[j,i] = A_pred[i,j]
            end
        end

        nu_matrix = zeros(N,K*K)
        for i in 1:N
            nu_matrix[i,:] .= [ν[:,:,i]...]
        end
        data_dir = "data/preliminary_results/no_covariates/$(map_file[14:end-4])/"
        if !isdir(data_dir)
            mkdir(data_dir)
        end

        open("$(data_dir)$(i_run)_thetas_$(N)_$(K)$(map_file[13:end])", "a") do io
            writedlm(io, thetas)
        end
        open("$(data_dir)$(i_run)_elbows_$(N)_$(K)$(map_file[13:end])", "a") do io
            writedlm(io, elbows')
        end
        open("$(data_dir)$(i_run)_nu_$(N)_$(K)$(map_file[13:end])", "a") do io
            writedlm(io, nu_matrix)
        end
        open("$(data_dir)$(i_run)_lambda_$(N)_$(K)$(map_file[13:end])", "a") do io
            writedlm(io, λ)
        end
        open("$(data_dir)$(i_run)_pred_map_$(N)_$(K)$(map_file[13:end])", "a") do io
            writedlm(io, A_pred)
        end
        open("$(data_dir)$(i_run)_B_$(N)_$(K)$(map_file[13:end])", "a") do io
            writedlm(io, B)
        end
        open("$(data_dir)$(i_run)_Sigma_$(N)_$(K)$(map_file[13:end])", "a") do io
            writedlm(io, Σ)
        end
        open("$(data_dir)$(i_run)_like_var_$(N)_$(K)$(map_file[13:end])", "a") do io
            writedlm(io, like_var)
        end
    end

end

##########################################################################################################################
### This version of the function initializes the global Gamma matrix to some non-random initial value.
function run_inference_gauss(n_iter::Int, start_node::Int, end_node::Int, n_runs::Int, covariate_file::String, map_file::String, gamma_file::String,K::Int)

    io = open(gamma_file,"r")
    gamma_init = readdlm(io, Float64)
    close(io)

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

    Threads.@threads for i_run in 1:n_runs
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
        σ_2 = rand(InverseGamma(1,1), K)      # prior covariance on the transformation coefficients Γ
        #like_var = rand(InverseGamma(1,1), 1)
        #like_var[1]=0.2
        like_var = zeros(K,K)
        for k in 1:K
            for g  in 1:k
                like_var[k,g] = rand(InverseGamma(1,1))
                like_var[g,k] = like_var[k,g]
            end
        end

        B = zeros(K,K)
        for k in 1:K
            B[k,k] = randn()*0.2+1.0
            for g in 1:k-1
                B[k,g] = randn()*0.2
                B[g,k] = B[k,g]
            end
        end
        #B .+= Matrix(1.0I, K, K)*0.8

        Γ = zeros(K,P)
        Γ .= gamma_init

        μ = Γ * X;
        for i in 1:N
            ϕ[i,i,:] .= 0
        end

        elbows = run_VEM_gauss_2!(n_iter, ϕ, λ, ν, Σ, σ_2, B, like_var, μ, Y, X, Γ, K, N, P)

        # make predictions
        thetas = zeros(K,N)
        for i in 1:N
            #thetas[:,i] .= exp.(λ[:,i]) ./ sum(exp.(λ[:,i]))
            thetas[:,i] .= softmax(λ[:,i])
        end

        like_std = sqrt.(like_var)
        println(like_std)
        A_pred = zeros(N,N)
        A_expected = zeros(N,N)
        for i in 1:N
            for j in 1:i-1
                z_i = sample(Weights(thetas[:,i]))
                z_j = sample(Weights(thetas[:,j]))

                r = B[z_i,z_j] #*(1-ρ) #*exp(-abs(i-j)/(N))
                #if rand() < rate
                #    A_pred[i,j] = 1
                #    A_pred[j,i] = A_pred[i,j]
                #end
                A_expected[i,j] = r
                A_expected[j,i] = A_expected[i,j]
                A_pred[i,j] = rand(Normal(A_expected[i,j],like_std[z_i,z_j]))
                A_pred[j,i] = A_pred[i,j]
            end
        end

        nu_matrix = zeros(N,K*K)
        for i in 1:N
            nu_matrix[i,:] .= [ν[:,:,i]...]
        end
        data_dir = "data/preliminary_results/$(map_file[14:end-4])/"
        if !isdir(data_dir)
            mkdir(data_dir)
        end

        open("$(data_dir)$(i_run)_thetas_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, thetas)
        end
        open("$(data_dir)$(i_run)_elbows_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, elbows')
        end
        open("$(data_dir)$(i_run)_nu_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, nu_matrix)
        end
        open("$(data_dir)$(i_run)_lambda_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, λ)
        end
        open("$(data_dir)$(i_run)_pred_map_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, A_pred)
        end
        open("$(data_dir)$(i_run)_B_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, B)
        end
        open("$(data_dir)$(i_run)_Sigma_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, Σ)
        end
        open("$(data_dir)$(i_run)_Gamma_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, Γ)
        end
        open("$(data_dir)$(i_run)_sigma_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, σ_2')
        end
        open("$(data_dir)$(i_run)_like_var_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, like_var)
        end
    end
end


##########################################################################################################################

function run_inference_gauss(n_iter::Int, start_node::Int, end_node::Int, n_runs::Int, covariate_file::String, map_file::String, K::Int)
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

    Threads.@threads for i_run in 1:n_runs
        println("run number: ", i_run)
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
        σ_2 = rand(InverseGamma(1,1), K)      # prior covariance on the transformation coefficients Γ
        #like_var = rand(InverseGamma(1,1), 1)
        #like_var[1]=0.2
        like_var = zeros(K,K)
        for k in 1:K
            for g  in 1:k
                like_var[k,g] = rand(InverseGamma(1,1))
                like_var[g,k] = like_var[k,g]
            end
        end

        B = zeros(K,K)
        for k in 1:K
            B[k,k] = randn()*0.2+1.0
            for g in 1:k-1
                B[k,g] = randn()*0.2
                B[g,k] = B[k,g]
            end
        end
        #B .+= Matrix(1.0I, K, K)*0.8

        Γ = zeros(K,P)
        for k in 1:K
            Γ[k,:] .= randn(P)* sqrt(σ_2[k])
        end

        μ = Γ * X;
        for i in 1:N
            ϕ[i,i,:] .= 0
        end

        elbows = run_VEM_gauss!(n_iter, ϕ, λ, ν, Σ, σ_2, B, like_var, μ, Y, X, Γ, K, N, P)

        # make predictions
        thetas = zeros(K,N)
        for i in 1:N
            #thetas[:,i] .= exp.(λ[:,i]) ./ sum(exp.(λ[:,i]))
            thetas[:,i] .= softmax(λ[:,i])
        end

        like_std = sqrt.(like_var)
        println(like_std)
        A_pred = zeros(N,N)
        A_expected = zeros(N,N)
        for i in 1:N
            for j in 1:i-1
                z_i = sample(Weights(thetas[:,i]))
                z_j = sample(Weights(thetas[:,j]))

                r = B[z_i,z_j] #*(1-ρ) #*exp(-abs(i-j)/(N))
                #if rand() < rate
                #    A_pred[i,j] = 1
                #    A_pred[j,i] = A_pred[i,j]
                #end
                A_expected[i,j] = r
                A_expected[j,i] = A_expected[i,j]
                A_pred[i,j] = rand(Normal(A_expected[i,j],like_std[z_i,z_j]))
                A_pred[j,i] = A_pred[i,j]
            end
        end

        nu_matrix = zeros(N,K*K)
        for i in 1:N
            nu_matrix[i,:] .= [ν[:,:,i]...]
        end
        data_dir = "data/preliminary_results/$(map_file[14:end-4])/"
        if !isdir(data_dir)
            mkdir(data_dir)
        end

        open("$(data_dir)$(i_run)_thetas_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, thetas)
        end
        open("$(data_dir)$(i_run)_elbows_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, elbows')
        end
        open("$(data_dir)$(i_run)_nu_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, nu_matrix)
        end
        open("$(data_dir)$(i_run)_lambda_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, λ)
        end
        open("$(data_dir)$(i_run)_pred_map_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, A_pred)
        end
        open("$(data_dir)$(i_run)_B_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, B)
        end
        open("$(data_dir)$(i_run)_Sigma_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, Σ)
        end
        open("$(data_dir)$(i_run)_Gamma_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, Γ)
        end
        open("$(data_dir)$(i_run)_sigma_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, σ_2')
        end
        open("$(data_dir)$(i_run)_like_var_$(N)_$(K)$(covariate_file[13:end])", "a") do io
            writedlm(io, like_var)
        end
    end
end






## this is a slight variation of the previous function that allows for multiple regions to be considered together

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
                writedlm(io, λ[:,(i_region-1)*N+1:i_region*N])
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


function run_inference_gauss_multi(n_iter::Int, start_node::Int, end_node::Int, n_runs::Int, covariate_file_names::String, map_file_names::String, K::Int, R::Float64)

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
        println("trace norm!! yay, R = ", R)
        ##############################    Initialize all parameters

        println("run number: ", i_run)
        ϕ = [ones(N,N,K) for i_region in 1:n_regions]
        for i_region in 1:n_regions
            for i in 1:N
                for j in 1:N
                    ϕ[i_region][i,j,:] = rand(Dirichlet(K,0.5))
                end
            end
        end

        λ = randn(K, N*n_regions)
        ν = [zeros(K,K,N) for i_region in 1:n_regions]
        for i_region in n_regions
            for i in 1:N
                ν[i_region][:,:,i] = rand(Wishart(K,Matrix(.5I,K, K)))
            end
        end

        #Σ = rand(Wishart(K,Matrix(.5I,K, K)))
        Σ = Matrix(0.7I, K, K)
        σ_2 = rand(InverseGamma(1,1), K)

        like_var = zeros(K,K)
        for k in 1:K
            for g  in 1:k
                like_var[k,g] = rand(InverseGamma(1,1))
                like_var[g,k] = like_var[k,g]
            end
        end

        B = zeros(K,K)
        for k in 1:K
            B[k,k] = randn()*0.2+1.0
            for g in 1:k-1
                B[k,g] = randn()*0.2
                B[g,k] = B[k,g]
            end
        end

        Γ = zeros(K,P)
        for k in 1:K
            Γ[k,:] .= randn(P)* sqrt(σ_2[k])
        end

        μ = zeros(K,N*n_regions)
        μ = Γ * X;
        for i_region in 1:n_regions
            for i in 1:N
                ϕ[i_region][i,i,:] .= 0
            end
        end
        #println(μ)
        ###### end of initialization

        elbows, det_Sigma, det_nu = run_VEM_gauss!(n_iter, ϕ, λ, ν, Σ, σ_2, B, like_var, μ, Y, X, Γ, K, N, P, n_regions, R)
        data_dir = "data/results/100k_multiregion_regularized$(R)/"

        μ = Γ * X;

        if !isdir(data_dir)
            mkdir(data_dir)
        end

        for i_region in 1:n_regions
            nu_matrix = zeros(N,K*K)
            for i in 1:N
                nu_matrix[i,:] .= [ν[i_region][:,:,i]...]
            end
            println((covariate_files[i_region][27:end]))

            open("$(data_dir)elbows_$(N)_$(K)$(covariate_files[i_region][27:end])", "a") do io
                writedlm(io, elbows[i_region,:]')
            end
            open("$(data_dir)nu_$(N)_$(K)$(covariate_files[i_region][27:end])", "a") do io
                writedlm(io, nu_matrix)
            end
            open("$(data_dir)lambda_$(N)_$(K)$(covariate_files[i_region][27:end])", "a") do io
                writedlm(io, λ[:,(i_region-1)*N+1:i_region*N])
            end
            open("$(data_dir)det_nu_$(N)_$(K)$(covariate_files[i_region][27:end])", "a") do io
                writedlm(io, det_nu[i_region]')
            end
        end

        open("$(data_dir)B_$(N)_$(K)_100k.txt", "a") do io
            writedlm(io, B)
        end
        open("$(data_dir)Sigma_$(N)_$(K)_100k.txt", "a") do io
            writedlm(io, Σ)
        end
        open("$(data_dir)Gamma_$(N)_$(K)_100k.txt", "a") do io
            writedlm(io, Γ)
        end
        open("$(data_dir)sigma_$(N)_$(K)_100k.txt", "a") do io
            writedlm(io, σ_2')
        end
        open("$(data_dir)like_var_$(N)_$(K)_100k.txt", "a") do io
            writedlm(io, like_var)
        end
        open("$(data_dir)det_Sigma_$(N)_$(K)_100k.txt", "a") do io
            writedlm(io, det_Sigma')
        end

    end
end

#= in this version the function that maps the covariates X --> eta --> theta is not a
   simple linear transformation matrix Γ, but a generic Flux model (that could also
   be a simple linear transformation). For now we start with a simple 2 layer NN.
   The loss function that we choose if the KL-divergence between the tranformed X
   and the lambda variational parameters =#
function run_inference_gauss_multi_NN(n_iter::Int, start_node::Int, end_node::Int, n_runs::Int, covariate_file_names::String, map_file_names::String, K::Int, R::Float64)
    println("trace norm NN!! yay R = ", R)
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
    X_i = readdlm(io, Float64; header=true)[1][1:end-1,:]
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
        X_i = readdlm(io, Float64; header=true)[1][1:end-1,:]
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
        ##############################    Initialize all parameters

        println("run number: ", i_run)
        ϕ = [ones(N,N,K) for i_region in 1:n_regions]
        for i_region in 1:n_regions
            for i in 1:N
                for j in 1:N
                    ϕ[i_region][i,j,:] = rand(Dirichlet(K,0.5))
                end
            end
        end

        λ = randn(K, N*n_regions)
        ν = [zeros(K,K,N) for i_region in 1:n_regions]
        for i_region in n_regions
            for i in 1:N
                ν[i_region][:,:,i] = rand(Wishart(K,Matrix(.5I,K, K)))
            end
        end

        #Σ = rand(Wishart(K,Matrix(.5I,K, K)))
        #σ_2 = rand(InverseGamma(1,1), K)
        Σ = Matrix(0.7I, K, K)

        like_var = zeros(K,K)
        for k in 1:K
            for g  in 1:k
                like_var[k,g] = rand(InverseGamma(1,1))
                like_var[g,k] = like_var[k,g]
            end
        end

        B = zeros(K,K)
        for k in 1:K
            B[k,k] = randn()*0.2+1.0
            for g in 1:k-1
                B[k,g] = randn()*0.2
                B[g,k] = B[k,g]
            end
        end

        # here we define the flux model that maps X into θ
        Γ   = Chain(Dense(P, 32, relu), Dense(32, K))
        ps  = Flux.params(Γ)
        #opt = ADAM(0.01) # the value in brackts is the learnin rate for the optmizer

        #Γ = zeros(K,P)
        #for k in 1:K
        #    Γ[k,:] .= randn(P)* sqrt(σ_2[k])
        #end

        μ = zeros(K,N*n_regions)
        #μ = Γ * X;
        μ = Γ(Float32.(X));
        for i_region in 1:n_regions
            for i in 1:N
                ϕ[i_region][i,i,:] .= 0
            end
        end
        #println(μ)
        ###### end of initialization

        elbows, det_Sigma, det_nu = run_VEM_gauss_NN!(n_iter, ϕ, λ, ν, Σ, B, like_var, μ, Y, X, Γ, ps, K, N, P, n_regions, R)

        μ = Γ(X);
        data_dir = "data/results/100k_multiregion_NN_regularized$(R)/"

        if !isdir(data_dir)
            mkdir(data_dir)
        end

        for i_region in 1:n_regions
            nu_matrix = zeros(N,K*K)
            for i in 1:N
                nu_matrix[i,:] .= [ν[i_region][:,:,i]...]
            end
            println((covariate_files[i_region][27:end]))

            open("$(data_dir)elbows_$(N)_$(K)$(covariate_files[i_region][27:end])", "a") do io
                writedlm(io, elbows[i_region,:]')
            end
            open("$(data_dir)nu_$(N)_$(K)$(covariate_files[i_region][27:end])", "a") do io
                writedlm(io, nu_matrix)
            end
            open("$(data_dir)lambda_$(N)_$(K)$(covariate_files[i_region][27:end])", "a") do io
                writedlm(io, λ[:,(i_region-1)*N+1:i_region*N])
            end
            open("$(data_dir)mu_$(N)_$(K)$(covariate_files[i_region][27:end])", "a") do io
                writedlm(io, μ[:,(i_region-1)*N+1:i_region*N])
            end
            open("$(data_dir)det_nu_$(N)_$(K)$(covariate_files[i_region][27:end])", "a") do io
                writedlm(io, det_nu[i_region]')
            end
        end

        open("$(data_dir)B_$(N)_$(K)_chr2_100k.txt", "a") do io
            writedlm(io, B)
        end
        open("$(data_dir)Sigma_$(N)_$(K)_chr2_100k.txt", "a") do io
            writedlm(io, Σ)
        end
        open("$(data_dir)like_var_$(N)_$(K)_chr2_100k.txt", "a") do io
            writedlm(io, like_var)
        end

        open("$(data_dir)det_Sigma_$(N)_$(K)_chr2_100k.txt", "a") do io
            writedlm(io, det_Sigma')
        end

        model_state = Flux.state(Γ)
        jldsave("$(data_dir)$(i_run)_Gamma_$(N)_$(K)_100k.jld2"; model_state)

        #@save "$(data_dir)$(i_run)_Gamma_$(N)_$(K)_chr2_100k.bson" Γ

    end
end
