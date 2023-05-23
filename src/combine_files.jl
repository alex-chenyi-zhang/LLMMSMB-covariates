using DelimitedFiles
N = 650
for K in [10]
    #reg = "_$(N)_$(K)_1000_81000_100k_filter_offset.txt"
    data_dir = "data/preliminary_results/log_chr2_1000_66000_100k_oe_gf2/"
    reg = "_$(N)_$(K)_chr2_1000_66000_100k.txt"
    #reg2 = "_$(N)_$(K)_chr2_1000_66000_100k.txt"
    println(K, "\n")
    n_iter = 8
    for iter in 1:n_iter
        println("iter: ", iter)
        #### thetas
        println("thetas")
        io = open(data_dir*"$(iter)_thetas"*reg, "r")
        thetas = readdlm(io, Float64)
        close(io)

        open(data_dir*"thetas"*reg, "a") do io
            writedlm(io, thetas)
        end

        #### elbows
        println("elbows")
        io = open(data_dir*"$(iter)_elbows"*reg, "r")
        elbows = readdlm(io, Float64)
        close(io)

        open(data_dir*"elbows"*reg, "a") do io
            writedlm(io, elbows)
        end

        #### nu
        println("nu")
        io = open(data_dir*"$(iter)_nu"*reg, "r")
        nu_matrix = readdlm(io, Float64)
        close(io)

        open(data_dir*"nu"*reg, "a") do io
            writedlm(io, nu_matrix)
        end

        #### lambda
        println("lambda")
        io = open(data_dir*"$(iter)_lambda"*reg, "r")
        lambda = readdlm(io, Float64)
        close(io)

        open(data_dir*"lambda"*reg, "a") do io
            writedlm(io, lambda)
        end

        #### pred_map
        println("pred_map")
        io = open(data_dir*"$(iter)_pred_map"*reg, "r")
        pred_map = readdlm(io, Float64)
        close(io)

        open(data_dir*"pred_map"*reg, "a") do io
            writedlm(io, pred_map)
        end

        #### B
        println("B")
        io = open(data_dir*"$(iter)_B"*reg, "r")
        B = readdlm(io, Float64)
        close(io)

        open(data_dir*"B"*reg, "a") do io
            writedlm(io, B)
        end

        #### Sigma
        println("Sigma")
        io = open(data_dir*"$(iter)_Sigma"*reg, "r")
        Sigma = readdlm(io, Float64)
        close(io)

        open(data_dir*"Sigma"*reg, "a") do io
            writedlm(io, Sigma)
        end

        #### Gamma
        println("Gamma")
        io = open(data_dir*"$(iter)_Gamma"*reg, "r")
        Gamma = readdlm(io, Float64)
        close(io)

        open(data_dir*"Gamma"*reg, "a") do io
            writedlm(io, Gamma)
        end

        #### sigma
        println("sigma")
        io = open(data_dir*"$(iter)_sigma"*reg, "r")
        sigma_2 = readdlm(io, Float64)
        close(io)

        open(data_dir*"sigma"*reg, "a") do io
            writedlm(io, sigma_2)
        end

        #### like_var
        println("like_var")
        io = open(data_dir*"$(iter)_like_var"*reg, "r")
        like_var = readdlm(io, Float64)
        close(io)

        open(data_dir*"like_var"*reg, "a") do io
            writedlm(io, like_var)
        end
    end

    for iter in 1:n_iter
        rm(data_dir*"$(iter)_thetas"*reg)
        rm(data_dir*"$(iter)_elbows"*reg)
        rm(data_dir*"$(iter)_nu"*reg)
        rm(data_dir*"$(iter)_lambda"*reg)
        rm(data_dir*"$(iter)_pred_map"*reg)
        rm(data_dir*"$(iter)_B"*reg)
        rm(data_dir*"$(iter)_Sigma"*reg)
        rm(data_dir*"$(iter)_Gamma"*reg)
        rm(data_dir*"$(iter)_sigma"*reg)
        rm(data_dir*"$(iter)_like_var"*reg)

    end
end
