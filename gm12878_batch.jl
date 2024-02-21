include("src/VEM_main.jl")

R = 0.1
#Ks = [2, 3, 4, 5, 6, 10, 12, 14]
#iters = [400, 400, 500, 600, 700, 850, 950, 950]

Ks = [14]
iters = [950]

# Ks = [3, 12]
# iters = [400, 950]
#
# Ks = [4, 10]
# iters = [500, 850]
#
# Ks = [5, 6]
# iters = [600, 700]

dirX = "data/12878_input_files/X_files_12878"
dirY = "data/12878_input_files/Y_files_12878"

suffix = ["_even_100k.txt", "_odd_100k.txt", "_even_100k_nofilt.txt", "_odd_100k_nofilt.txt",]

Threads.@threads for j in 1:4
    for i in 1:8
        run_inference_gauss_multi_NN(iters[i], 12, "$(dirX)$(suffix[j])", "$(dirY)$(suffix[j])", Ks[i], R)
    end
end
