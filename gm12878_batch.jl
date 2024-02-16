include("src/VEM_main.jl")

R = 0.1
Ks = [2, 3, 4, 5, 6, 10, 12, 14]
iters = [200, 200, 200, 300, 300, 300, 350, 350]
dirX = "data/12878_input_files/X_files_12878"
dirY = "data/12878_input_files/Y_files_12878"

suffix = ["_16even22_100k.txt", "_15odd21_100k.txt", "_16even22_100k_nofilt.txt", "_15odd21_100k_nofilt.txt",]

Threads.@threads for j in 1:4
    for i in 1:8
        run_inference_gauss_multi_NN(iters[i], 12, "$(dirX)$(suffix[j])", "$(dirY)$(suffix[j])", Ks[i], R)
    end
end
