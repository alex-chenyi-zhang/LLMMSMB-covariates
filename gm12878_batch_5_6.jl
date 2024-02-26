include("src/VEM_main.jl")

R = 0.1
# Ks = [14]
# iters = [950]

Ks = [5, 6]
iters = [600, 700]

dirX = "data/HCT116_input_files/X_files_HCT116"
dirY = "data/HCT116_input_files/Y_files_HCT116"

suffix = ["_even_100k.txt", "_odd_100k.txt"]

Threads.@threads for j in 1:2
    for i in 1:2
        run_inference_gauss_multi_NN(iters[i], 6, "$(dirX)$(suffix[j])", "$(dirY)$(suffix[j])", Ks[i], R)
    end
end
