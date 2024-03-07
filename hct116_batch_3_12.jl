include("src/VEM_main.jl")

R = 0.1
# Ks = [14]
# iters = [950]

Ks = [3, 12]
iters = [500, 1050]

dirX = "data/HCT116_input_files/X_files_HCT116"
dirY = "data/HCT116_input_files/Y_files_HCT116"

suffix = ["_100k.txt"]

Threads.@threads for i in 1:2
    j = 1
    run_inference_gauss_multi_NN(iters[i], 6, "$(dirX)$(suffix[j])", "$(dirY)$(suffix[j])", Ks[i], R)
end
