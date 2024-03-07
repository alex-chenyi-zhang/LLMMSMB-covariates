include("src/VEM_main.jl")

R = 0.1
Ks = [14]
iters = [1050]

dirX = "data/GM12878_input_files/X_files_GM12878"
dirY = "data/GM12878_input_files/Y_files_GM12878"

suffix = ["_100k.txt"]

i = 1
j = 1
run_inference_gauss_multi_NN(iters[i], 6, "$(dirX)$(suffix[j])", "$(dirY)$(suffix[j])", Ks[i], R)
