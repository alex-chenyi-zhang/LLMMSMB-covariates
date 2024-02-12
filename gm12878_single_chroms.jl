include("src/VEM_main.jl")

R = 0.1
Ks = [2, 3, 4, 5, 6, 10, 12]
iters = [250, 250, 250, 300, 300, 350, 350]
chrs = [12, 13, 14, 15, 16, 17, 18, 19, 20 , 21 ,22]
dirX = "data/12878_input_files/X_files_12878_chr"
dirY = "data/12878_input_files/Y_files_12878_chr"

res = ["_100k.txt", "_100k.txt", "_100k.txt", "_100k.txt", "_100k.txt", "_100k.txt", "_100k.txt", "_100k.txt", "_100k.txt", "_100k.txt", "_100k.txt"]

Threads.@threads for j in 1:11
    for i in 1:7
    	run_inference_gauss_multi_NN(iters[i], 6, "$(dirX)$(chrs[j])$(res[j])", "$(dirY)$(chrs[j])$(res[j])", Ks[i], R)
    end
end