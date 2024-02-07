include("src/VEM_main.jl")

R = 0.1
Ks = [2, 3, 5, 6, 10, 12]
iters = [250, 250, 300, 300, 350, 350]
chrs = [2, 10, 14, 21]
dir = "data/12878_input_files/"
res = ["_100k.txt", "_100k.txt", "_100k.txt", "_50k.txt"]

Threads.@threads for j in 4
    for i in 1:6
    	run_inference_gauss_multi_NN(iters[i], 6, "$(dir)X_files_12878_chr$(chrs[j])$(res[j])", "$(dir)Y_files_12878_chr$(chrs[j])$(res[j])", Ks[i], R)
    end
end
