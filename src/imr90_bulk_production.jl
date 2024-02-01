include("src/VEM_main.jl")

R = 0.1
Ks = [2, 6, 10, 12, 14]
iters = [300, 300, 350, 350, 400]
Threads.@threads for i in 1:5
    run_inference_gauss_multi_NN(iters[i], 6, "data/X_files_IMR90_chr21_50k.txt", "data/Y_files_IMR90_chr21_50k.txt", Ks[i], R)
end

run_inference_gauss_multi_NN(400, 6, "data/X_files_IMR90_chr2_250k.txt", "data/Y_files_IMR90_chr2_250k.txt", 14, R)
