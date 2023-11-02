include("src/VEM_main.jl")

run_inference_gauss_multi_NN(300, 1, 400, 4, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 8)

run_inference_gauss_multi_NN(300, 1, 400, 4, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 10)

run_inference_gauss_multi(300, 1, 400, 4, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 8)

run_inference_gauss_multi(300, 1, 400, 4, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 10)
