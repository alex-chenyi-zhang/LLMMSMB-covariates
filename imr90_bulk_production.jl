include("src/VEM_main.jl")

R = 0.1
run_inference_gauss_multi_NN(300, 6, "data/X_files_IMR90_chr2_100k.txt", "data/Y_files_IMR90_chr2_100k.txt", 2, R)
run_inference_gauss_multi_NN(300, 6, "data/X_files_IMR90_chr2_100k.txt", "data/Y_files_IMR90_chr2_100k.txt", 6, R)
run_inference_gauss_multi_NN(350, 6, "data/X_files_IMR90_chr2_100k.txt", "data/Y_files_IMR90_chr2_100k.txt", 10, R)
run_inference_gauss_multi_NN(350, 6, "data/X_files_IMR90_chr2_100k.txt", "data/Y_files_IMR90_chr2_100k.txt", 12, R)
