include("src/VEM_main.jl")

Threads.@threads for R in [0.0001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
	run_inference_gauss_multi_NN(350, 1, 400, 6, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 14, R)
end

Threads.@threads for R in [0.0001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
	run_inference_gauss_multi(350, 1, 400, 6, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 16, R)
end

#=
Threads.@threads for R in [0.05, 0.1, 0.2, 0.5, 1.0 , 2.0]
	run_inference_gauss_multi_NN(350, 1, 400, 6, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 10, R)
end

Threads.@threads for R in [0.05, 0.1, 0.2, 0.5, 1.0 , 2.0]
	run_inference_gauss_multi(350, 1, 400, 6, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 10, R)
end

Threads.@threads for R in [0.05, 0.1, 0.2, 0.5, 1.0 , 2.0]
	run_inference_gauss_multi_NN(350, 1, 400, 6, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 12, R)
end

Threads.@threads for R in [0.05, 0.1, 0.2, 0.5, 1.0 , 2.0]
	run_inference_gauss_multi(350, 1, 400, 6, "data/X_2_files_chr2_100k_2.txt", "data/Y_files_chr2_100k_2.txt", 12, R)
end=#
