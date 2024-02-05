include("src/VEM_main.jl")

R = 0.1
Ks = [2, 3, 5, 6, 10, 12]
iters = [250, 250, 300, 300, 350, 350]
chrs = [2, 10, 14]
dir = "data/IMR90_input_files/"
for i in 1:6
    Threads.@threads for chr in chrs
        run_inference_gauss_multi_NN(iters[i], 6, "$(dir)X_files_IMR90_chr$(chr)_100k.txt", "$(dir)Y_files_IMR90_chr$(chr)_100k.txt", Ks[i], R)
    end
end
