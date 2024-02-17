include("src/VEM_main.jl")

R = 0.1
Ks = [4, 5, 6, 10, 12]
iters = [450, 500, 500, 550, 550]
#chrs = [1, 2, 3, 4, 5, 6, 7, 8, 9 , 10 , 11]
chrs = [12]
dirX = "data/12878_input_files/X_files_12878_chr"
dirY = "data/12878_input_files/Y_files_12878_chr"

#res = ["_500k.txt", "_500k.txt", "_500k.txt", "_500k.txt", "_500k.txt", "_500k.txt", "_500k.txt", "_500k.txt", "_500k.txt", "_500k.txt", "_500k.txt"]
res = ["_100k.txt"]
#Threads.@threads for j in 1:11
for j in 1:1
    for i in 1:7
    	run_inference_gauss_multi_NN(iters[i], 6, "$(dirX)$(chrs[j])$(res[j])", "$(dirY)$(chrs[j])$(res[j])", Ks[i], R)
    end
end
