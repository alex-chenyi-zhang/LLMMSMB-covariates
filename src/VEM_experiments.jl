include("VEM_main.jl")
 ks = [6, 8, 10, 12]#, 7, 8]
 niters = [450, 500, 550, 600]#, 600, 600]

name = "chr2_1000_66000_100k.txt"
name_2 = "chr2_1000_66000_100k_oe_gf2.txt"
println(name)
println(name_2)
for i_classes in 1:length(ks)
	run_inference_gauss(niters[i_classes], 1, 650, 8, "data/input/X_"*name, "data/input/Y_log_"*name_2, ks[i_classes])
	#run_inference(niters[i_classes], 1, 1000, 10, "data/input/X_gaussian_noise_95_105.txt", "data/input/Y_bin_"*name, ks[i_classes])
end
#end
