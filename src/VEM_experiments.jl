include("VEM_main.jl")
 ks = [4, 5, 6]#, 7, 8]
 niters = [50, 50, 60]#, 600, 600]
 #niters = [5,5]#,5,5,5]
 file_names_2 = ["1000_11000_10k_2.txt", "95000_105000_10k_2.txt", "100000_110000_10k_2.txt", "150000_160000_10k_2.txt", "221000_231000_10k_2.txt", "218320_224090_5k_2.txt"]
 file_names = ["1000_11000_10k.txt", "95000_105000_10k.txt", "100000_110000_10k.txt", "150000_160000_10k.txt", "221000_231000_10k.txt", "218320_224090_5k.txt"]
 #for name in file_names
 i = 6
name = "1000_81000_100k_filter_offset.txt"#file_names[i]
name_2 = "1000_81000_100k_oe_gf.txt"#file_names_2[i]
println(name)
println(name_2)
Threads.@threads for i_classes in 1:length(ks)
	run_inference_gauss(niters[i_classes], 1, 300, 3, "data/input/X_"*name, "data/input/Y_log_"*name_2, ks[i_classes])
	#run_inference(niters[i_classes], 1, 1000, 10, "data/input/X_gaussian_noise_95_105.txt", "data/input/Y_bin_"*name, ks[i_classes])
end
#end
