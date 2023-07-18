include("VEM_main.jl")
 ks = [12]#, 7, 8]
 niters = [350]#, 600, 600]

names = ["chr2_1000_46000_100k.txt", "chr2_10000_55000_100k.txt", "chr2_19000_64000_100k.txt",
		"chr2_28000_73000_100k.txt", "chr2_37000_82000_100k.txt"]
names_2 = ["chr2_1000_46000_100k_oe_uf3.txt", "chr2_10000_55000_100k_oe_uf3.txt", "chr2_19000_64000_100k_oe_uf3.txt",
			"chr2_28000_73000_100k_oe_uf3.txt", "chr2_37000_82000_100k_oe_uf3.txt"]
for i_name in 1:length(names)
	name = names[i_name]
	name_2 = names_2[i_name]
	println(name)
	println(name_2)
	for i_classes in 1:length(ks)
		run_inference_gauss(niters[i_classes], 1, 500, 1, "data/input/X_"*name, "data/input/Y_log_"*name_2,ks[i_classes])
		#run_inference_gauss(niters[i_classes], 1, 500, 4, "data/input/Y_log_"*name_2, ks[i_classes])

		#run_inference(niters[i_classes], 1, 1000, 10, "data/input/X_gaussian_noise_95_105.txt", "data/input/Y_bin_"*name, ks[i_classes])
	end
#end
end
