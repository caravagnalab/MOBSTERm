rm(list = ls())
args <- commandArgs(TRUE)
N <- as.numeric(args[1])
K <- as.numeric(args[2])
D <- as.numeric(args[3])
p <- as.numeric(args[4])
cov <- as.numeric(args[5])

# N, K, D, p, cov

require(dplyr)
library(tidyr)
library(randnet)

p = 1.0
cov = 100
D = 2
N = 5000
K = 4
idx = 0

setwd('test_generative/mobsterm_and_mobster')


NMI_values <- c()
N_list = c(5000,10000,15000)
K_list = c(4,6,8,15)
D_list = c(2,3,4)
purity_list = c(0.7,0.9,1.0)
cov_list = c(70,100)

for (N in N_list) {
  for (K in K_list) {
    for (D in D_list) {
      for (p in purity_list) {
        for (cov in cov_list) {
          print(paste0(N, " ",K, " ", D, " ",cov,  " ", p))
          csv_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "_cov_", cov, "/D_", D, "/csv_mobster_viber/")
          if (p == 1){
            csv_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "0_cov_", cov, "/D_", D, "/csv_mobster_viber/")
          }
          NMI_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "_cov_", cov, "/D_", D, "/nmi_viber/")
          if (p == 1){
            NMI_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "0_cov_", cov, "/D_", D, "/nmi_viber/")
          }
          if (!dir.exists(NMI_path)) {
            dir.create(NMI_path, recursive = TRUE)
          }
          NMI_values <- c()
          for (idx in 0:14){
            file_name = paste0(csv_path, "N_", N, "_K_", K, "_D_", D, "_df_", idx, ".csv")
            if (file.exists(file_name)) {
              final_df = read.csv(paste0(csv_path, "N_", N, "_K_", K, "_D_", D, "_df_", idx, ".csv"))#, row.names = FALSE)
            
              # viber_labels <- viber_labels[viber_labels != 0]
              final_df <- final_df[final_df$MVPredDist != 0, ]
              
              true_cluster = final_df$true_cluster
              viber_labels <- final_df$MVPredDist
              
              # viber_labels <- viber_fit$labels$cluster.Binomial
              # viber_labels <- as.integer(sub("C", "", viber_labels))
              if (length(unique(viber_labels)) == 1) {
                viber_labels[1] <- 0
              }
              # compute NMI between viber_fit$labels and df_viber$true_cluster
              nmi = randnet::NMI(true_cluster,viber_labels)
              NMI_values = c(NMI_values,nmi)
            }
          }
          
          write.csv(data.frame(NMI = NMI_values), paste0(NMI_path, "N_", N, "_K_", K, "_D_", D, ".csv"), row.names = FALSE)
        }}}}}
