rm(list = ls())
args <- commandArgs(TRUE)
N <- as.numeric(args[1])
K <- as.numeric(args[2])
D <- as.numeric(args[3])
p <- as.numeric(args[4])
cov <- as.numeric(args[5])

# N, K, D, p, cov

require(mobster)
require(dplyr)
require(ggplot2)
library(tidyr)
require(VIBER)
require(ggpubr)
library(randnet)

# p = 1.0
# cov = 100
# D = 2
# N = 5000
# K = 4
# idx = 1

# p_07_cov_70/D_3/accuracy_mobster/N_15000_K_6_D_3

csv_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "_cov_", cov, "/D_", D, "/csv_mobster_viber/")
if (p == 1){
  csv_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "0_cov_", cov, "/D_", D, "/csv_mobster_viber/")
}
if (!dir.exists(csv_path)) {
  dir.create(csv_path, recursive = TRUE)
}

accuracy_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "_cov_", cov, "/D_", D, "/accuracy_mobster_viber/")
if (p == 1){
  accuracy_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "0_cov_", cov, "/D_", D, "/accuracy_mobster_viber/")
}
if (!dir.exists(accuracy_path)) {
  dir.create(accuracy_path, recursive = TRUE)
}

NMI_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "_cov_", cov, "/D_", D, "/nmi_mobster_viber/")
if (p == 1){
  NMI_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "0_cov_", cov, "/D_", D, "/nmi_mobster_viber/")
}
if (!dir.exists(NMI_path)) {
  dir.create(NMI_path, recursive = TRUE)
}

NMI_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "_cov_", cov, "/D_", D, "/nmi_mobster_viber/")
if (p == 1){
  NMI_path = paste0("./results/p_", gsub("\\.", "", as.character(p)), "0_cov_", cov, "/D_", D, "/nmi_mobster_viber/")
}
if (!dir.exists(NMI_path)) {
  dir.create(NMI_path, recursive = TRUE)
}


plot_path = paste0("./plots/p_", gsub("\\.", "", as.character(p)), "_cov_", cov, "/D_", D, "/fit_mobster/")
if (p == 1){
  plot_path = paste0("./plots/p_", gsub("\\.", "", as.character(p)), "0_cov_", cov, "/D_", D, "/fit_mobster/")
}
if (!dir.exists(plot_path)) {
  dir.create(plot_path, recursive = TRUE)
}

plot_viber_path = paste0("./plots/p_", gsub("\\.", "", as.character(p)), "_cov_", cov, "/D_", D, "/fit_viber/")
if (p == 1){
  plot_viber_path = paste0("./plots/p_", gsub("\\.", "", as.character(p)), "0_cov_", cov, "/D_", D, "/fit_viber/")
}
if (!dir.exists(plot_viber_path)) {
  dir.create(plot_viber_path, recursive = TRUE)
}

# setwd('~/orfeo_remote/scratch/tesimagistrale/subclonal_deconvolution_mv/test_generative/final_our_model')

accuracy_values <- c()
NMI_values <- c()
NMI_values_complete <- c()

for (idx in 0:14){
  # idx = 6
  folder_name = paste0("./results/p_", gsub("\\.", "", as.character(p)), "_cov_", cov, "/D_", D, "/csv/")
  if (p == 1){
    folder_name = paste0("./results/p_", gsub("\\.", "", as.character(p)), "0_cov_", cov, "/D_", D, "/csv/")
  }
  
  file_name = paste0(folder_name, "N_", N, "_K_", K, "_D_", D, "_df_", idx, ".csv")
  
  df <- read.csv(file_name, stringsAsFactors = FALSE)
  
  parse_list_column <- function(x) {
    lapply(x, function(s) {
      as.numeric(strsplit(gsub("\\[|\\]", "", s), ",\\s*")[[1]])
    })
  }
  
  NV_list <- parse_list_column(df$NV)
  DP_list <- parse_list_column(df$DP)
  
  true_dist_list = parse_list_column(df$True_distribution)
  
  NV <- do.call(rbind, NV_list)
  DP <- do.call(rbind, DP_list)
  true_dist = do.call(rbind, true_dist_list)
  
  new_colnames <- paste0("S", 1:D)
  
  colnames(NV) <- new_colnames
  colnames(DP) <- new_colnames
  colnames(true_dist) <- new_colnames
  
  NV <- as.data.frame(NV)
  DP <- as.data.frame(DP)
  true_dist <- as.data.frame(true_dist)
  
  NV$original_index <- seq_len(nrow(NV))
  DP$original_index <- seq_len(nrow(DP))
  true_dist$original_index <- seq_len(nrow(true_dist))
  
  
  n <- length(NV[[1]])
  final_df <- data.frame(matrix(nrow = n, ncol = 0))
  
  for (d in 1:D) {
    # d = 2
    col = paste0("NV_S", d)
    final_df[[col]] = NV[[d]]
    
    col = paste0("DP_S", d)
    final_df[[col]] = DP[[d]]
    
    col = paste0("TrueDist_S", d)
    final_df[[col]] = true_dist[[d]]
    
    col = paste0("VAF_S", d)
    final_df[[col]] = NV[[d]]/DP[[d]]
    
    col = "true_cluster"
    final_df[[col]] = df$True_cluster
  }
  
  for (d in 1:D) {
    # print(NV[d])
    # d = 2
    vaf <- data.frame(VAF = (NV[d]/DP[d]))/p
    colnames(vaf) <- "VAF"
    vaf['NV'] = NV[d]
    vaf['DP'] = DP[d]
    vaf['true_dist'] = true_dist[d]
    vaf['original_index'] = true_dist['original_index']
    
    # vaf$original_index <- seq_len(nrow(vaf))
    zero_indices <- which(vaf$VAF == 0)
    # vaf_mobster <- vaf[which(vaf$VAF != 0), ]
    # vaf_mobster <- vaf[vaf$VAF != 0, , drop = FALSE]
    vaf_mobster <- vaf[vaf$VAF != 0 & vaf$VAF <= 1, , drop = FALSE]
    mobster:::template_parameters_fast_setup()
    
    # FIRST ADJUST FOR PURITY!!
    fit = mobster_fit(
      vaf_mobster,
      auto_setup = "FAST",
      seed = 12345
    )
    
    pl = plot(fit$best)
    # pl
    ggsave(paste0(plot_path, "N_", N, "_K_", K, "_D_", D, "_mobster_", idx, "_S", d, ".png"),plot = pl, dpi = 600, width = 8, height = 5)
    labels = fit$best$data
    vaf_with_labels <- merge(vaf, labels, by = c("original_index", "VAF", "NV", "DP", "true_dist"), all.x = TRUE)
    
    vaf_with_labels$cluster_binary <- ifelse(vaf_with_labels$cluster == "Tail", 0, 1)
    vaf_with_labels$cluster_binary[is.na(vaf_with_labels$cluster)] <- 1
    vaf_with_labels$cluster[is.na(vaf_with_labels$cluster)] <- 'P'
    
    vaf_with_labels <- vaf_with_labels %>%
      arrange(original_index)
    
    if (d==1){
      final_df$original_index <- vaf_with_labels$original_index
    }
    
    col = paste0("MPredDist_S", d)
    final_df[[col]] = vaf_with_labels$cluster
    
    col = paste0("MTypePredDist_S", d)
    final_df[[col]] = vaf_with_labels$cluster_binary
    
    ### Now compute the accuracy for true_dist == 0
    # Extract rows with true_dist == 0
    subset <- vaf_with_labels[vaf_with_labels$true_dist == 0, ]
    
    # Count how many of those have cluster == 0
    correct <- sum(subset$cluster_binary == 0)
    # Total number of entries where true_dist == 0
    total <- nrow(subset)
    # Compute accuracy
    accuracy <- correct / total
    accuracy_values = c(accuracy_values,accuracy)
    
  }
  
  # write.csv(final_df, paste0(csv_path, "N_", N, "_K_", K, "_D_", D, "_df_", idx, ".csv"), row.names = FALSE)
  
  ### Now with the mutations where cluster != Tail, run VIBER on all the samples we have
  # Then compute NMI
  # Extract rows of final_df where all the columns PredDist_Sx are != Tail
  
  # Identify columns starting with "MPredDist_S" (which contain MOBSTER labels)
  pred_cols <- grep("^MPredDist_S", names(final_df), value = TRUE)
  rows_with_tail <- apply(final_df[, pred_cols], 1, function(row) any(row == "Tail", na.rm = TRUE))
  
  # col_MV = "MVPredDist"
  # final_df[[col_MV]] = 0
  
  
  # Remove those rows
  df_viber <- final_df[!rows_with_tail, ]
  
  # Create a logical vector where TRUE means the row has any "Tail" in PredDist columns
  # is_tail <- apply(final_df[pred_cols], 1, function(row) any(row == "Tail"))
  # df_viber <- final_df[!is_tail, ]
  
  # options(easypar.parallel = TRUE)
  # # VIBER fit
  NVs = df_viber %>% select(starts_with('NV'))
  DPs = df_viber %>% select(starts_with('DP'))
  
  colnames(DPs) = colnames(NVs) = paste0("S", 1:D)
  
  viber_fit = VIBER::variational_fit(x = NVs, y = DPs, samples = D, epsilon_conv = 1e-10, K = K)
  # saveRDS(viber_fit, file = "./Set7/Set7_mobster_viber_fit.rds")
  # Plot a 3x2 figure -- raw fit (all clusters)
  if (D == 4){
    pl_viber = ggarrange(
      plotlist = plot(
        viber_fit
      ),
      ncol = 3,
      nrow = 2)
    w = 15
    h = 10
  } else if (D == 3){
    pl_viber = ggarrange(
      plotlist = plot(
        viber_fit
      ),
      ncol = D,
      nrow = 1)
    w = 15
    h = 5
  }else if (D == 2){
    pl_viber = ggarrange(
      plotlist = plot(
        viber_fit
      ),
      ncol = 1,
      nrow = 1)
    w = 15
    h = 5
  }
  
  ggsave(paste0(plot_viber_path, "N_", N, "_K_", K, "_D_", D, "_viber_", idx, ".png"),plot = pl_viber, dpi = 600, width = w, height = h)
  
  viber_labels <- viber_fit$labels$cluster.Binomial
  viber_labels <- as.integer(sub("C", "", viber_labels))
  if (length(unique(viber_labels)) == 1) {
    viber_labels[1] <- 0
  }
  # compute NMI between viber_fit$labels and df_viber$true_cluster
  nmi = randnet::NMI(df_viber$true_cluster,viber_labels)
  NMI_values = c(NMI_values,nmi)
  
  # Here I need to create a final table with VIBER labels, and set to 0 mutations not included in VIBER, i.e. tail mutations
  # Combine df_viber and viber_labels
  df_viber_labeled <- df_viber %>%
    mutate(MVPredDist = viber_labels)
  
  final_df <- final_df %>%
    left_join(df_viber_labeled %>% select(original_index,MVPredDist), by = "original_index")
    
  final_df <- final_df %>%
    mutate(MVPredDist = replace_na(MVPredDist, 0))
  
  nmi = randnet::NMI(final_df$true_cluster,final_df$MVPredDist)
  NMI_values_complete = c(NMI_values_complete,nmi)
  
  write.csv(final_df, paste0(csv_path, "N_", N, "_K_", K, "_D_", D, "_df_", idx, ".csv"), row.names = FALSE)
}

write.csv(data.frame(accuracy = accuracy_values), paste0(accuracy_path, "N_", N, "_K_", K, "_D_", D, ".csv"), row.names = FALSE)
write.csv(data.frame(NMI = NMI_values_complete), paste0(NMI_path, "N_", N, "_K_", K, "_D_", D, ".csv"), row.names = FALSE)
