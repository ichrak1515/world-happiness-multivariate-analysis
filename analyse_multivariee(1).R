# =============================================================================
# MINI-PROJET – Analyse Multivariée du Bonheur Mondial
# Module  : Méthodes Statistiques et Étude de Données
# Niveau  : Ingénieur Data Science & IA
# Dataset : World Happiness Report (250 pays × 9 variables)
# Outils  : R | FactoMineR | factoextra | cluster | ggplot2 | corrplot
# =============================================================================


# -----------------------------------------------------------------------------
# 0. INSTALLATION DES PACKAGES (décommenter si première utilisation)
# -----------------------------------------------------------------------------
# install.packages(c("tidyverse", "FactoMineR", "factoextra", "cluster",
#                    "corrplot", "gridExtra", "dendextend", "NbClust"))


# -----------------------------------------------------------------------------
# 1. CHARGEMENT DES PACKAGES
# -----------------------------------------------------------------------------
library(tidyverse)
library(FactoMineR)
library(factoextra)
library(cluster)
library(corrplot)
library(gridExtra)
library(dendextend)
library(NbClust)


# -----------------------------------------------------------------------------
# 2. IMPORT DU DATASET
# -----------------------------------------------------------------------------
df <- read.csv("C:/Users/ichra/Downloads/happiness_data.csv", stringsAsFactors = FALSE)

cat("Dimensions :", dim(df), "\n")
cat("Variables  :", names(df), "\n")
head(df)


# -----------------------------------------------------------------------------
# 3. ÉTAPE 1 – COMPRÉHENSION DU PROBLÈME
# -----------------------------------------------------------------------------
cat("\n===== STATISTIQUES DESCRIPTIVES =====\n")
summary(df)

num_vars <- c("Happiness_Score", "GDP_per_Capita", "Social_Support",
              "Healthy_Life_Expectancy", "Freedom", "Generosity",
              "Corruption_Perception", "Unemployment_Rate", "Education_Index")

cat("\nDistribution des régions :\n")
print(table(df$Region))


# -----------------------------------------------------------------------------
# 4. ÉTAPE 2 – PRÉPARATION DES DONNÉES
# -----------------------------------------------------------------------------

# Valeurs manquantes
cat("\nValeurs manquantes par variable :\n")
print(colSums(is.na(df)))

# Outliers (règle IQR)
cat("\nNombre d'outliers par variable :\n")
for (var in num_vars) {
  Q1 <- quantile(df[[var]], 0.25)
  Q3 <- quantile(df[[var]], 0.75)
  IQR_val <- Q3 - Q1
  n_out <- sum(df[[var]] < (Q1 - 1.5*IQR_val) | df[[var]] > (Q3 + 1.5*IQR_val))
  cat(sprintf("  %-30s : %d outlier(s)\n", var, n_out))
}

# Normalisation z-score
X_scaled <- scale(df[, num_vars])
cat("\nVérification normalisation (moyennes ≈ 0) :\n")
print(round(colMeans(X_scaled), 6))


# -----------------------------------------------------------------------------
# 5. FIGURE 1 – DISTRIBUTIONS DES VARIABLES
# -----------------------------------------------------------------------------
plots_hist <- lapply(num_vars, function(var) {
  ggplot(df, aes_string(x = var)) +
    geom_histogram(bins = 25, fill = "#2E86AB", color = "white", alpha = 0.85) +
    geom_vline(aes(xintercept = mean(df[[var]])),
               color = "black", linetype = "dashed", linewidth = 0.8) +
    labs(title = gsub("_", " ", var), x = "", y = "Fréquence") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", size = 10))
})

png("fig1_distributions.png", width = 1600, height = 1300, res = 120)
grid.arrange(grobs = plots_hist, ncol = 3)
dev.off()
cat("Figure 1 sauvegardée\n")


# -----------------------------------------------------------------------------
# 6. FIGURE 2 – MATRICE DE CORRÉLATIONS
# -----------------------------------------------------------------------------
corr_matrix <- cor(df[, num_vars])

png("fig2_correlation.png", width = 1200, height = 1100, res = 120)
corrplot(corr_matrix,
         method = "color", type = "lower",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         number.cex = 0.75,
         col = colorRampPalette(c("#C73E1D", "white", "#2E86AB"))(200),
         title = "Figure 2 – Matrice de Corrélations (Pearson)",
         mar = c(0,0,2,0))
dev.off()
cat("Figure 2 sauvegardée\n")

cat("\nCorrélations avec Happiness_Score :\n")
print(sort(corr_matrix["Happiness_Score",], decreasing = TRUE))


# -----------------------------------------------------------------------------
# 7. ÉTAPE 3 – ANALYSE EN COMPOSANTES PRINCIPALES (ACP)
# -----------------------------------------------------------------------------

res_pca <- PCA(df[, num_vars], scale.unit = TRUE, graph = FALSE)

cat("\n===== VARIANCE EXPLIQUÉE =====\n")
variance_df <- data.frame(
  PC            = paste0("PC", 1:9),
  Valeur_propre = round(res_pca$eig[,1], 3),
  Variance_pct  = round(res_pca$eig[,2], 2),
  Cumulative    = round(res_pca$eig[,3], 2)
)
print(variance_df)

cat(sprintf("\nPC1 seule = %.2f%% de la variance\n", res_pca$eig[1,2]))
cat(sprintf("PC1+PC2   = %.2f%% de la variance\n", res_pca$eig[2,3]))


# -----------------------------------------------------------------------------
# 8. FIGURE 3 – SCREE PLOT
# -----------------------------------------------------------------------------
png("fig3_screeplot.png", width = 1200, height = 700, res = 120)
p_scree <- fviz_eig(res_pca, addlabels = TRUE, ylim = c(0,75),
                    barfill = "#2E86AB", barcolor = "white",
                    linecolor = "#F18F01",
                    main = "Figure 3 – Éboulis des Valeurs Propres (Scree Plot)") +
  theme_minimal(base_size = 12)
print(p_scree)
dev.off()
cat("Figure 3 sauvegardée\n")


# -----------------------------------------------------------------------------
# 9. FIGURE 4 – CERCLE DES CORRÉLATIONS
# -----------------------------------------------------------------------------
png("fig4_cercle_correlations.png", width = 1000, height = 1000, res = 120)
p_var <- fviz_pca_var(res_pca,
                      col.var = "contrib",
                      gradient.cols = c("#C73E1D","#F18F01","#2E86AB"),
                      repel = TRUE,
                      title = "Figure 4 – Cercle des Corrélations (PC1 × PC2)") +
  theme_minimal(base_size = 12)
print(p_var)
dev.off()
cat("Figure 4 sauvegardée\n")

cat("\nContributions à PC1 :\n")
print(round(sort(res_pca$var$contrib[,1], decreasing=TRUE), 2))
cat("\nContributions à PC2 :\n")
print(round(sort(res_pca$var$contrib[,2], decreasing=TRUE), 2))


# -----------------------------------------------------------------------------
# 10. FIGURE 5 – BIPLOT
# -----------------------------------------------------------------------------
df$PC1 <- res_pca$ind$coord[,1]
df$PC2 <- res_pca$ind$coord[,2]

couleurs_region <- c(
  "Western Europe"            = "#2E86AB",
  "North America & ANZ"       = "#A23B72",
  "Latin America & Caribbean" = "#F18F01",
  "Central & Eastern Europe"  = "#C73E1D",
  "East Asia"                 = "#44BBA4",
  "Southeast Asia"            = "#E94F37",
  "Sub-Saharan Africa"        = "#6C3483"
)

png("fig5_biplot.png", width = 1400, height = 1100, res = 120)
p_biplot <- fviz_pca_biplot(res_pca,
                             geom.ind = "point",
                             col.ind  = df$Region,
                             palette  = couleurs_region,
                             addEllipses = TRUE,
                             ellipse.type = "convex",
                             col.var = "black",
                             repel = TRUE,
                             legend.title = "Région",
                             title = "Figure 5 – Biplot ACP : Individus et Variables") +
  theme_minimal(base_size = 11)
print(p_biplot)
dev.off()
cat("Figure 5 sauvegardée\n")


# -----------------------------------------------------------------------------
# 11. ÉTAPE 4 – CHOIX DU NOMBRE DE CLUSTERS
# -----------------------------------------------------------------------------

# Méthode du coude
png("fig6_elbow.png", width = 1000, height = 600, res = 120)
p_elbow <- fviz_nbclust(X_scaled, kmeans, method = "wss", k.max = 10,
                         linecolor = "#2E86AB") +
  geom_vline(xintercept = 4, linetype = "dashed", color = "red", linewidth = 1) +
  labs(title = "Figure 6 – Méthode du Coude (Elbow)") +
  theme_minimal(base_size = 12)
print(p_elbow)
dev.off()
cat("Figure 6 sauvegardée\n")

# Score de silhouette
png("fig7_silhouette_choix.png", width = 1000, height = 600, res = 120)
p_sil <- fviz_nbclust(X_scaled, kmeans, method = "silhouette", k.max = 10,
                       linecolor = "#F18F01") +
  labs(title = "Figure 7 – Score de Silhouette par k") +
  theme_minimal(base_size = 12)
print(p_sil)
dev.off()
cat("Figure 7 sauvegardée\n")


# -----------------------------------------------------------------------------
# 12. K-MEANS (k=4)
# -----------------------------------------------------------------------------
set.seed(42)
km4 <- kmeans(X_scaled, centers = 4, nstart = 25, iter.max = 100)
df$Cluster <- as.factor(km4$cluster)

cat("\n===== RÉSULTATS K-MEANS (k=4) =====\n")
cat("Taille des clusters :\n")
print(table(df$Cluster))

sil <- silhouette(km4$cluster, dist(X_scaled))
cat(sprintf("Score de silhouette moyen : %.4f\n", mean(sil[,3])))
cat(sprintf("Inertie expliquée         : %.2f%%\n", 100*km4$betweenss/km4$totss))

# Profils moyens
cat("\n===== PROFILS MOYENS DES CLUSTERS =====\n")
profils <- df %>%
  group_by(Cluster) %>%
  summarise(across(all_of(num_vars), ~round(mean(.), 3)), n = n())
print(profils)


# -----------------------------------------------------------------------------
# 13. FIGURE 8 – K-MEANS DANS L'ESPACE ACP
# -----------------------------------------------------------------------------
png("fig8_kmeans_pca.png", width = 1300, height = 1000, res = 120)
p_km <- fviz_cluster(km4, data = X_scaled,
                      ellipse.type = "convex",
                      palette = c("#2E86AB","#F18F01","#C73E1D","#44BBA4"),
                      ggtheme = theme_minimal(base_size = 12),
                      main = "Figure 8 – Clusters K-Means (k=4) dans l'Espace ACP",
                      repel = TRUE)
print(p_km)
dev.off()
cat("Figure 8 sauvegardée\n")


# -----------------------------------------------------------------------------
# 14. FIGURE 9 – SILHOUETTE PLOT
# -----------------------------------------------------------------------------
png("fig9_silhouette_plot.png", width = 1000, height = 800, res = 120)
p_sil_plot <- fviz_silhouette(sil,
                               palette = c("#2E86AB","#F18F01","#C73E1D","#44BBA4"),
                               ggtheme = theme_minimal(base_size = 11),
                               main = "Figure 9 – Diagramme de Silhouette (k=4)")
print(p_sil_plot)
dev.off()
cat("Figure 9 sauvegardée\n")


# -----------------------------------------------------------------------------
# 15. CLASSIFICATION ASCENDANTE HIÉRARCHIQUE (CAH)
# -----------------------------------------------------------------------------
set.seed(42)
sample_idx <- sample(1:nrow(X_scaled), 80)
dist_mat   <- dist(X_scaled[sample_idx,], method = "euclidean")
hc_ward    <- hclust(dist_mat, method = "ward.D2")

png("fig10_dendrogramme.png", width = 1600, height = 900, res = 120)
dend <- as.dendrogram(hc_ward)
dend <- color_branches(dend, k = 4,
                        col = c("#2E86AB","#F18F01","#C73E1D","#44BBA4"))
dend <- set(dend, "labels_cex", 0.55)
plot(dend,
     main = "Figure 10 – Dendrogramme CAH (Ward, n=80)",
     ylab = "Distance de Ward", xlab = "Observations")
abline(h = 8, col = "red", lty = 2, lwd = 2)
legend("topright", legend = "Seuil de coupure (4 clusters)",
       lty = 2, col = "red", lwd = 2, cex = 0.9)
dev.off()
cat("Figure 10 sauvegardée\n")

cat("\nComparaison CAH vs K-Means :\n")
clusters_cah <- cutree(hclust(dist(X_scaled, method="euclidean"),
                               method="ward.D2"), k=4)
print(table(CAH=clusters_cah, KMeans=km4$cluster))


# -----------------------------------------------------------------------------
# 16. ÉTAPE 5 – ANALYSE COMBINÉE : CONTRIBUTIONS + RÉGIONS
# -----------------------------------------------------------------------------

# Figure 11 – Contributions
png("fig11_contributions.png", width = 1400, height = 700, res = 120)
p_c1 <- fviz_contrib(res_pca, choice="var", axes=1, fill="#2E86AB",
                      color="white", top=9) +
  labs(title="Contributions à PC1") + theme_minimal(base_size=11)
p_c2 <- fviz_contrib(res_pca, choice="var", axes=2, fill="#F18F01",
                      color="white", top=9) +
  labs(title="Contributions à PC2") + theme_minimal(base_size=11)
grid.arrange(p_c1, p_c2, ncol=2,
             top="Figure 11 – Contributions des Variables aux Axes Factoriels")
dev.off()
cat("Figure 11 sauvegardée\n")

# Figure 12 – Composition régionale
cross_tab <- df %>%
  group_by(Cluster, Region) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Cluster) %>%
  mutate(pct = round(100*n/sum(n), 1))

png("fig12_regions_clusters.png", width = 1300, height = 700, res = 120)
p_reg <- ggplot(cross_tab, aes(x=Cluster, y=pct, fill=Region)) +
  geom_bar(stat="identity", color="white") +
  scale_fill_manual(values=couleurs_region) +
  labs(title="Figure 12 – Composition Régionale des Clusters",
       x="Cluster", y="Proportion (%)", fill="Région") +
  theme_minimal(base_size=12)
print(p_reg)
dev.off()
cat("Figure 12 sauvegardée\n")


# -----------------------------------------------------------------------------
# 17. ÉTAPE 6 – HEATMAP DES PROFILS
# -----------------------------------------------------------------------------
profils_long <- profils %>%
  select(-n) %>%
  pivot_longer(-Cluster, names_to="Variable", values_to="Valeur") %>%
  group_by(Variable) %>%
  mutate(Valeur_norm = (Valeur - min(Valeur)) / (max(Valeur) - min(Valeur)))

png("fig13_heatmap_profils.png", width=1200, height=800, res=120)
p_heat <- ggplot(profils_long, aes(x=Cluster, y=Variable, fill=Valeur_norm)) +
  geom_tile(color="white", linewidth=0.5) +
  geom_text(aes(label=round(Valeur, 2)), size=3.2, color="black") +
  scale_fill_gradient2(low="#C73E1D", mid="white", high="#2E86AB",
                       midpoint=0.5, name="Valeur\nnormalisée") +
  labs(title="Figure 13 – Heatmap des Profils de Clusters", x="Cluster", y="") +
  theme_minimal(base_size=12)
print(p_heat)
dev.off()
cat("Figure 13 sauvegardée\n")


# -----------------------------------------------------------------------------
# 18. RÉSUMÉ FINAL
# -----------------------------------------------------------------------------
cat("\n=============================================================\n")
cat("  RÉSUMÉ DE L'ANALYSE MULTIVARIÉE\n")
cat("=============================================================\n")
cat(sprintf("  Dataset          : %d pays x %d variables\n", nrow(df), length(num_vars)))
cat(sprintf("  PC1 variance     : %.2f%%\n", res_pca$eig[1,2]))
cat(sprintf("  PC1+PC2 variance : %.2f%%\n", res_pca$eig[2,3]))
cat(sprintf("  Clusters K-Means : k = 4\n"))
cat(sprintf("  Score silhouette : %.4f\n", mean(sil[,3])))
cat(sprintf("  Inertie expliquée: %.2f%%\n", 100*km4$betweenss/km4$totss))
cat("=============================================================\n")
cat("  Toutes les figures sont sauvegardées (.png)\n")
cat("  Script R termine avec succes !\n")
cat("=============================================================\n")
