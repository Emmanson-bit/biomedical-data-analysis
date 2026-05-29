# install.packages(c("tidyverse", "ggpubr", "corrplot", "pROC", "caret"))
library(tidyverse)
library(ggpubr)
library(corrplot)
library(pROC)
library(caret)
#Load Data
data <- read.csv("breast_cancer_data.csv")
head(data)
str(data)
names(data)
#Clean and prepare diagnosis variable
data$diagnosis <- as.factor(data$diagnosis)
levels(data$diagnosis)
# Convert to numeric for some analyses
data$diagnosis_num <- ifelse(data$diagnosis == "M", 1, 0)
table(data$diagnosis, data$diagnosis_num)
#Select only important variables
selected_data <- data %>%
  select(
    diagnosis, diagnosis_num,
    fractal_dimension_mean,
    fractal_dimension_worst,
    symmetry_mean,
    symmetry_worst,
    compactness_mean,
    compactness_worst,
    concavity_mean,
    concave.points_mean,
    fractal_dimension_se,
    symmetry_se,
    compactness_se,
    radius_mean,
    area_mean,
    perimeter_mean
  )
# Check missing values
colSums(is.na(selected_data))
# Descriptive statistics by diagnosis group
selected_data %>%
  group_by(diagnosis) %>%
  summarise(across(where(is.numeric), list(mean = mean, sd = sd), .names = "{.col}_{.fn}"))
# Compare malignant vs benign for key variables
ggboxplot(selected_data, x = "diagnosis", y = "fractal_dimension_mean",
          color = "diagnosis", palette = "jco",
          add = "jitter")

ggboxplot(selected_data, x = "diagnosis", y = "symmetry_mean",
          color = "diagnosis", palette = "jco",
          add = "jitter")

ggboxplot(selected_data, x = "diagnosis", y = "concavity_mean",
          color = "diagnosis", palette = "jco",
          add = "jitter")

ggboxplot(selected_data, x = "diagnosis", y = "compactness_mean",
          color = "diagnosis", palette = "jco",
          add = "jitter")
# Run Test
t.test(fractal_dimension_mean ~ diagnosis, data = selected_data)
t.test(fractal_dimension_worst ~ diagnosis, data = selected_data)
t.test(symmetry_mean ~ diagnosis, data = selected_data)
t.test(symmetry_worst ~ diagnosis, data = selected_data)
t.test(compactness_mean ~ diagnosis, data = selected_data)
t.test(compactness_worst ~ diagnosis, data = selected_data)
t.test(concavity_mean ~ diagnosis, data = selected_data)
t.test(concave.points_mean ~ diagnosis, data = selected_data)
t.test(fractal_dimension_se ~ diagnosis, data = selected_data)
t.test(symmetry_se ~ diagnosis, data = selected_data)
t.test(compactness_se ~ diagnosis, data = selected_data)

wilcox.test(fractal_dimension_mean ~ diagnosis, data = selected_data)

#Compae Indices
# Standardize variables
selected_data <- selected_data %>%
  mutate(
    z_fractal_mean = as.numeric(scale(fractal_dimension_mean)),
    z_symmetry_mean = as.numeric(scale(symmetry_mean)),
    z_compactness_mean = as.numeric(scale(compactness_mean)),
    z_concavity_mean = as.numeric(scale(concavity_mean)),
    z_concave_points_mean = as.numeric(scale(concave.points_mean))
  )
# Create Neuro-Mimicry Index
selected_data <- selected_data %>%
  mutate(
    Neuro_Mimicry_Index =
      z_fractal_mean +
      z_symmetry_mean +
      z_compactness_mean +
      z_concavity_mean +
      z_concave_points_mean
  )
# Standardize instability variables
selected_data <- selected_data %>%
  mutate(
    z_fractal_se = as.numeric(scale(fractal_dimension_se)),
    z_symmetry_se = as.numeric(scale(symmetry_se)),
    z_compactness_se = as.numeric(scale(compactness_se))
  )
# Create Neuro-Instability Index
selected_data <- selected_data %>%
  mutate(
    Neuro_Instability_Index =
      z_fractal_se +
      z_symmetry_se +
      z_compactness_se
  )
names(selected_data)

#Compare indices
selected_data %>%
  group_by(diagnosis) %>%
  summarise(
    mean_neuro_mimicry = mean(Neuro_Mimicry_Index),
    sd_neuro_mimicry = sd(Neuro_Mimicry_Index),
    mean_neuro_instability = mean(Neuro_Instability_Index),
    sd_neuro_instability = sd(Neuro_Instability_Index)
  )
#Test indices
t.test(Neuro_Mimicry_Index ~ diagnosis, data = selected_data)
t.test(Neuro_Instability_Index ~ diagnosis, data = selected_data)
# Logistic Regression Modeling
# Model 1: Indices only

model1 <- glm(diagnosis_num ~ Neuro_Mimicry_Index + Neuro_Instability_Index,
              data = selected_data,
              family = binomial)

summary(model1)
# Adjusted Model with Tumor Size
# Model 2: Adjusted for tumor size variables

model2 <- glm(diagnosis_num ~ Neuro_Mimicry_Index + Neuro_Instability_Index +
                radius_mean + area_mean + perimeter_mean,
              data = selected_data,
              family = binomial)

summary(model2)
# Odds Ratios
# Compute Odds Ratios and Confidence Intervals

exp(cbind(OR = coef(model2), confint(model2)))

# Model Performance (ROC Analysis)
# Generate predicted probabilities

selected_data$pred_prob <- predict(model2, type = "response")

# ROC curve

roc_obj <- roc(selected_data$diagnosis_num, selected_data$pred_prob)

plot(roc_obj, print.auc = TRUE)

auc(roc_obj)
#Model Evaluation
# Classification performance

selected_data$pred_class <- ifelse(selected_data$pred_prob > 0.5, 1, 0)

confusionMatrix(as.factor(selected_data$pred_class),
                as.factor(selected_data$diagnosis_num))


# Aggressive (Worst-State) Morphometric Features
# Table for worst features
worst_table <- selected_data %>%
  group_by(diagnosis) %>%
  summarise(
    fractal_worst_mean = mean(fractal_dimension_worst),
    fractal_worst_sd = sd(fractal_dimension_worst),
    
    symmetry_worst_mean = mean(symmetry_worst),
    symmetry_worst_sd = sd(symmetry_worst),
    
    compactness_worst_mean = mean(compactness_worst),
    compactness_worst_sd = sd(compactness_worst)
  )

worst_table
View(worst_table)
worst_table$compactness_worst_mean
worst_table$compactness_worst_sd
# Add p value
t.test(fractal_dimension_worst ~ diagnosis, data = selected_data)
t.test(symmetry_worst ~ diagnosis, data = selected_data)
t.test(compactness_worst ~ diagnosis, data = selected_data)

# Reshape data for plotting
library(tidyr)

worst_long <- selected_data %>%
  select(diagnosis,
         fractal_dimension_worst,
         symmetry_worst,
         compactness_worst) %>%
  pivot_longer(cols = -diagnosis,
               names_to = "Feature",
               values_to = "Value")

# Plot
ggboxplot(worst_long,
          x = "diagnosis",
          y = "Value",
          color = "diagnosis",
          add = "jitter") +
  facet_wrap(~Feature, scales = "free") +
  labs(title = "Worst-State Morphometric Features by Diagnosis")
# Variability table
variability_table <- selected_data %>%
  group_by(diagnosis) %>%
  summarise(
    fractal_se_mean = mean(fractal_dimension_se),
    fractal_se_sd = sd(fractal_dimension_se),
    
    symmetry_se_mean = mean(symmetry_se),
    symmetry_se_sd = sd(symmetry_se),
    
    compactness_se_mean = mean(compactness_se),
    compactness_se_sd = sd(compactness_se)
  )

print(variability_table, width = Inf)
t.test(fractal_dimension_se ~ diagnosis, data = selected_data)
t.test(symmetry_se ~ diagnosis, data = selected_data)
t.test(compactness_se ~ diagnosis, data = selected_data)

#Plot
library(tidyr)

variability_long <- selected_data %>%
  select(diagnosis,
         fractal_dimension_se,
         symmetry_se,
         compactness_se) %>%
  pivot_longer(cols = -diagnosis,
               names_to = "Feature",
               values_to = "Value")

ggboxplot(variability_long,
          x = "diagnosis",
          y = "Value",
          color = "diagnosis",
          add = "jitter") +
  facet_wrap(~Feature, scales = "free") +
  labs(title = "Morphometric Variability Features by Diagnosis")

library(ggplot2)

# Neuro-Mimicry
ggplot(selected_data, aes(x = diagnosis, y = Neuro_Mimicry_Index, fill = diagnosis)) +
  geom_violin(alpha = 0.6, trim = FALSE) +
  geom_boxplot(width = 0.15, fill = "white") +
  scale_fill_manual(values = c("B" = "#4CAF50", "M" = "#E53935")) +
  labs(title = "Neuro-Mimicry Index by Tumor Type",
       x = "Diagnosis",
       y = "Neuro-Mimicry Index") +
  theme_minimal()

ggplot(selected_data, aes(x = diagnosis, y = Neuro_Instability_Index, fill = diagnosis)) +
  geom_violin(alpha = 0.6, trim = FALSE) +
  geom_boxplot(width = 0.15, fill = "white") +
  scale_fill_manual(values = c("B" = "#1E88E5", "M" = "#FB8C00")) +
  labs(title = "Neuro-Instability Index by Tumor Type",
       x = "Diagnosis",
       y = "Neuro-Instability Index") +
  theme_minimal()

selected_data %>%
  group_by(diagnosis) %>%
  summarise(
    mean_mimic = mean(Neuro_Mimicry_Index),
    sd_mimic = sd(Neuro_Mimicry_Index)
  ) %>%
  ggplot(aes(x = diagnosis, y = mean_mimic, fill = diagnosis)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_errorbar(aes(ymin = mean_mimic - sd_mimic,
                    ymax = mean_mimic + sd_mimic),
                width = 0.2) +
  scale_fill_manual(values = c("B" = "#6A1B9A", "M" = "#D81B60")) +
  theme_minimal() +
  labs(title = "Mean Neuro-Mimicry Index with Variability")

#Combine plots
library(ggpubr)

p1 <- ggboxplot(selected_data, x="diagnosis", y="fractal_dimension_mean", color="diagnosis")
p2 <- ggboxplot(selected_data, x="diagnosis", y="symmetry_mean", color="diagnosis")
p3 <- ggboxplot(selected_data, x="diagnosis", y="compactness_mean", color="diagnosis")
p4 <- ggboxplot(selected_data, x="diagnosis", y="concavity_mean", color="diagnosis")

ggarrange(p1, p2, p3, p4,
          labels = c("A", "B", "C", "D"),
          ncol = 2, nrow = 2)

library(ggpubr)
library(tidyr)

# Create worst-state plot
worst_plot <- selected_data %>%
  select(diagnosis,
         fractal_dimension_worst,
         symmetry_worst,
         compactness_worst) %>%
  pivot_longer(cols = -diagnosis,
               names_to = "Feature",
               values_to = "Value") %>%
  ggboxplot(x = "diagnosis",
            y = "Value",
            color = "diagnosis",
            add = "jitter") +
  facet_wrap(~Feature, scales = "free") +
  labs(title = "Worst-State Morphometric Features") +
  theme_minimal()
# Create variability plot
variability_plot <- selected_data %>%
  select(diagnosis,
         fractal_dimension_se,
         symmetry_se,
         compactness_se) %>%
  pivot_longer(cols = -diagnosis,
               names_to = "Feature",
               values_to = "Value") %>%
  ggboxplot(x = "diagnosis",
            y = "Value",
            color = "diagnosis",
            add = "jitter") +
  facet_wrap(~Feature, scales = "free") +
  labs(title = "Morphometric Variability Features") +
  theme_minimal()
#Combine
ggarrange(
  worst_plot,
  variability_plot,
  labels = c("A", "B"),
  ncol = 1, nrow = 2
)


library(ggpubr)

combined_plot <- ggarrange(
  p1, p2, p3, p4,
  labels = c("A", "B", "C", "D"),
  ncol = 2, nrow = 2
)

annotate_figure(
  combined_plot,
  top = text_grob("Baseline Morphometric Features by Tumor Type",
                  face = "bold", size = 14)
)
#Increase spacing
annotate_figure(
  combined_plot,
  top = text_grob("Baseline Morphometric Features by Tumor Type",
                  face = "bold", size = 16),
  bottom = text_grob("B = Benign, M = Malignant", size = 10)
)

#Neuromimicry
mimicry_plot <- ggplot(selected_data,
                       aes(x = diagnosis,
                           y = Neuro_Mimicry_Index,
                           fill = diagnosis)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_boxplot(width = 0.15, fill = "white") +
  scale_fill_manual(values = c("B" = "#2E7D32", "M" = "#C62828")) +
  theme_minimal() +
  labs(title = "Neuro-Mimicry Index",
       x = "Diagnosis",
       y = "Index Value")

mimicry_plot <- ggplot(selected_data,
                       aes(x = diagnosis,
                           y = Neuro_Mimicry_Index,
                           fill = diagnosis)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_boxplot(width = 0.15, fill = "white") +
  scale_fill_manual(values = c("B" = "#2E7D32", "M" = "#C62828")) +
  theme_minimal() +
  labs(title = "Neuro-Mimicry Index",
       x = "Diagnosis",
       y = "Index Value")

instability_plot <- ggplot(selected_data,
                           aes(x = diagnosis,
                               y = Neuro_Instability_Index,
                               fill = diagnosis)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_boxplot(width = 0.15, fill = "white") +
  scale_fill_manual(values = c("B" = "#1565C0", "M" = "#EF6C00")) +
  theme_minimal() +
  labs(title = "Neuro-Instability Index",
       x = "Diagnosis",
       y = "Index Value")

ggarrange(
  mimicry_plot,
  instability_plot,
  labels = c("A", "B"),
  ncol = 2
)

# Create table for composite indices for Neuromimicry
indices_table <- selected_data %>%
  group_by(diagnosis) %>%
  summarise(
    mean_mimicry = mean(Neuro_Mimicry_Index),
    sd_mimicry = sd(Neuro_Mimicry_Index),
    
    mean_instability = mean(Neuro_Instability_Index),
    sd_instability = sd(Neuro_Instability_Index)
  )

print(indices_table, width = Inf)
# Neuro-Mimicry pvalue
t_mimic <- t.test(Neuro_Mimicry_Index ~ diagnosis, data = selected_data)

# Neuro-Instability
t_instab <- t.test(Neuro_Instability_Index ~ diagnosis, data = selected_data)

t_mimic
t_instab

# Create formatted table
final_indices_table <- data.frame(
  Index = c("Neuro-Mimicry Index", "Neuro-Instability Index"),
  
  Benign = c(
    paste0(round(indices_table$mean_mimicry[indices_table$diagnosis=="B"], 2),
           " ± ",
           round(indices_table$sd_mimicry[indices_table$diagnosis=="B"], 2)),
    
    paste0(round(indices_table$mean_instability[indices_table$diagnosis=="B"], 2),
           " ± ",
           round(indices_table$sd_instability[indices_table$diagnosis=="B"], 2))
  ),
  
  Malignant = c(
    paste0(round(indices_table$mean_mimicry[indices_table$diagnosis=="M"], 2),
           " ± ",
           round(indices_table$sd_mimicry[indices_table$diagnosis=="M"], 2)),
    
    paste0(round(indices_table$mean_instability[indices_table$diagnosis=="M"], 2),
           " ± ",
           round(indices_table$sd_instability[indices_table$diagnosis=="M"], 2))
  ),
  
  p_value = c(
    formatC(t_mimic$p.value, format = "e", digits = 2),
    formatC(t_instab$p.value, format = "e", digits = 2)
  )
)

final_indices_table
# Function for stars
get_stars <- function(p) {
  if (p < 0.001) return("***")
  else if (p < 0.01) return("**")
  else if (p < 0.05) return("*")
  else return("ns")
}

final_indices_table$Significance <- c(
  get_stars(t_mimic$p.value),
  get_stars(t_instab$p.value)
)

final_indices_table


#Model Performance and Classification Accuracy
# Model 1: Composite indices only
model1 <- glm(
  diagnosis_num ~ Neuro_Mimicry_Index + Neuro_Instability_Index,
  data = selected_data,
  family = binomial
)

summary(model1)

# Model 2: Adjusted model with size-related variables
model2 <- glm(
  diagnosis_num ~ Neuro_Mimicry_Index + Neuro_Instability_Index +
    radius_mean + area_mean + perimeter_mean,
  data = selected_data,
  family = binomial
)

summary(model2)
# Create Regression Results Table
# Extract odds ratios and 95% CI from adjusted model
or_table <- exp(cbind(OR = coef(model2), confint(model2)))

# Convert to data frame
or_table <- as.data.frame(or_table)
or_table$Variable <- rownames(or_table)
rownames(or_table) <- NULL

# Extract p-values
pvals <- summary(model2)$coefficients[, 4]

# Add p-values
or_table$p_value <- pvals

# Reorder columns
or_table <- or_table[, c("Variable", "OR", "2.5 %", "97.5 %", "p_value")]

# Round values
or_table$OR <- round(or_table$OR, 3)
or_table$`2.5 %` <- round(or_table$`2.5 %`, 3)
or_table$`97.5 %` <- round(or_table$`97.5 %`, 3)
or_table$p_value <- signif(or_table$p_value, 3)

# Add significance stars
get_stars <- function(p) {
  if (p < 0.001) return("***")
  else if (p < 0.01) return("**")
  else if (p < 0.05) return("*")
  else return("ns")
}

or_table$Significance <- sapply(or_table$p_value, get_stars)

# View final table
or_table

# create a cleaner display table
regression_table <- data.frame(
  Variable = c("Intercept",
               "Neuro-Mimicry Index",
               "Neuro-Instability Index",
               "Radius (mean)",
               "Area (mean)",
               "Perimeter (mean)"),
  
  OR_95CI = c(
    paste0(or_table$OR[or_table$Variable == "(Intercept)"], " (",
           or_table$`2.5 %`[or_table$Variable == "(Intercept)"], "–",
           or_table$`97.5 %`[or_table$Variable == "(Intercept)"], ")"),
    
    paste0(or_table$OR[or_table$Variable == "Neuro_Mimicry_Index"], " (",
           or_table$`2.5 %`[or_table$Variable == "Neuro_Mimicry_Index"], "–",
           or_table$`97.5 %`[or_table$Variable == "Neuro_Mimicry_Index"], ")"),
    
    paste0(or_table$OR[or_table$Variable == "Neuro_Instability_Index"], " (",
           or_table$`2.5 %`[or_table$Variable == "Neuro_Instability_Index"], "–",
           or_table$`97.5 %`[or_table$Variable == "Neuro_Instability_Index"], ")"),
    
    paste0(or_table$OR[or_table$Variable == "radius_mean"], " (",
           or_table$`2.5 %`[or_table$Variable == "radius_mean"], "–",
           or_table$`97.5 %`[or_table$Variable == "radius_mean"], ")"),
    
    paste0(or_table$OR[or_table$Variable == "area_mean"], " (",
           or_table$`2.5 %`[or_table$Variable == "area_mean"], "–",
           or_table$`97.5 %`[or_table$Variable == "area_mean"], ")"),
    
    paste0(or_table$OR[or_table$Variable == "perimeter_mean"], " (",
           or_table$`2.5 %`[or_table$Variable == "perimeter_mean"], "–",
           or_table$`97.5 %`[or_table$Variable == "perimeter_mean"], ")")
  ),
  
  p_value = c(
    signif(summary(model2)$coefficients["(Intercept)", 4], 3),
    signif(summary(model2)$coefficients["Neuro_Mimicry_Index", 4], 3),
    signif(summary(model2)$coefficients["Neuro_Instability_Index", 4], 3),
    signif(summary(model2)$coefficients["radius_mean", 4], 3),
    signif(summary(model2)$coefficients["area_mean", 4], 3),
    signif(summary(model2)$coefficients["perimeter_mean", 4], 3)
  )
)

regression_table

# Create model fit summary table
model_fit_table <- data.frame(
  Model = c("Model 1: Indices only", "Model 2: Adjusted model"),
  Null_Deviance = c(model1$null.deviance, model2$null.deviance),
  Residual_Deviance = c(model1$deviance, model2$deviance),
  AIC = c(AIC(model1), AIC(model2))
)

# Round for presentation
model_fit_table$Null_Deviance <- round(model_fit_table$Null_Deviance, 2)
model_fit_table$Residual_Deviance <- round(model_fit_table$Residual_Deviance, 2)
model_fit_table$AIC <- round(model_fit_table$AIC, 2)

model_fit_table

# Predicted probabilities from adjusted model
selected_data$pred_prob <- predict(model2, type = "response")

head(selected_data$pred_prob)

library(pROC)

# ROC analysis
roc_obj <- roc(selected_data$diagnosis_num, selected_data$pred_prob)

# Plot ROC curve
roc_plot <- ggroc(roc_obj, size = 1.2, colour = "#C62828") +
  theme_minimal() +
  labs(
    title = "ROC Curve for Predictive Modeling of Malignancy",
    x = "1 - Specificity",
    y = "Sensitivity"
  ) +
  annotate("text", x = 0.65, y = 0.15,
           label = paste("AUC =", round(auc(roc_obj), 3)),
           size = 5)

roc_plot

# Predicted class using 0.5 cutoff
selected_data$pred_class <- ifelse(selected_data$pred_prob > 0.5, 1, 0)

# Confusion matrix
cm <- confusionMatrix(
  as.factor(selected_data$pred_class),
  as.factor(selected_data$diagnosis_num)
)

cm

# Extract performance metrics
performance_table <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Positive Predictive Value",
             "Negative Predictive Value", "Balanced Accuracy", "Kappa", "AUC"),
  Value = c(
    cm$overall["Accuracy"],
    cm$byClass["Sensitivity"],
    cm$byClass["Specificity"],
    cm$byClass["Pos Pred Value"],
    cm$byClass["Neg Pred Value"],
    cm$byClass["Balanced Accuracy"],
    cm$overall["Kappa"],
    as.numeric(auc(roc_obj))
  )
)

performance_table$Value <- round(as.numeric(performance_table$Value), 3)

performance_table


# Build confusion matrix data frame for plotting
cm_table <- as.data.frame(cm$table)

# Plot heatmap
confusion_plot <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "#E3F2FD", high = "#1565C0") +
  theme_minimal() +
  labs(
    x = "Actual Diagnosis",
    y = "Predicted Diagnosis"
  )

confusion_plot

# ===============================
# CORRELATION HEATMAP
# ===============================

# Select only numeric variables (exclude diagnosis)
numeric_data <- selected_data %>%
  select(
    fractal_dimension_mean,
    symmetry_mean,
    compactness_mean,
    concavity_mean,
    concave.points_mean,
    fractal_dimension_se,
    symmetry_se,
    compactness_se,
    radius_mean,
    area_mean,
    perimeter_mean
  )

# Compute correlation matrix
cor_matrix <- cor(numeric_data)

# Plot heatmap
corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.col = "black",
  tl.cex = 0.8,
  col = colorRampPalette(c("#1E88E5", "white", "#D32F2F"))(100),
  addCoef.col = "black",
  number.cex = 0.6
)


# install.packages("GGally")

library(GGally)
install.packages("ggplot2")
library(ggplot2)
library(GGally)
library(dplyr)

# Select key variables
scatter_data <- selected_data %>%
  select(
    diagnosis,
    compactness_mean,
    concavity_mean,
    concave.points_mean,
    symmetry_mean,
    fractal_dimension_mean
  )

# Plot
ggpairs(
  scatter_data,
  aes(color = diagnosis, alpha = 0.6),
  upper = list(continuous = wrap("cor", size = 4)),
  lower = list(continuous = wrap("points", size = 1.5)),
  diag = list(continuous = wrap("densityDiag"))
)

# install.packages("qgraph")
library(qgraph)

qgraph(
  cor_matrix,
  layout = "spring",
  labels = colnames(cor_matrix),
  color = "skyblue",
  edge.color = "black",
  minimum = 0.3  # show only strong correlations
)


# install.packages("pheatmap")
library(pheatmap)

pheatmap(
  cor_matrix,
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  clustering_method = "complete",
  color = colorRampPalette(c("blue", "white", "red"))(100)
)
