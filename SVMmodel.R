# ============================================================================
# Libraries
# ============================================================================
library(tidyverse)
library(tidymodels)
library(jsonlite)
library(textrecipes)
library(kernlab)            # SVM engine
tidymodels_prefer()

# ============================================================================
# Load Data
# ============================================================================
train <- fromJSON(read_file("/Users/juneferre/fall2025/stat348/Kaggle/WhatsCooking/train.json"))
test  <- fromJSON(read_file("/Users/juneferre/fall2025/stat348/Kaggle/WhatsCooking/test.json"))

# Combine ingredients into a single string
train$ingredient_text <- map_chr(train$ingredients, ~ paste(.x, collapse = " "))
test$ingredient_text  <- map_chr(test$ingredients,  ~ paste(.x, collapse = " "))

# ============================================================================
# Recipe: Tokenize + Filter + TF-IDF
# ============================================================================
# Increase max_tokens to 4000–6000 for better accuracy
recipe_tfidf <- recipe(cuisine ~ ingredient_text, data = train) %>%
  step_tokenize(ingredient_text) %>%
  step_tokenfilter(ingredient_text, max_tokens = 6000) %>%   # <- Stronger than 3000
  step_tfidf(ingredient_text) %>%
  step_zv(all_predictors())

# ============================================================================
# Linear SVM Model
# ============================================================================
svm_spec <- svm_linear(
  cost = tune()           # hyperparameter C
) %>%
  set_engine("LiblineaR") %>%
  set_mode("classification")

# ============================================================================
# Workflow
# ============================================================================
svm_wf <- workflow() %>%
  add_recipe(recipe_tfidf) %>%
  add_model(svm_spec)

# ============================================================================
# Cross-validation folds
# ============================================================================
set.seed(123)
folds <- vfold_cv(train, v = 3, strata = cuisine)

# Grid of cost values to test: log scale 1e-3 → 1e3
cost_grid <- tibble(cost = 10 ^ seq(-3, 3, length.out = 12))

# ============================================================================
# Tune the SVM
# ============================================================================
svm_results <- tune_grid(
  svm_wf,
  resamples = folds,
  grid = cost_grid,
  metrics = metric_set(accuracy),
  control = control_grid(save_pred = TRUE)
)

# View accuracy estimates
svm_results %>% collect_metrics()

# Select best-performing cost value
best_svm <- select_best(svm_results, metric = "accuracy")

# ============================================================================
# Final SVM Model
# ============================================================================
final_svm <- finalize_workflow(svm_wf, best_svm)

# Fit final model on full training set
final_fit <- fit(final_svm, data = train)

# ============================================================================
# Predict on Test Set
# ============================================================================
test_preds <- predict(final_fit, new_data = test, type = "class")

submission <- tibble(
  id = test$id,
  cuisine = test_preds$.pred_class
)

write_csv(submission, "svm_linear_submission.csv")

print("Saved: svm_linear_submission.csv")
