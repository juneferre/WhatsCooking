library(tidyverse)
library(tidymodels)
library(tidytext)
library(jsonlite)
library(textrecipes)

# Read data
train <- read_file("/Users/juneferre/fall2025/stat348/Kaggle/WhatsCooking/train.json") |> 
  fromJSON()

test <- read_file("/Users/juneferre/fall2025/stat348/Kaggle/WhatsCooking/test.json") |> 
  fromJSON()

# Tidy format: one row per ingredient
# tidy_train <- train %>%
#   unnest(ingredients)

train_text <- train %>%
  mutate(ingredient_text = map_chr(ingredients, ~ paste(.x, collapse = " ")))


recipe_tfidf <- recipe(cuisine ~ ingredient_text, data = train_text) %>%
  step_tokenize(ingredient_text) %>%
  step_tokenfilter(ingredient_text, max_tokens = 3000) %>%
  step_tfidf(ingredient_text) %>%
  step_zv(all_predictors())


# recipe_tfidf <- recipe(cuisine ~ ingredients + id, data = tidy_train) %>%
#   update_role(id, new_role = "id") %>%      # don't use id as a predictor
#   step_tokenize(ingredients) %>%            # turn ingredients into tokens
#   step_tokenfilter(ingredients, max_tokens = 6000) %>%  # keep top 6k ingredients
#   step_tfidf(ingredients) %>%               # compute TF-IDF scores
#   step_zv(all_predictors())                 # remove zero variance columns
# 

log_reg_spec <- multinom_reg(
  penalty = tune(),      # we will tune lambda
  mixture = 1            # LASSO regularization (best for sparse inputs)
) %>%
  set_engine("glmnet")


set.seed(123)
folds <- vfold_cv(train_text, v = 3, strata = cuisine)


log_reg_wf <- workflow() %>%
  add_model(log_reg_spec) %>%
  add_recipe(recipe_tfidf)



grid <- grid_regular(penalty(), levels = 20)

tune_results <- tune_grid(
  log_reg_wf,
  resamples = folds,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)



tune_results %>% collect_metrics()


best_penalty <- tune::select_best(tune_results, metric = "accuracy")



final_wf <- finalize_workflow(log_reg_wf, best_penalty)

final_fit <- final_wf %>%
  fit(data = train_text)





test_text <- test %>%
  mutate(ingredient_text = map_chr(ingredients, ~ paste(.x, collapse = " ")))

test_preds <- predict(final_fit, new_data = test_text, type = "class")


head(test_preds)

submission <- tibble(
  id = test$id,
  cuisine = test_preds$.pred_class
)

write_csv(submission, "multinomial_logregression.csv")





