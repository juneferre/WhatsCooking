library(tidyverse)
library(jsonlite)
library(quanteda)
library(quanteda.textmodels)

# ===============================================================
# Load Data
# ===============================================================
train <- read_file("/Users/juneferre/fall2025/stat348/Kaggle/WhatsCooking/train.json") |> 
  fromJSON()

test <- read_file("/Users/juneferre/fall2025/stat348/Kaggle/WhatsCooking/test.json") |> 
  fromJSON()

train$ingredient_text <- map_chr(train$ingredients, ~ paste(.x, collapse = " "))
test$ingredient_text  <- map_chr(test$ingredients,  ~ paste(.x, collapse = " "))

# ===============================================================
# Build quanteda corpus + dfm
# ===============================================================
train_corpus <- corpus(train, text_field = "ingredient_text")
test_corpus  <- corpus(test,  text_field = "ingredient_text")

# Tokenize + build TF-IDF dfm
train_dfm <- train_corpus |> 
  tokens() |> 
  tokens_remove(stopwords("en")) |> 
  dfm() |> 
  dfm_tfidf()

test_dfm <- test_corpus |> 
  tokens() |> 
  tokens_remove(stopwords("en")) |> 
  dfm() |> 
  dfm_match(featnames(train_dfm)) |>   # align columns
  dfm_tfidf()

# ===============================================================
# Train Naive Bayes
# ===============================================================
nb_model <- textmodel_nb(train_dfm, train$cuisine)

# ===============================================================
# Predict
# ===============================================================
nb_preds <- predict(nb_model, test_dfm)

submission <- tibble(
  id = test$id,
  cuisine = nb_preds
)

write_csv(submission, "quanteda_nb_submission.csv")

print("Saved quanteda_nb_submission.csv")
