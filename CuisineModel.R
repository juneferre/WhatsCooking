library(tidyverse)
library(jsonlite)
library(tidytext)
library(vroom)

# what's cooking kaggle 

# ---------------------------
# Read data
# ---------------------------
train <- read_file("/Users/juneferre/fall2025/stat348/Kaggle/WhatsCooking/train.json") |>
  fromJSON()

test  <- read_file("/Users/juneferre/fall2025/stat348/Kaggle/WhatsCooking/test.json") |>
  fromJSON()


train %>%
  count(cuisine, sort = TRUE)


# ---------------------------
# EDA
# ---------------------------
top_ingredients_by_cuisine <- tidy_train %>%
  count(cuisine, ingredients, sort = TRUE) %>%
  group_by(cuisine) %>%
  slice_max(n, n = 10)   # top 10 per cuisine

top_ingredients_by_cuisine



# --------------------------------
# feature engineering the data
# --------------------------------
tidy_train <- train %>%
  unnest(ingredients)

# 1) number of ingredients
feature_num_ingredients <- tidy_train %>%
  count(id, cuisine, name = "num_ingredients")

# 2) presence of a few specific ingredients
feature_presence_small <- tidy_train %>%
  mutate(value = 1) %>%
  filter(ingredients %in% c("garlic cloves", "soy sauce", "cumin")) %>%  # pick a few
  distinct(id, cuisine, ingredients, .keep_all = TRUE) %>%              # no dupes
  pivot_wider(
    id_cols    = c(id, cuisine),           # keep one row per recipe
    names_from = ingredients,
    values_from = value,
    values_fill = list(value = 0)          # <-- key fix
  )

# 3) number of "spices"
spices <- c("cumin", "coriander", "paprika", "turmeric", "cinnamon")

feature_spices <- tidy_train %>%
  mutate(is_spice = ingredients %in% spices) %>%
  group_by(id, cuisine) %>%
  summarize(num_spices = sum(is_spice), .groups = "drop")

# combine into small feature table
final_small <- feature_num_ingredients %>%
  left_join(feature_presence_small, by = c("id", "cuisine")) %>%
  left_join(feature_spices,        by = c("id", "cuisine"))


# ---------------------------
# EDA
# ---------------------------

# overall top ingredients
top_ingredients2 <- tidy_train %>%
  count(ingredients, sort = TRUE)

# cuisine and it's top ingredients
top_ingredients <- tidy_train %>%
  count(cuisine, ingredients, sort = TRUE)


top_ingredients_by_cuisine <- tidy_train %>%
  count(cuisine, ingredients, sort = TRUE) %>%
  group_by(cuisine) %>%
  slice_max(n, n = 10)   # top 10 per cuisine


# average # ingredients by cuisine
avg_ingredients_by_cuisine <- feature_num_ingredients %>%
  group_by(cuisine) %>%
  summarize(avg_num_ingredients = mean(num_ingredients)) %>%
  arrange(desc(avg_num_ingredients))

avg_ingredients_by_cuisine


# seeing which ingredients ar pertinent to cuisines
library(tidytext)

tfidf_cuisine <- tidy_train %>%
  count(cuisine, ingredients) %>%
  bind_tf_idf(ingredients, cuisine, n) %>%
  arrange(desc(tf_idf))

top_tfidf_by_cuisine <- tfidf_cuisine %>%
  group_by(cuisine) %>%
  slice_max(tf_idf, n = 10)

top_tfidf_by_cuisine


# using lift to understand how unique an ingredient is to a cuisine

ingredient_lift <- tidy_train %>%
  distinct(id, cuisine, ingredients) %>%
  count(cuisine, ingredients) %>%
  group_by(ingredients) %>%
  mutate(total = sum(n)) %>%
  ungroup() %>%
  mutate(lift = n / total) %>%
  arrange(desc(lift))

ingredient_lift %>%
  group_by(cuisine) %>%
  slice_max(lift, n = 10)

ingredient_pct <- tidy_train %>%
  distinct(id, cuisine, ingredients) %>%
  count(cuisine, ingredients) %>%
  group_by(cuisine) %>%
  mutate(pct = n / sum(n)) %>%
  arrange(desc(pct))

ingredient_pct %>%
  group_by(cuisine) %>%
  slice_max(pct, n = 10)




train %>%
  count(cuisine, sort = TRUE) %>%
  ggplot(aes(x = reorder(cuisine, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Number of Recipes per Cuisine",
       x = "Cuisine",
       y = "Number of Recipes")


