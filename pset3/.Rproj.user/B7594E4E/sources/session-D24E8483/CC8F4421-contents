library(tidyverse)
library(janitor)
library(R.matlab)


data <- readMat('./dataassign22.mat')

X1t <- data[['X1t']] |> 
  data.frame() |>
  mutate(id = row_number())
  
  
X <- X1t |>  
  pivot_longer(X1:X10, names_to = "time", values_to = "X_present") |>
  mutate(time = substring(time, 2) |> as.numeric()) |>
  group_by(id) |>
  arrange(time) |>
  mutate(X_past = lag(X_present, 1))

Y <- data[['Y']] |> data.frame() |>
  mutate(id = row_number()) |>
  pivot_longer(X1:X10, names_to = "time", values_to = "Y_present") |>
  mutate(time = substring(time, 2) |> as.numeric()) |>
  group_by(id) |>
  arrange(time) |>
  mutate(Y_past = lag(Y_present, 1)) |>
  select(-Y_present)



dataset_q2 <- X |>
  left_join(Y) |>
  filter(Y_past != 0 & !is.na(X_past)) |>
  mutate(Y1 = (Y_past==1)+0,
         Y2 = (Y_past==2)+0,
         Y3 = (Y_past==3)+0,
         Y4 = (Y_past==4)+0)

write.csv(dataset_q2, file = "dataset_q2long.csv", row.names = FALSE)
