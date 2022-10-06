library(tidyverse)
library(janitor)
library(R.matlab)


data <- readMat("./dataassign3.mat")


### Estimating state transition parameters first - we need to calculate a transition
### matrix, which should be constant through time, so our counts are not conditional
### on time, only on the past state

State_long <- data$State %>%
  as.data.frame() %>%
  mutate(
    market = row_number()
  ) %>%
  relocate(market) %>%
  pivot_longer(V1:V5,
    names_to = "time",
    names_prefix = "V",
    values_to = "state_current"
  ) %>%
  mutate(
    time = as.numeric(time)
  )

State1 <- State_long %>%
  group_by(market) %>%
  arrange(time) %>%
  mutate(state_lag = lag(state_current, 1)) %>%
  na.omit() %>%
  group_by(state_current, state_lag) %>%
  summarise(count = n()) %>%
  group_by(state_lag) %>%
  mutate(transition_prob = count / sum(count))

transition_state <- matrix(
  c(State1$transition_prob),
  nrow = 2
)

write.csv(transition_state, file = "transitionState_allObserved.csv", row.names = FALSE)



### Estimating CCPs - there are functions of every possible combination of
### PState, State, and lagged-firm choice


### Putting in long-format
Firm1 <- data$Firm1 %>% 
  as.data.frame() %>%
  mutate(
    market = row_number()
  ) %>%
  relocate(market) %>%
  pivot_longer(V1:V5,
               names_to = "time",
               names_prefix = "V",
               values_to = "choice_current"
  ) %>%
  mutate(
    time = as.numeric(time)
  ) %>%
  group_by(market) %>%
  arrange(time) %>%
  mutate(choice_lag = lag(choice_current, 1)) %>%
  #firms are all out of the market at t=0
  mutate(choice_lag = ifelse(time == 1, 0, choice_lag))


PState <- data$PState %>% 
  as.data.frame() %>%
  mutate(
    market = row_number()
  ) %>%
  rename(
    Pstate = V1
  )

CCP_calculation <- Firm1 %>%
  left_join(State_long) %>%
  left_join(PState) %>%
  group_by(choice_current, state_current, Pstate, choice_lag) %>%
  summarise(
    count = n()
  ) %>%
  group_by(state_current, Pstate, choice_lag) %>%
  mutate(
    CCP = count/sum(count)
  ) %>%
  select(-count)

write.csv(CCP_calculation, file = "ccp_allObserved.csv", row.names = FALSE)


