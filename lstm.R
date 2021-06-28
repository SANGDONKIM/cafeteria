library(tidyverse)
library(lubridate)
library(tsibble)
library(feasts)
library(tsibbledata)
library(torch)

vic_elec %>% glimpse()

vic_elec_2014 <- vic_elec %>% 
        filter(year(Date) == 2014) %>% 
        select(-c(Date, Holiday)) %>% 
        mutate(Demand = scale(Demand), Temperature = scale(Temperature)) %>% 
        pivot_longer(-Time, names_to = "variable") %>% 
        update_tsibble(key = variable)


vic_elec_2014 %>% filter(month(Time) == 7) %>% 
        autoplot() + 
        scale_colour_manual(values = c("#08c5d1", "#00353f")) +
        theme_minimal()

vic_elec_2014 %>% filter(month(Time) == 1) %>% 
        autoplot() + 
        scale_colour_manual(values = c("#08c5d1", "#00353f")) +
        theme_minimal()



elec_dataset <- dataset(
        name = "elec_dataset",
        
        initialize = function(x, n_timesteps, sample_frac = 1) {
                
                self$n_timesteps <- n_timesteps
                self$x <- torch_tensor((x - train_mean) / train_sd)
                
                n <- length(self$x) - self$n_timesteps 
                
                self$starts <- sort(sample.int(
                        n = n,
                        size = n * sample_frac
                ))
                
        },
        
        .getitem = function(i) {
                
                start <- self$starts[i]
                end <- start + self$n_timesteps - 1
                
                list(
                        x = self$x[start:end],
                        y = self$x[end + 1]
                )
                
        },
        
        .length = function() {
                length(self$starts) 
        }
)


vic_elec_get_year <- function(year, month = NULL) {
        vic_elec %>%
                filter(year(Date) == year, month(Date) == if (is.null(month)) month(Date) else month) %>%
                as_tibble() %>%
                select(Demand)
}

vic_elec %>% tail()

elec_train <- vic_elec_get_year(2012) %>% as.matrix()
elec_valid <- vic_elec_get_year(2013) %>% as.matrix()
elec_test <- vic_elec_get_year(2014, 1) %>% as.matrix() # or 2014, 7, alternatively

elec_train %>% dim()
elec_valid %>% dim()  
elec_test %>% dim() # 1488 : 2014년 1월 30분 간격의 demand 


train_mean <- mean(elec_train)
train_sd <- sd(elec_train)


n_timesteps <- 7 * 24 * 2 # days * hours * half-hours : timestep으로 7일 지정 

train_ds <- elec_dataset(elec_train, n_timesteps, sample_frac = 0.5)
length(train_ds)








