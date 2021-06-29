library(tidyverse)
library(lubridate)
library(tsibble)
library(feasts)
library(tsibbledata)
library(torch)


vic_elec_get_year <- function(year, month = NULL) {
        vic_elec %>%
                filter(year(Date) == year, month(Date) == if (is.null(month)) month(Date) else month) %>%
                as_tibble() %>%
                select(Demand, Temperature)
}


elec_train <- vic_elec_get_year(2012) %>% as.matrix()
elec_valid <- vic_elec_get_year(2013) %>% as.matrix()
elec_test <- vic_elec_get_year(2014, 1) %>% as.matrix() # or 2014, 7, alternatively

elec_train %>% dim() # (17568, 2)
elec_valid %>% dim() # (17520, 2)  
elec_test %>% dim() # (1488, 2) : 2014년 1월 30분 간격의 demand 


train_mean <- apply(elec_train, 2, mean)
train_sd <- apply(elec_train, 2, sd)

aa <- (elec_train-train_mean)/train_sd
aa %>% dim()

(elec_train[,1] - mean(elec_train[,1]))/sd(elec_train[,1])

scale(elec_train[, 1])[1, ]

scale(elec_train)[1,]

n <- length(aa[,1]) - n_timesteps

starts <- sort(sample.int(
        n = n,
        size = n * 1 # 1:17232  : sample_frac 만큼 1:17232에서 sampling 
))

length(starts)





n_timesteps <- 7 * 24 * 2 

elec_dataset <- dataset(
        name = "elec_dataset",
        
        # x = elec_train : (17568, 2)
        # n_timesteps : 7x24x2 = 336
        
        initialize = function(x, n_timesteps, sample_frac = 1) {  
                
                self$n_timesteps <- n_timesteps
                self$x <- torch_tensor(scale(x)) 
                
                n <- length(self$x[,1]) - self$n_timesteps # 17568 - 336 = 17232
                
                self$starts <- sort(sample.int(
                        n = n,
                        size = n * sample_frac # 1:17232  : sample_frac 만큼 1:17232에서 sampling 
                ))
                
        },
        
        .getitem = function(i) {
                
                start <- self$starts[i] # 1, 2, 3,...,17231, 17232 
                end <- start + self$n_timesteps - 1 # 336, 337, 338,...,17566, 17567
                
                list(
                        x = self$x[start:end, ], # 1:336, 2:337, 3:338, ..., 17231:17566, 17232:17567  
                        y = self$x[end + 1, ]  # 337, 338, 339, ..., 17567, 17568
                )
                
        },
        
        .length = function() { # 데이터셋의 길이, 즉 총 샘플의 수를 적어주는 부분 
                length(self$starts) # 17232
        }
)

train_ds <- elec_dataset(elec_train, n_timesteps, sample_frac = 0.5)
length(train_ds) # 17232/2 = 8616

train_ds[1]$x %>% head() # -0.4141, -0.5541, -0.8053
train_ds[2]$x %>% head() #          -0.5541, -0.8053, -1.0062           
train_ds[3]$x %>% head() #                   -0.8053, -1.0062, -0.8203
train_ds[1]$y



batch_size <- 32
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
length(train_dl) # 8616/32 = 269.25 => 270

b <- train_dl %>% dataloader_make_iter() %>% dataloader_next()
b$x$size() # (32, 336, 2)  
b$y$size() # (32, 2)  


valid_ds <- elec_dataset(elec_valid, n_timesteps, sample_frac = 0.5)
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)

test_ds <- elec_dataset(elec_test, n_timesteps)
test_dl <- test_ds %>% dataloader(batch_size = 1)


test_dl %>% dataloader_make_iter() %>% dataloader_next() # x: (1, 336, 2), y: (1, 2)




# Model 

model <- nn_module(
        
        initialize = function(type, input_size, hidden_size, num_layers = 1, dropout = 0) {
                
                self$type <- type
                self$num_layers <- num_layers
                
                self$rnn <- if (self$type == "gru") {
                        nn_gru(
                                input_size = input_size,
                                hidden_size = hidden_size,
                                num_layers = num_layers,
                                dropout = dropout,
                                batch_first = TRUE # [seq_len, batch_size, hidden_size] => [batch_size, seq_len, hidden_size]
                        )
                } else {
                        nn_lstm(
                                input_size = input_size,
                                hidden_size = hidden_size,
                                num_layers = num_layers,
                                dropout = dropout,
                                batch_first = TRUE
                        )
                }
                
                self$output <- nn_linear(hidden_size, 2)
                
        },
        
        forward = function(x) {
                # (batch_size, n_timesteps, hidden_size) : (32, 336, 32)
                x <- self$rnn(x)[[1]]
                x <- x[ , dim(x)[2], ] # dim(x)[2] = 마지막 336번째만 선택
                x %>% self$output() 
        }
        
)


device <- torch_device(if (cuda_is_available()) "cuda" else "cpu") # cpu, gpu 선택 

net <- model("gru", 2, 32) # (2, 32) : input_size(변수 두개), hidden_size
net <- net$to(device = device)

b$x$size() # (32, 336, 2)

myrnn <- nn_lstm(2, 32, num_layers = 1, dropout = 0, batch_first = TRUE) # self$rnn(x)에 해당함
bb <- myrnn(b$x)[[1]]
dim(bb)

net$output




optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 3

train_batch <- function(b) {
        
        optimizer$zero_grad()
        output <- net(b$x$to(device = device)) # (32, 2)
        target <- b$y$to(device = device) # (32, 2) 
        
        loss <- nnf_mse_loss(output, target)
        loss$backward()
        optimizer$step()
        
        loss$item()
}



valid_batch <- function(b) {
        
        output <- net(b$x$to(device = device))
        target <- b$y$to(device = device)
        
        loss <- nnf_mse_loss(output, target)
        loss$item()
        
}


for (epoch in 1:num_epochs) {
        
        net$train()
        train_loss <- c()
        
        coro::loop(for (b in train_dl) {
                loss <-train_batch(b)
                train_loss <- c(train_loss, loss)
        })
        
        cat(sprintf("\nEpoch %d, training: loss: %3.5f \n", epoch, mean(train_loss)))
        
        net$eval()
        valid_loss <- c()
        
        coro::loop(for (b in valid_dl) {
                loss <- valid_batch(b)
                valid_loss <- c(valid_loss, loss)
        })
        
        cat(sprintf("\nEpoch %d, validation: loss: %3.5f \n", epoch, mean(valid_loss)))
}

net$eval()

preds <- rep(NA, n_timesteps)

coro::loop(for (b in test_dl) {
        output <- net(b$x$to(device = device))
        preds <- c(preds, output %>% as.numeric())
})

vic_elec_jan_2014 <-  vic_elec %>%
        filter(year(Date) == 2014, month(Date) == 1) %>%
        select(Demand)

preds_ts <- vic_elec_jan_2014 %>%
        add_column(forecast = preds * train_sd + train_mean) %>%
        pivot_longer(-Time) %>%
        update_tsibble(key = name)

preds_ts %>%
        autoplot() +
        scale_colour_manual(values = c("#08c5d1", "#00353f")) +
        theme_minimal()


n_forecast <- 2 * 24 * 7

test_preds <- vector(mode = "list", length = length(test_dl))

i <- 1

coro::loop(for (b in test_dl) {
        
        input <- b$x
        output <- net(input$to(device = device))
        preds <- as.numeric(output)
        
        for(j in 2:n_forecast) {
                input <- torch_cat(list(input[ , 2:length(input), ], output$view(c(1, 1, 1))), dim = 2)
                output <- net(input$to(device = device))
                preds <- c(preds, as.numeric(output))
        }
        
        test_preds[[i]] <- preds
        i <<- i + 1
        
})



