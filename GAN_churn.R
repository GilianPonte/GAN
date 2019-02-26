rm(list=ls())
library(keras)
use_implementation("tensorflow")

setwd("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Thesis/data")

churn <- read.csv2("churn.csv", stringsAsFactors = F)
churn$AreaCode <- NULL
churn$Phone <- NULL

library(dplyr)
churn <- mutate_all(churn, .funs = as.numeric)

for(i in 1:dim(churn)[-1]){
  churn[,i] <- (churn[,i] - mean(churn[,i])) / sd(churn[,i])
  print(i)
}

dim(churn)
str(churn)

generator_input <- layer_input(batch_shape = c(dim(churn)))

generator_output <- generator_input %>% 
  layer_dense(units = 512) %>%
  layer_activation_leaky_relu() %>%
  layer_dense(units = 14, activation = "tanh")

generator <- keras_model(generator_input, generator_output)
summary(generator)


# Discriminator -----------------------------------------------------------
discriminator_input <- layer_input(batch_shape = c(6666,14))

discriminator_output <- discriminator_input %>% 
  layer_dense(units = 64) %>% 
  layer_activation_leaky_relu() %>%
  # One dropout layer - important trick!
  layer_dropout(rate = 0.4) %>%  
  # Classification layer
  layer_dense(units = 1, activation = "sigmoid")

discriminator <- keras_model(discriminator_input, discriminator_output)

summary(discriminator)


# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
discriminator_optimizer <- optimizer_rmsprop( 
  lr = 0.0008, 
  clipvalue = 1.0,
  decay = 1e-8
)

freeze_weights(discriminator)

discriminator %>% compile(
  optimizer = discriminator_optimizer,
  loss = "binary_crossentropy"
)

# Set discriminator weights to non-trainable, when training the generator
# (will only apply to the `gan` model)

gan_input <- layer_input(batch_shape = dim(churn))

gan_output <- discriminator(generator(gan_input))

gan <- keras_model(gan_input, gan_output)

gan_optimizer <- optimizer_rmsprop(
  lr = 0.0004, 
  clipvalue = 1.0, 
  decay = 1e-8
)
gan %>% compile(
  optimizer = gan_optimizer, 
  loss = "binary_crossentropy"
)


iterations <- 10000
batch_size <- 3333
save_dir <- "gan_churn"
dir.create(save_dir)

# Start the training loop
start <- 1
latent_dim = 14
for (step in 1:iterations) {
  
  # Samples random points in the latent space
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim), 
                                  nrow = batch_size, ncol = latent_dim)
  # Decodes them to fake images
  try(generated_data <- generator %>% predict(random_latent_vectors))
  
  # Combines them with real images
  stop <- start + batch_size - 1 
  real_data <- as.matrix(churn[start:stop,])
  rows <- nrow(real_data)
  combined_data <- array(0, dim = c(rows * 2, dim(churn)[-1]))
  
  combined_data <- rbind(generated_data, real_data)
  # Assembles labels discriminating real from fake images
  labels <- rbind(matrix(1, nrow = batch_size, ncol = 1),
                  matrix(0, nrow = batch_size, ncol = 1))
  
  # Adds random noise to the labels -- an important trick!
  labels <- labels + (0.5 * array(runif(prod(dim(labels))),
                                  dim = dim(labels)))
  
  # Trains the discriminator
  d_loss <- discriminator %>% train_on_batch(x = combined_data, y = labels)

  # Samples random points in the latent space
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim), 
                                  nrow = batch_size, ncol = latent_dim)
  
  dim(random_latent_vectors)
  # Assembles labels that say "all real images"
  misleading_targets <- array(0, dim = c(batch_size, 1))
  
  # Trains the generator (via the gan model, where the 
  # discriminator weights are frozen)
  a_loss <- gan %>% train_on_batch( 
    random_latent_vectors, 
    misleading_targets
  )  
  
  start <- start + batch_size
  if (start > (nrow(churn) - batch_size))
    start <- 1
  
  # Occasionally saves images
  if (step %% 100 == 0) { 
    
    # Saves model weights
    save_model_weights_hdf5(gan, "gan_churn.h5")
    
    # Prints metrics
    cat("discriminator loss:", d_loss, "\n")
    cat("adversarial loss:", a_loss, "\n")
    generated_data <- as.data.frame(generated_data)
    colnames(generated_data) <- colnames(churn)
    
    generated_data$DayMins <- (generated_data$DayMins + mean(churn$DayMins)) * sd(churn$DayMins)
    generated_data$DayCalls <- (generated_data$DayCalls + mean(churn$DayCalls)) * sd(churn$DayCalls)
    generated_data$DayCharge <- (generated_data$DayCharge + mean(churn$DayCharge)) * sd(churn$DayCharge)
    generated_data$EveMins <- (generated_data$EveMins + mean(churn$EveMins)) * sd(churn$EveMins)
    generated_data$EveCalls <- (generated_data$EveCalls + mean(churn$EveCalls)) * sd(churn$EveCalls)
    generated_data$EveCharge <- (generated_data$EveCharge + mean(churn$EveCharge)) * sd(churn$EveCharge)
    generated_data$NightMins <- (generated_data$NightMins + mean(churn$NightMins)) * sd(churn$NightMins)
    generated_data$NightCalls <- (generated_data$NightCalls + mean(churn$NightCalls)) * sd(churn$NightCalls)
    generated_data$NightCharge <- (generated_data$NightCharge + mean(churn$NightCharge)) * sd(churn$NightCharge)
    generated_data$IntlMins <- (generated_data$IntlMins + mean(churn$IntlMins)) * sd(churn$IntlMins)
    generated_data$IntlCalls <- (generated_data$IntlCalls + mean(churn$IntlCalls)) * sd(churn$IntlCalls)
    generated_data$IntlCharge <- (generated_data$IntlCharge + mean(churn$IntlCharge)) * sd(churn$IntlCharge)
    generated_data$CustServCalls <- (generated_data$CustServCalls + mean(churn$CustServCalls)) * sd(churn$CustServCalls)
    generated_data$Churn <- (generated_data$Churn + mean(churn$Churn)) * sd(churn$Churn)
    
    # Saves one generated data
    write.csv(generated_data, file = paste0("generated_churn_data", step,".csv"), row.names = F)
    

  }
}


