rm(list=ls())
library(keras)
use_implementation("tensorflow")

setwd("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Thesis/data")

## read data
churn <- read.csv2("churn.csv", stringsAsFactors = F)


## delete not numeric data
churn$AreaCode <- NULL
churn$Phone <- NULL

## make all data numeric
library(dplyr)
churn <- mutate_all(churn, .funs = as.numeric)

## normalize data
for(i in 1:dim(churn)[-1]){
  churn[,i] <- (churn[,i] - mean(churn[,i])) / sd(churn[,i])
  print(i)
}

dim(churn)
churn <- as.matrix(churn)

churn <- array_reshape(churn, c(3333,1,1,18))
hist(churn[,15])

## generator
latent_dim = 18

x_dim = 1
y_dim = 1
z_dim = 18

generator_input <- layer_input(shape = c(latent_dim))
generator_output <- generator_input %>% 
  
  # First, transform the input into a 16x16 128-channels feature map
  layer_dense(units = 1*1*18) %>%
  layer_activation_leaky_relu() %>% 
  layer_reshape(target_shape = c(1, 1, 18)) %>% 
  
  # Then, add a convolution layer
  layer_conv_2d(filters = 18, kernel_size = 5, 
                padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  
  # Few more conv layers
  layer_conv_2d(filters = 18, kernel_size = 5, 
                padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 18, kernel_size = 5, 
                padding = "same") %>% 
  layer_activation_leaky_relu() %>% 
  
  # Produce a 32x32 1-channel feature map
  layer_conv_2d(filters = z_dim, kernel_size = 7,
                activation = "tanh", padding = "same")
generator <- keras_model(generator_input, generator_output)
summary(generator)


# Discriminator -----------------------------------------------------------
discriminator_input <- layer_input(shape = c(x_dim,y_dim,z_dim))
discriminator_output <- discriminator_input %>% 
  layer_conv_2d(filters = 18, kernel_size = 1) %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 18, kernel_size = 1, strides = 1) %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 18, kernel_size = 1, strides = 1) %>% 
  layer_activation_leaky_relu() %>% 
  layer_conv_2d(filters = 18, kernel_size = 1, strides = 1) %>% 
  layer_activation_leaky_relu() %>% 
  layer_flatten() %>%
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

discriminator %>% compile(
  optimizer = discriminator_optimizer,
  loss = "binary_crossentropy"
)

# Set discriminator weights to non-trainable, when training the generator
# (will only apply to the `gan` model)
freeze_weights(discriminator)

gan_input <- layer_input(batch_shape = c(20,18))

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

summary(gan)

iterations <- 10000
batch_size <- 20
save_dir <- "gan_churn"
dir.create(save_dir)

## dataframe to store losses
losses <- data.frame(a_loss = NA, d_loss = rep(NA,iterations))

# Start the training loop
start <- 1
for (step in 1:iterations) {
  
  # Samples random points in the latent space
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim), 
                                  nrow = batch_size, ncol = latent_dim)
  # Decodes them to fake images
  try(generated_data <- generator %>% predict(random_latent_vectors))
  
  # Combines them with real images
  stop <- start + batch_size - 1 
  real_data <- as.matrix(churn[start:stop,,,])
  rows <- nrow(real_data)
  combined_data <- array(0, dim = c(rows * 2, dim(churn)[-1]))
  combined_data[1:rows,,,] <- generated_data
  combined_data[(rows+1):(rows*2),,,] <- real_data
  
  # Assembles labels discriminating real from fake images
  labels <- rbind(matrix(1, nrow = batch_size, ncol = 1),
                  matrix(0, nrow = batch_size, ncol = 1))
  
  # Adds random noise to the labels -- an important trick!
  labels <- labels + (0.5 * array(runif(prod(dim(labels))),
                                  dim = dim(labels)))
  
  # Trains the discriminator
  d_loss <- discriminator %>% train_on_batch(x = combined_data, y = labels)
  losses[step,1] <- d_loss
  

  # Samples random points in the latent space
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim), 
                                  nrow = batch_size, ncol = latent_dim)
  # Assembles labels that say "all real images"
  misleading_targets <- array(0, dim = c(batch_size, 1))
  
  # Trains the generator (via the gan model, where the 
  # discriminator weights are frozen)
  a_loss <- gan %>% train_on_batch( 
    random_latent_vectors, 
    misleading_targets
  )
  losses[step,2] <- a_loss
  
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
    
    ## read old data for denormalization
    churn2 <- read.csv2("churn.csv", stringsAsFactors = F)
    
    ## delete not numeric data
    churn2$AreaCode <- NULL
    churn2$Phone <- NULL
    
    ## give the right column names
    colnames(generated_data) <- colnames(churn2)
    
    ## make all data numeric
    library(dplyr)
    churn2 <- mutate_all(churn2, .funs = as.numeric)
    dim(generated_data)
    
    ## denormalize data again
    for(i in 1:18){
      generated_data[i] <- (generated_data[i] + mean(churn2[,i])) * sd(churn2[,i])
    }
    
    # Saves one generated data
    write.csv(generated_data, file = paste0("generated_churn_data", step,".csv"), row.names = F)
  }
}

## did our model converge?
library(ggplot2)
plot <- ggplot(losses, aes(x = 1:10000, y = a_loss)) + geom_line()
plot + geom_line(aes(x = 1:10000, y = d_loss, color = "d_loss"))

       