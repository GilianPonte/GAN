rm(list=ls())
library(keras)
library(ggplot2)
library(dplyr)

# Randomness seed
set.seed(123)

# Settings training loop
iterations <- 50000
batch_size <- 256
save_dir <- "gan_churn"
dir.create(save_dir)

# Model related settings
latent_dim = 18
x_dim = 1
y_dim = 1
z_dim = 18

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
beta_1 = .5
beta_2 = .9
lr = 1e-4

# Read data
churn <- read.csv2("churn.csv", stringsAsFactors = F)
churn$AreaCode <- NULL
churn$Phone <- NULL

# Make all data numeric
churn <- mutate_all(churn, .funs = as.numeric)

# Normalize data
for(i in 1:dim(churn)[-1]){
  churn[,i] <- 2*(churn[,i] - min(churn[,i]))/(max(churn[,i]) - min(churn[,i]))-1
  print(i)
}

# Convert to a matrix
churn <- as.matrix(churn)
churn <- array_reshape(churn, c(3333,1,dim(churn)[2]))

# Generator
generator_input <- layer_input(shape = c(latent_dim))
generator_output <- generator_input %>% 
  
  # Hidden layers
  layer_dense(units = 1*18) %>%
  layer_activation_leaky_relu(alpha = .2) %>% 
  layer_reshape(target_shape = c(1, 18)) %>% 
  # Some convolutional magic
  layer_conv_1d(filters = 18, kernel_size = 5, 
                padding = "same") %>% 
  layer_activation_leaky_relu(alpha = .2) %>% 
  layer_batch_normalization(momentum = .8) %>%
  # More magic
  layer_conv_1d(filters = 18, kernel_size = 5, 
                padding = "same") %>% 
  layer_activation_leaky_relu(alpha = .2) %>% 
  layer_batch_normalization(momentum = .8) %>%
  # Output layer aka activation fuction
  layer_conv_1d(filters = z_dim, kernel_size = 7,
                activation = "tanh", padding = "same")
generator <- keras_model(generator_input, generator_output)

# Optimizer
generator_optimizer <- optimizer_adam( 
  lr = lr, 
  clipvalue = 1.0,
  decay = 1e-8,
  beta_1 = .5,
  beta_2 = .9
)

# Compile model
generator %>% compile(
  optimizer = generator_optimizer,
  loss = "binary_crossentropy"
)

# Architecture
summary(generator)

# Discriminator -----------------------------------------------------------
discriminator_input <- layer_input(shape = c(x_dim,z_dim))
discriminator_output <- discriminator_input %>% 
  layer_conv_1d(filters = 512, kernel_size = 1, strides = 9) %>% 
  layer_activation_leaky_relu(alpha = .2) %>% 
  layer_dropout(rate = 0.4) %>%
  layer_conv_1d(filters = 18, kernel_size = 1, strides = 9) %>% 
  layer_activation_leaky_relu(alpha = .2) %>%
  layer_conv_1d(filters = 18, kernel_size = 1, strides = 9) %>% 
  layer_activation_leaky_relu(alpha = .2) %>%
  layer_flatten() %>%
  # One dropout layer - important trick!
  layer_dropout(rate = 0.5) %>%  
  # Classification layer
  layer_dense(units = 1, activation = "sigmoid")

discriminator <- keras_model(discriminator_input, discriminator_output)

# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
discriminator_optimizer <- optimizer_adam( 
  lr = lr, 
  clipvalue = 1.0,
  decay = 1e-8,
  beta_1 = .5,
  beta_2 = .9
)

# Compile model
discriminator %>% compile(
  optimizer = discriminator_optimizer,
  loss = "binary_crossentropy"
)

summary(discriminator)

# Set discriminator weights to non-trainable, when training the generator (will only apply to the `gan` model)
freeze_weights(discriminator)

# Input layer of the GAN
gan_input <- layer_input(batch_shape = c(batch_size,latent_dim))

# Combine the discriminator and the generator
gan_output <- discriminator(generator(gan_input))

# Combine the input and the combined D and G
gan <- keras_model(gan_input, gan_output)

gan_optimizer <- optimizer_adam(
  lr = lr, 
  clipvalue = 1.0, 
  decay = 1e-8,
  beta_1 = .5,
  beta_2 = .9
)
gan %>% compile(
  optimizer = gan_optimizer, 
  loss = "binary_crossentropy"
)

summary(gan)

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
  real_data <- as.matrix(churn[start:stop,,])
  rows <- nrow(real_data)
  combined_data <- array(0, dim = c(rows * 2, dim(churn)[-1]))
  combined_data[1:rows,,] <- generated_data
  combined_data[(rows+1):(rows*2),,] <- real_data
  
  # Assembles labels discriminating real from fake images
  labels <- rbind(matrix(1, nrow = batch_size, ncol = 1),
                  matrix(0, nrow = batch_size, ncol = 1))
  
  # One sided label smoothing!
  labels <- labels + (0.5 * array(runif(prod(dim(labels))),
                                  dim = dim(labels)))
  # Trains the discriminator
  d_loss <- discriminator %>% train_on_batch(x = combined_data, y = labels)
  
  # Store the loss
  losses[step,1] <- d_loss
  
  # Samples random points in the latent space
  random_latent_vectors <- matrix(rnorm(batch_size * latent_dim), 
                                  nrow = batch_size, ncol = latent_dim)
  
  # Assembles labels that say "all real images"
  misleading_targets <- array(0, dim = c(batch_size, 1))
  
  # Trains the generator (via the gan model, where the discriminator weights are frozen)
  a_loss <- gan %>% train_on_batch( 
    random_latent_vectors, 
    misleading_targets
  )
  
  # Store the adversarial loss
  losses[step,2] <- a_loss
  
  start <- start + batch_size
  if (start > (nrow(churn) - batch_size))
    start <- 1
  
  # Occasionally saves data
  if (step %% 1000 == 0) { 
    
    # Saves model weights
    save_model_weights_hdf5(gan, "gan_churn.h5")
    
    # Prints metrics
    cat("discriminator loss:", d_loss, "\n")
    cat("adversarial loss:", a_loss, "\n")
    
    # Convert fake data to dataframe
    generated_data <- as.data.frame(generated_data)
    
    # Write out the data before de-normalization
    write.csv(generated_data, file = paste0("gan_churn/generated_churn_data_before_normalization", step,".csv"), row.names = F)
    
    ## Read old data for de-normalization metrics
    churn2 <- read.csv2("churn.csv", stringsAsFactors = F)
    
    ## Delete not numeric data
    churn2$AreaCode <- NULL
    churn2$Phone <- NULL
    
    ## Give the right column names
    colnames(generated_data) <- colnames(churn2)
    
    ## Make all data numeric
    churn2 <- mutate_all(churn2, .funs = as.numeric)
    
    ## De-normalize data again by column
    for(i in 1:latent_dim){
      generated_data[,i] <- (((generated_data[,i] + 1 )/2)*(max(churn2[,i]) - min(churn2[,i])) + min(churn2[,i]))
    }
    
    # Saves generated data
    write.csv(generated_data, file = paste0("gan_churn/generated_churn_data", step,".csv"), row.names = F)
    }
}

## Did our model converge?
plot <- ggplot(losses, aes(x = 1:iterations, y = d_loss)) + geom_smooth()
plot + geom_smooth(aes(x = 1:iterations, y = a_loss, color = "GAN loss"))
