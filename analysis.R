rm(list = ls())
library(dplyr)
library(ggplot2)

generated <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization40000.csv", stringsAsFactors = F)
churn2 <- read.csv2("churn.csv", stringsAsFactors = F)

## delete not numeric data
churn2$AreaCode <- NULL
churn2$Phone <- NULL

## give the right column names
colnames(generated) <- colnames(churn2)

## make all data numeric
churn2 <- mutate_all(churn2, .funs = as.numeric)
generated <- mutate_all(generated, .funs = as.numeric)

## normalize data
for(i in 1:dim(churn2)[-1]){
  churn2[,i] <- 2*(churn2[,i] - min(churn2[,i]))/(max(churn2[,i]) - min(churn2[,i]))-1
  print(i)
}

churn2$rf <- "real data"
generated$rf <- "fake data"

test <- rbind(churn2, generated)
ggplot(test, aes(AccountLength, fill = rf)) + geom_density(alpha = .9)


## de-normalization
churn2 <- read.csv2("churn.csv", stringsAsFactors = F)
generated <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization40000.csv", stringsAsFactors = F)

summary(generated)

## delete not numeric data
churn2$AreaCode <- NULL
churn2$Phone <- NULL

## make all data numeric
churn2 <- mutate_all(churn2, .funs = as.numeric)
generated <- mutate_all(generated, .funs = as.numeric)

## give the right column names
colnames(generated) <- colnames(churn2)

## de-normalize
#for(i in 1:13){
#  generated[,i] <- ((generated[,i] + min(churn2[,i]))*(max(churn2[,i]) - min(churn2[,i]))/2)+1
#}

## de-normalize
for(i in 1:18){
  generated[,i] <- (((generated[,i] + 1 )/2)*(max(churn2[,i]) - min(churn2[,i])) + min(churn2[,i]))
}


churn2$rf <- "real data"
generated$rf <- "fake data"

test <- rbind(churn2, generated)
ggplot(test, aes(Churn, fill = rf)) + geom_density(alpha = .9)

summary(generated)

t.test(churn2$AccountLength,generated$AccountLength)
