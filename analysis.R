rm(list = ls())

generated <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization40000.csv")


churn2 <- read.csv2("churn.csv", stringsAsFactors = F)

## delete not numeric data
churn2$AreaCode <- NULL
churn2$Phone <- NULL
churn2$VMailMessage <- NULL
churn2$DayCharge <- NULL
churn2$EveCharge <- NULL
churn2$NightCharge <- NULL
churn2$IntlCharge <- NULL

## give the right column names
colnames(generated) <- colnames(churn2)

## make all data numeric
churn2 <- mutate_all(churn2, .funs = as.numeric)

## normalize data
for(i in 1:dim(churn2)[-1]){
  churn2[,i] <- 2*(churn2[,i] - min(churn2[,i]))/(max(churn2[,i]) - min(churn2[,i]))-1
  print(i)
}

churn2$rf <- "real data"
generated$rf <- "fake data"

test <- rbind(churn2, generated)
ggplot(test, aes(EveCalls, fill = rf)) + geom_density(alpha = .9)
