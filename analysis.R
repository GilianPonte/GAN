rm(list = ls())
library(dplyr)
library(ggplot2)
library(keras)

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
generated1 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization39000.csv", stringsAsFactors = F)
generated2 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization38000.csv", stringsAsFactors = F)
generated3 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization37000.csv", stringsAsFactors = F)
generated4 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization36000.csv", stringsAsFactors = F)
generated5 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization35000.csv", stringsAsFactors = F)
generated6 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization34000.csv", stringsAsFactors = F)
generated7 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization33000.csv", stringsAsFactors = F)
generated8 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization32000.csv", stringsAsFactors = F)
generated9 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization31000.csv", stringsAsFactors = F)
generated10 <- read.csv("C:/Users/Gilia/Dropbox/RUG - MSc Marketing/Learning Community Data Science/GAN/GAN/gan_churn/generated_churn_data_before_normalization30000.csv", stringsAsFactors = F)

generated <- rbind(generated, generated1, generated2, generated3, generated4,generated5,generated6,generated7,generated8, generated9, generated10)
rm(generated1, generated2, generated3, generated4,generated5,generated6,generated7,generated8, generated9, generated10)
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
ggplot(test, aes(AccountLength, fill = rf)) + geom_density(alpha = .9)

summary(generated)

# models
generated$rf <- NULL

# data manipulated
generated$Churn[generated$Churn < .5] <- 0
generated$Churn[generated$Churn >= .5] <- 1

generated$IntlPlan[generated$IntlPlan < .5] <- 0
generated$IntlPlan[generated$IntlPlan >= .5] <- 1

generated$VMailPlan[generated$VMailPlan < .5] <- 0
generated$VMailPlan[generated$VMailPlan >= .5] <- 1

generated$VMailMessage[generated$VMailMessage < 0.01] <- 0
generated <- round(generated, digits = 2)

## logit
independent <- paste(colnames(generated)[1:(ncol(generated)-2)], collapse = ' + ')
BaseFormula <- as.formula(paste0("Churn ~ ", independent))

Logit_fake <- glm(BaseFormula, data = generated, family = "binomial")
Logit_real <- glm(BaseFormula, data = churn2, family = "binomial")

summary(Logit_fake)
summary(Logit_real)

library(stargazer)
logitoutput <- stargazer(Logit_fake, Logit_real, column.labels = c("fake", "real"), type = "html", summary = T, title = "Fake vs Real", out = "logisticregressionfakereal.html", column.sep.width = "1pt", dep.var.labels = "Churn 0/1")


## 75% of the sample size
smp_size <- floor(0.75 * nrow(generated))

## FAKE - train and test sample
train_ind <- sample(seq_len(nrow(generated)), size = smp_size)
generated_train <- generated[train_ind, ]
generated_test <- generated[-train_ind, ]

## REAL - train and test sample
train_ind <- sample(seq_len(nrow(churn2)), size = smp_size)
churn_train <- churn2[train_ind, ]
churn_test <- churn2[-train_ind, ]

## combined generated and train data
churn_train$rf <- NULL
combined <- rbind(generated, churn_train)

## logit
logit <- NULL
independent <- paste(colnames(generated)[1:(ncol(generated)-2)], collapse = ' + ')
BaseFormula <- as.formula(paste0("Churn ~ ", independent))

Logit_fake <- glm(BaseFormula, data = generated, family = "binomial")
Logit_real <- glm(BaseFormula, data = churn_train, family = "binomial")
Logit_combined <- glm(BaseFormula, data = combined, family = "binomial")

summary(Logit_fake)
summary(Logit_real)
summary(Logit_combined)

# fake data model
logit$predictionlogit <- predict(Logit_fake, newdata= churn_test, type = "response")

# real data model
logit$predictionlogit <- predict(Logit_real, newdata= churn_test, type = "response")

# combined data model
logit$predictionlogit <- predict(Logit_combined, newdata= churn_test, type = "response")

logit$predictionlogitclass[logit$predictionlogit>.5] <- 1
logit$predictionlogitclass[logit$predictionlogit<=.5] <- 0

logit$correctlogit <- logit$predictionlogitclass == churn_test$Churn
hitrate <- mean(logit$correctlogit) *100
print(paste("% of predicted classifications correct", mean(logit$correctlogit)))
LogitOutput <- makeLiftPlot(logit$predictionlogit, churn_test$Churn, "Logit")
LogitOutput + ggtitle("Logistic regression")


makeLiftPlot <- function(Prediction, Evaluate, ModelName){
  iPredictionsSorted <- sort(Prediction,index.return=T,decreasing=T)[2]$ix #extract the index order according to predicted retention
  CustomersSorted <- Evaluate[iPredictionsSorted] #sort the true behavior of customers according to predictions
  SumChurnReal<- sum(Evaluate == 1) #total number of real churners in the evaluation set
  CustomerCumulative=seq(length(Evaluate))/length(Evaluate) #cumulative fraction of customers
  ChurnCumulative=apply(matrix(CustomersSorted == 1),2,cumsum)/SumChurnReal #cumulative fraction of churners
  ProbTD = sum(CustomersSorted[1:floor(length(Evaluate)*.1)]== 1)/floor(length(Evaluate)*.1) #probability of churn in 1st decile
  ProbOverall = SumChurnReal / length(Evaluate) #overall churn probability
  TDL = ProbTD / ProbOverall
  GINI = sum((ChurnCumulative-CustomerCumulative)/(t(matrix(1,1,length(Evaluate))-CustomerCumulative)),na.rm=T)/length(Evaluate)
  Lift <- data.frame(CustomerCumulative, ChurnCumulative)
  ggplot(data = Lift, aes(x=CustomerCumulative, y=ChurnCumulative)) + geom_line() + geom_abline(slope = 1, col = "blue", linetype = 2) + annotate("rect", xmin = 0, xmax = .1, ymin = 0, ymax = 1, alpha = .2) + annotate("text", x = .5, y = 0.00, label = paste("TDL = ",round(TDL,2), "; GINI = ", round(GINI,2), "; Hit rate = ", round(hitrate, 2))) + xlab("Cumulative fraction of customers (sorted by predicted churn probability)") + ylab("Cumulative fraction of churners")
  #return(data.frame(TDL,GINI))
}


