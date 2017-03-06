library(datasets)
library(tibble)
library(munsell)
library(klaR)
library(caret)
library(glmnet)
library(plotmo)

setwd("E:/Documents/Git/AML/hw4/7-11")
abalone <- read.csv(file="abalone.data", header = TRUE)

part_a.lm = lm(Rings ~ Length+Diameter+Height+Whole_weight+Shucked_weight+Viscera_weight+Shell_weight, data = abalone)
part_a.res = resid(part_a.lm)

plot(abalone$Rings, part_a.res, ylab="Residuals", xlab="Rings", main="A) Residuals Predicting Rings w/o Gender") 
abline(0, 0) 

  # abalone <- lapply(abalone, function(x) {gsub("M", 1, x)}) These mess with the data set type and wreak havoc on lm, so i'm just preprocessing the data now
  # abalone <- lapply(abalone, function(x) {gsub("F", -1, x)})
  # abalone <- lapply(abalone, function(x) {gsub("I", 0, x)})

part_b.lm = lm(Rings ~ Sex+Length+Diameter+Height+Whole_weight+Shucked_weight+Viscera_weight+Shell_weight, data = abalone)
part_b.res = resid(part_b.lm)

plot(abalone$Rings, part_b.res, ylab="Residuals", xlab="Rings", main="B) Residuals Predicting Rings w/ Gender") 
abline(0, 0) 


part_c.lm = lm(log(Rings) ~ Length+Diameter+Height+Whole_weight+Shucked_weight+Viscera_weight+Shell_weight, data = abalone)
part_c.res = resid(part_c.lm)

plot(abalone$Rings, part_c.res, ylab="Residuals", xlab="Rings", main="C) Residuals Predicting log(Rings) w/o Gender") 
abline(0, 0) 

part_d.lm = lm(log(Rings) ~ Sex + Length+Diameter+Height+Whole_weight+Shucked_weight+Viscera_weight+Shell_weight, data = abalone)
part_d.res = resid(part_d.lm)

plot(abalone$Rings, part_d.res, ylab="Residuals", xlab="Rings", main="D) Residuals Predicting log(Rings) w/ Gender") 
abline(0, 0) 

trainingSet = createDataPartition(y=abalone$Rings, p=.8, list=FALSE)


#F_a

YTrain = as.matrix(abalone[trainingSet, 9])
XTrain = as.matrix(abalone[trainingSet, -c(1,9), ])#no sex

YTest = as.matrix(abalone[-trainingSet, 9])
XTest = as.matrix(abalone[-trainingSet, -c(1,9), ])

part_f.model = glmnet(x=XTrain, y=YTrain, alpha = 1.0, family = "gaussian")
part_f.cv = cv.glmnet(x=XTrain, y=YTrain, type.measure="mse", alpha=1.0,family="gaussian")
plot(part_f.cv)
plot( YTest, predict(part_f.cv, newx = XTest,  s = "lambda.min") - YTest, ylab = "Residuals", xlab = "Rings", main = "F-a) Regularizer predicting Rings at\n minimum Lambda w/o Sex")
abline(0, 0) 

#F_b
YTrain = as.matrix(abalone[trainingSet, 9])
XTrain = as.matrix(abalone[trainingSet, -c(9), ])#with sex

YTest = as.matrix(abalone[-trainingSet, 9])
XTest = as.matrix(abalone[-trainingSet, -c(9), ])#with sex


part_f.model = glmnet(x=XTrain, y=YTrain, alpha = 1.0, family = "gaussian")
part_f.cv = cv.glmnet(x=XTrain, y=YTrain, type.measure="mse", alpha=1.0,family="gaussian")
plot(part_f.cv)
plot( YTest, predict(part_f.cv, newx = XTest,  s = "lambda.min") - YTest, ylab = "Residuals", xlab = "Rings", main = "F-b) Regularizer predicting Rings at\n minimum Lambda w/ Sex")
abline(0, 0) 
#F_c
YTrain = as.matrix(log(abalone[trainingSet, 9]))
XTrain = as.matrix( abalone[trainingSet, -c(1,9), ] )#no sex

YTest = as.matrix(log(abalone[-trainingSet, 9]))
XTest = as.matrix(abalone[-trainingSet, -c(1,9), ])

part_f.model = glmnet(x=XTrain, y=YTrain, alpha = 1.0, family = "gaussian")
part_f.cv = cv.glmnet(x=XTrain, y=YTrain, type.measure="mse", alpha=1.0,family="gaussian")
plot(part_f.cv)
plot( YTest, predict(part_f.cv, newx = XTest,  s = "lambda.min") - YTest, ylab = "Residuals", xlab = "Rings", main = "F-c) Regularizer predicting log(Rings) at\n minimum Lambda w/o Sex")
abline(0, 0) 

#F_d
YTrain = as.matrix(log(abalone[trainingSet, 9]))
XTrain = as.matrix( abalone[trainingSet, -c(9), ] )#with sex

YTest = as.matrix(log(abalone[-trainingSet, 9]))
XTest = as.matrix(abalone[-trainingSet, -c(9), ])#with sex

part_f.model = glmnet(x=XTrain, y=YTrain, alpha = 1.0, family = "gaussian")
part_f.cv = cv.glmnet(x=XTrain, y=YTrain, type.measure="mse", alpha=1.0,family="gaussian")
plot(part_f.cv)

plot( YTest, predict(part_f.cv, newx = XTest,  s = "lambda.min") - YTest, ylab = "Residuals", xlab = "Rings", main = "F-d) Regularizer predicting log(Rings) at\n minimum Lambda w/ Sex")
abline(0, 0) 

