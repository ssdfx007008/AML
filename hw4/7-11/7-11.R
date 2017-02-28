library(datasets)
library(tibble)
library(munsell)
library(klaR)
library(caret)
library(glmnet)

setwd("E:/Documents/Git/AML/hw4/7-11")
abalone <- read.csv(file="abalone.data", header = TRUE)

part_a.lm = lm(Rings ~ Length+Diameter+Height+Whole_weight+Shucked_weight+Viscera_weight+Shell_weight, data = abalone)
part_a.res = resid(part_a.lm)

plot(abalone$Rings, part_a.res, ylab="Residuals", xlab="Rings", main="Age") 
abline(0, 0) 

  # abalone <- lapply(abalone, function(x) {gsub("M", 1, x)}) These mess with the data set type and wreak havoc on lm, so i'm just preprocessing the data now
  # abalone <- lapply(abalone, function(x) {gsub("F", -1, x)})
  # abalone <- lapply(abalone, function(x) {gsub("I", 0, x)})

part_b.lm = lm(Rings ~ Sex+Length+Diameter+Height+Whole_weight+Shucked_weight+Viscera_weight+Shell_weight, data = abalone)
part_b.res = resid(part_b.lm)

plot(abalone$Rings, part_b.res, ylab="Residuals", xlab="Rings", main="Age") 
abline(0, 0) 
