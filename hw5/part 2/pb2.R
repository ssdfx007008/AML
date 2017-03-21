library(glmnet)
library(MASS)
setwd("/media/guangzhe/DL/498AML")

raw_data <- read.csv("creditcard.csv", skip=1, header = TRUE)
x <- as.matrix(raw_data[,seq(2,24)])
y <- as.factor(raw_data[,25])
num_examples <- dim(raw_data)[1]
num_features <- dim(raw_data)[2]
fits <- glm(y~x, family = "binomial")
summary(fits)
fits$deviance
par(mfrow=c(2,2))
plot(fits)

LR <- cv.glmnet(x,y, alpha=0,family='binomial', type.measure="class", lambda = c(1e-12, 1e-11)) 
print(LR$cvm)
par(mfrow=c(1,1))
a <- cv.glmnet(x,y, alpha=0,family='binomial') 
b <- cv.glmnet(x,y, alpha=.3, family='binomial')
c <- cv.glmnet(x,y, alpha=.6, family='binomial')
d <- cv.glmnet(x,y, alpha=1, family='binomial') 
