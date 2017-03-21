###Problem 2 - logistic regression
library(glmnet)
library(MASS)
#setup
setwd("/media/guangzhe/DL/498AML")

#read in csv file
raw_data <- read.csv("creditcard.csv", skip=1, header = TRUE)
x <- as.matrix(raw_data[,seq(2,24)])
y <- as.factor(raw_data[,25])
num_examples <- dim(raw_data)[1]
num_features <- dim(raw_data)[2]

#Simple unregularized Logistic regression
logfit <- glm(y~x, family = "binomial")
summary(logfit)
logfit$deviance #report deviance
par(mfrow=c(2,2))
plot(logfit)

#comparison logistic regression
simplelr <- cv.glmnet(x,y, alpha=0,family='binomial', type.measure="class", lambda = c(1e-12, 1e-11)) #ridge
print(simplelr$cvm)


# train models
par(mfrow=c(1,1))
res0 <- cv.glmnet(x,y, alpha=0,family='binomial') #ridge
res2 <- cv.glmnet(x,y, alpha=.2, family='binomial')
res4 <- cv.glmnet(x,y, alpha=.4, family='binomial')
res5 <- cv.glmnet(x,y, alpha=.5, family='binomial')
res6 <- cv.glmnet(x,y, alpha=.6, family='binomial')
res8 <- cv.glmnet(x,y, alpha=.8, family='binomial')
res1 <- cv.glmnet(x,y, alpha=1, family='binomial') #lasso
