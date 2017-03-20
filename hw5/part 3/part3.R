library(glmnet)

# setup working directory
setwd(getwd())

# read input data
raw_data = read.csv("I2000.txt", header = FALSE, sep = " ")
data = t(as.matrix(raw_data))

raw_tissues = read.csv("tissues.txt", header = FALSE, sep = " ")
tissues = c(raw_tissues < 0) * 1

set.seed(1)
model = cv.glmnet(data, tissues, family = "binomial", type.measure = "deviance", alpha = 1, nfolds = 5)

#evaluate performance of model (deviance)
plot(model)
minlambda = model$lambda.min
deviance = model$cvm[which(model$lambda == minlambda)]
predictors = model$nzero[which(model$lambda == minlambda)]

set.seed(1)
modelauc = cv.glmnet(data, tissues, family = "binomial", type.measure = "auc", alpha = 1, nfolds = 5)
minauclambda = modelauc$lambda.min
#get auc of model with the lowest deviance
auc = modelauc$cvm[which(model$lambda == minlambda)]

minlambda
deviance
auc
predictors