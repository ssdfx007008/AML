library(datasets)
library(tibble)
library(munsell)
library(klaR)
library(caret)
library(glmnet)
library(plotmo)

setwd("E:/Documents/Git/AML/hw5/part 1")
raw_data <- read.csv(file="default_features_1059_tracks.txt", header = TRUE)

part_a_lat.lm = lm(Latitude ~ V0+V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+
                     V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32+V33+V34+V35+V36+V37+
                     V38+V39+V40+V41+V42+V43+V44+V45+V46+V47+V48+V49+V50+V51+V52+V53+V54+V55+V56+
                     V57+V58+V59+V60+V61+V62+V63+V64+V65+V66+V67, data = raw_data)
part_a_lat.res = resid(part_a_lat.lm)
plot(raw_data$Latitude, part_a_lat.res, ylab="Residuals", xlab="Latitude", main="A) Residuals Predicting Latitude from features") 
abline(0, 0) 

part_a_lon.lm = lm(Longitude ~ V0+V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+
                     V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32+V33+V34+V35+V36+V37+
                     V38+V39+V40+V41+V42+V43+V44+V45+V46+V47+V48+V49+V50+V51+V52+V53+V54+V55+V56+
                     V57+V58+V59+V60+V61+V62+V63+V64+V65+V66+V67, data = raw_data)
part_a_lon.res = resid(part_a_lat.lm)
plot(raw_data$Longitude, part_a_lon.res, ylab="Residuals", xlab="Longitude", main="A) Residuals Predicting Longitude from features") 
abline(0, 0) 

#-------------------------------------------------

#exponent <- function(a, pow) (abs(a)^pow)*sign(a)
bc <- function(y, lam)((y^lam - 1)/lam)

raw_data$Lat_adjusted = raw_data$Latitude + 90

# part_b_lat.lm = lm(Lat_adjusted ~ V0+V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+
#                                      V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32+V33+V34+V35+V36+V37+
#                                      V38+V39+V40+V41+V42+V43+V44+V45+V46+V47+V48+V49+V50+V51+V52+V53+V54+V55+V56+
#                                      V57+V58+V59+V60+V61+V62+V63+V64+V65+V66+V67, data = raw_data)
#part_b_lat.bc = boxcox(part_a_lat.lm, lambda = seq(-2, 7, 1/50), plotit = TRUE)
part_b_lat.bc = boxcox(Lat_adjusted ~ V0+V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+
                         V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32+V33+V34+V35+V36+V37+
                         V38+V39+V40+V41+V42+V43+V44+V45+V46+V47+V48+V49+V50+V51+V52+V53+V54+V55+V56+
                         V57+V58+V59+V60+V61+V62+V63+V64+V65+V66+V67, data = raw_data, lambda = seq(-5, 5, 1/50), plotit = TRUE)

part_b_lat.bestlam = part_b_lat.bc$x[which.max(part_b_lat.bc$y)]
part_b_lat.newmodel = lm(bc(Lat_adjusted,part_b_lat.bestlam) ~ V0+V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+
                           V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32+V33+V34+V35+V36+V37+
                           V38+V39+V40+V41+V42+V43+V44+V45+V46+V47+V48+V49+V50+V51+V52+V53+V54+V55+V56+
                           V57+V58+V59+V60+V61+V62+V63+V64+V65+V66+V67, data = raw_data)
part_b_lat.res = resid(part_b_lat.newmodel)
plot(raw_data$Lat_adjusted, part_b_lat.res, ylab="Residuals", xlab="Longitude", main="B) Residuals Predicting Latitude\nAfter Box-Cox") 
abline(0, 0) 



raw_data$Lon_adjusted = raw_data$Longitude + 90

part_b_lon.bc = boxcox(Lon_adjusted ~ V0+V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+
                         V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32+V33+V34+V35+V36+V37+
                         V38+V39+V40+V41+V42+V43+V44+V45+V46+V47+V48+V49+V50+V51+V52+V53+V54+V55+V56+
                         V57+V58+V59+V60+V61+V62+V63+V64+V65+V66+V67, data = raw_data, lambda = seq(-5, 5, 1/50), plotit = TRUE)

part_b_lon.bestlam = part_b_lon.bc$x[which.max(part_b_lon.bc$y)]
part_b_lon.newmodel = lm(bc(Lon_adjusted,part_b_lon.bestlam) ~ V0+V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+
                           V19+V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32+V33+V34+V35+V36+V37+
                           V38+V39+V40+V41+V42+V43+V44+V45+V46+V47+V48+V49+V50+V51+V52+V53+V54+V55+V56+
                           V57+V58+V59+V60+V61+V62+V63+V64+V65+V66+V67, data = raw_data)
part_b_lon.res = resid(part_b_lon.newmodel)
plot(raw_data$Lon_adjusted, part_b_lon.res, ylab="Residuals", xlab="Longitude", main="B) Residuals Predicting Longitude\nAfter Box-Cox") 
abline(0, 0) 

summary(part_a_lat.lm)$r.squared
summary(part_b_lat.newmodel)$r.squared

summary(part_a_lon.lm)$r.squared
summary(part_b_lon.newmodel)$r.squared


#-----------------------


part_c_lat.Y = as.matrix( bc(raw_data$Lat_adjusted,part_b_lat.bestlam) )
part_c_lat.X = as.matrix(raw_data[,!names(mtcars) %in% c("Latitude", "Longitude", "Lon_adjusted","Lat_adjusted" )])


part_c_lon.Y = as.matrix(raw_data$Longitude) #lambda was ~ 1, so no point in box cox
part_c_lon.X = as.matrix(raw_data[,!names(mtcars) %in% c("Latitude", "Longitude", "Lon_adjusted","Lat_adjusted" )])

part_c_lat.ridge.model = glmnet(x=part_c_lat.X, y=part_c_lat.Y, alpha = 0, family = "gaussian")
part_c_lat.ridge.cv = cv.glmnet(x=part_c_lat.X, y=part_c_lat.Y, type.measure="mse", alpha=0,family="gaussian")
plot(part_c_lat.ridge.cv)
plot(part_c_lat.Y, predict(part_c_lat.ridge.cv, newx = part_c_lat.X,  s = "lambda.min") - part_c_lat.Y, ylab = "Residuals", xlab = "Latitude", main = "Transformed Latitude, Ridge")
abline(0, 0) 

part_c_lat.lasso.model = glmnet(x=part_c_lat.X, y=part_c_lat.Y, alpha = 1, family = "gaussian")
part_c_lat.lasso.cv = cv.glmnet(x=part_c_lat.X, y=part_c_lat.Y, type.measure="mse", alpha=1,family="gaussian")
plot(part_c_lat.lasso.cv)
plot(part_c_lat.Y, predict(part_c_lat.lasso.cv, newx = part_c_lat.X,  s = "lambda.min") - part_c_lat.Y, ylab = "Residuals", xlab = "Latitude", main = "Transformed Latitude, Lasso")
abline(0, 0) 


#-------

part_c_lon.ridge.model = glmnet(x=part_c_lon.X, y=part_c_lon.Y, alpha = 0, family = "gaussian")
part_c_lon.ridge.cv = cv.glmnet(x=part_c_lon.X, y=part_c_lon.Y, type.measure="mse", alpha=0,family="gaussian")
plot(part_c_lon.ridge.cv)
plot(part_c_lon.Y, predict(part_c_lon.ridge.cv, newx = part_c_lon.X,  s = "lambda.min") - part_c_lon.Y, ylab = "Residuals", xlab = "Longitude", main = "Longitude, Ridge")
abline(0, 0) 

part_c_lon.lasso.model = glmnet(x=part_c_lon.X, y=part_c_lon.Y, alpha = 1, family = "gaussian")
part_c_lon.lasso.cv = cv.glmnet(x=part_c_lon.X, y=part_c_lon.Y, type.measure="mse", alpha=1,family="gaussian")
plot(part_c_lon.lasso.cv)
plot(part_c_lon.Y, predict(part_c_lon.lasso.cv, newx = part_c_lon.X,  s = "lambda.min") - part_c_lon.Y, ylab = "Residuals", xlab = "Longitude", main = "Longitude, Lasso")
abline(0, 0) 

#------------

mselm <- function(lm) (mean(lm$residuals^2))


mselm(part_b_lat.newmodel)
part_c_lat.lasso.cv$cvm[part_c_lat.lasso.cv$lambda == part_c_lat.lasso.cv$lambda.min]
part_c_lat.ridge.cv$cvm[part_c_lat.ridge.cv$lambda == part_c_lat.ridge.cv$lambda.min]

mselm(part_a_lon.lm)
part_c_lon.lasso.cv$cvm[part_c_lon.lasso.cv$lambda == part_c_lon.lasso.cv$lambda.min]
part_c_lon.ridge.cv$cvm[part_c_lon.ridge.cv$lambda == part_c_lon.ridge.cv$lambda.min]
