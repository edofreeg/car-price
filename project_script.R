# Libraries 
library(readxl)
library(dplyr)
library(car)
library(leaps)
library(glmnet)
library(ggplot2)
library(randomForest)

set.seed(16)

europe=c('Audi','BMW','FIAT','Volvo','Land Rover',
         'Mercedes-Benz','Lotus','Volkswagen','Saab')
asia=c('Acura','Genesis','Honda','Hyundai','Infiniti','Lexus',
       'Kia','Mitsubishi','Mazda','Nissan','Toyota','Suzuki','Subaru','Scion')
america=c('Cadillac','Chrysler','Dodge','Chevrolet','Buick',
          'GMC','Ford','HUMMER','Lincoln','Plymouth','Oldsmobile','Pontiac')#Vector in order to define the origin of the manufacturers

# upload the car dataset
#dataset link:
#https://www.kaggle.com/datasets/CooperUnion/cardataset
cars <- read_excel("Cartel1.xlsx")
cars <- cars[-duplicated(cars), ]

# filter the dataset
Cars <- cars %>% 
  filter(`Engine Fuel Type`  %in%  c("premium unleaded (recommended)",
                                     "regular unleaded")) %>%
  filter(`Transmission Type`  %in%  c("AUTOMATED_MANUAL",'MANUAL',
                                     "AUTOMATIC")) %>%
  filter(Year>2000)

# transform variable and costumize words in the dataset
Cars$`Engine Fuel Type`[Cars$`Engine Fuel Type`=='premium unleaded (recommended)'] <- 'premium'
Cars$`Engine Fuel Type`[Cars$`Engine Fuel Type`=='regular unleaded'] <- 'regular'
Cars$`Transmission Type`[Cars$`Transmission Type`=='AUTOMATED_MANUAL'] <- 'MIXED'

Cars$Driven_Wheels[Cars$Driven_Wheels=='front wheel drive'] <- 'front'
Cars$Driven_Wheels[Cars$Driven_Wheels=='all wheel drive'] <- 'all'
Cars$Driven_Wheels[Cars$Driven_Wheels=='four wheel drive'] <- 'all'
Cars$Driven_Wheels[Cars$Driven_Wheels=='rear wheel drive'] <- 'rear'


Cars$fueltype <- as.factor(Cars$`Engine Fuel Type`)
Cars$transmission <- as.factor(Cars$`Transmission Type`)
Cars$wheels <- as.factor(Cars$Driven_Wheels)
Cars$Vsize <- as.factor(Cars$`Vehicle Size`)
Cars$averageKMpL <- (Cars$`highway MPG`+ Cars$`city mpg`)/2*0.425144
Cars$MSRP <- log(Cars$MSRP,10) 
Cars$Year <- Cars$Year-min(Cars$Year) #in this way year 2000 goes in the intercept and 2001 has value=1 and so on 

# create a new variable origin 
for (i in 1:nrow(Cars)){
  if (Cars$Make[i]   %in% america){Cars$origin[i]='America'}
  else if (Cars$Make[i]   %in% europe){Cars$origin[i]='Europe'}
  else {Cars$origin[i]='Asia'}
}
Cars$origin <- factor(Cars$origin)
# Cars$cylinders <- factor(Cars$cylinders)  #only for the exploratory analysis graph

# clean the dataset from nuisances variables
Cars <- Cars[c(3,5,6,9,16,17,18,19,20,21,22)]
Cars <- na.omit(Cars)

# rename variable's names
names(Cars)[names(Cars) == "MSRP"] <- "Price"
names(Cars)[names(Cars) == "Engine HP"] <- "engineHP"
names(Cars)[names(Cars) == "Engine Cylinders"] <- "cylinders"
names(Cars)[names(Cars) == "Number of Doors"] <- "doors"


# Exploratory analysis (with NOT log price)

ggplot(Cars, aes(x=averageKMpL, y=10**Price,color=cylinders)) +
  geom_point(aes(shape=fueltype))+ ylab("Price in $")


ggplot(Cars, aes(x=origin, y=10**Price, fill=origin)) +
  geom_boxplot() +
  theme(legend.position="none")+ ylab("Price in $")

ggplot(Cars, aes(x=transmission, y=10**Price, fill=origin)) +
  geom_boxplot() + ylab("Price in $")

ggplot(Cars, aes(x=enginetype, y=10**Price, fill=origin)) +
  geom_boxplot()+ ylab("Price in $") 

ggplot(Cars, aes(x=Vsize, y=10**Price, fill=origin)) +
  geom_boxplot() + ylab("Price in $")

# split train / test dataset

train <- sample(1:nrow(Cars), nrow(Cars)*0.8)
Cars.train <- Cars[train, ]
Cars.test <- Cars[-train, ]

#############################
#############################
######OLS REGRESSION#########
#############################
#############################
model1 <- lm(Price~.,data=Cars.train)
summary(model1)
# drop the not significative variables: doors and cylinders

# new model 
model2 <- lm(Price~.-doors-cylinders,data=Cars.train)
summary(model2) # all covariates significative
vif(model2)  #no correlation between covariates, vif<10

# check the residuals of the OLS model
qqnorm (residuals (model2), ylab="Residuals")
qqline (residuals (model2)) 
# there is tails problem for the normality assumption of the residuals

# standardize residuals
rsta <- rstandard(model2) 
plot(fitted(model2), rsta,
     xlab="Fitted values", ylab="standardized Residuals",main='standardized residual',
     pch=19, cex=0.8,
     ylim=c(-6,6))
abline(h=0, col=2)
abline(h=2, col=3);abline(h=-2, col=3) #a lot of points are outside the green bands

# distribution of the residuals, quite normal, tail problems
hist(rsta,breaks=30,probability = 1,xlab='standardize residuals',main='distribution of standardized residual')
lines(density(rsta),lwd=2)
lines(seq(-6,8,0.1),dnorm(seq(-6,8,0.1),mean(rsta),sd(rsta)),col='blue',lwd=2)

# R2 and MSE for train data 
R_square.train= summary(model2)$r.squared
MSE.train.ols=mean(model2$residuals^2)

# R2 and MSE for test data 
y_predict.test.ols <- predict(model2,newdata=Cars.test) #y_hat OLS

SSE.test.ols <- sum((y_predict.test.ols - Cars.test$Price)^2)
SST.test.ols <- sum((Cars.test$Price - mean(Cars.test$Price))^2)

R_square.test <- 1 - (SSE.test.ols / SST.test.ols)
MSE.test.ols = SSE.test.ols/length(y_predict.test)

# scatter plot observed vs predict log price
plot(Cars.test$Price,y_predict.test.ols,ylim=c(4,5.5),xlim=c(4,5.5),xlab='y test',ylab='y predict',main='predict vs observed')
abline(coef = c(0,1),col='red',lwd=3)

# resume table 
ols.resume= data.frame('type'=c('ols.train','ols.test'),'MSE'=c(MSE.train.ols,MSE.test.ols),'Rsquare'=c(R_square.train,R_square.test))

#############################
#############################
####RIDGE AND LASSO##########
#############################
#############################
# Build the matrix of covariates both train and test
X_train = model.matrix(model2)[,-1]
X_test = model.matrix(Price~.-doors-cylinders,data=Cars.test)[,-1]
# Build the vector of response both train and test
y_train = Cars.train$Price
y_test = Cars.test$Price

# set the possible values for lambda both for RIDGE and LASSO
lambda.grid = 10^seq(5,-4, length = 1000)

#############################
#############################
#####RIDGE REGRESSION########
#############################
#############################
ridge.mod = glmnet(X_train,y_train,alpha=0, lambda=lambda.grid) # data are standardized by default

#CV for finding the optimal lambda
cv.out.ridge = cv.glmnet(X_train,y_train,alpha=0,lambda=lambda.grid)
plot(lambda.grid,cv.out.ridge$cvm,log='x',type='l',lwd=3,xlab='Lambda',ylab='CV error',main='Optimal Lambda')
abline(v=bestlam.ridge,col='orange',lwd=3)

# Plots the cross-validation curve,and upper and lower standard deviation curves, as a function of the lambda values used
plot(cv.out.ridge)

# set best lambda for Ridge regression
bestlam.ridge = cv.out.ridge$lambda.1se
abline(v=log(bestlam.ridge), lty=1,col='orange',lwd=3)

# After selecting the best lambda, thanks to cross validation,we refit the ridge regression model
coef.ridge <- predict(cv.out.ridge,type="coefficients",s=bestlam.ridge)

# we compare them with the LS solution:
ls.coef = coef(lm(y_train ~ X_train))

# resume coefficients table
ridge.resume=data.frame(cbind(coef.ridge@Dimnames[[1]],as.vector(round(ls.coef,4)),as.vector(round(coef.ridge,4))))

names(ridge.resume)[names(ridge.resume) == "X1"] <- "beta name"
names(ridge.resume)[names(ridge.resume) == "X2"] <- "ls.coef"
names(ridge.resume)[names(ridge.resume) == "X3"] <- "ridge.coef"

ridge.resume

# plot of the coefficients:
matplot(ridge.mod$lambda,t(ridge.mod$beta[-1,]),type='l',log='x',lty=1,ylab='Non standardized coefficients',xlab='lambda',main='Betas coef. value per lambda value in ridge regression',lwd=1.5)
legend(10**1,-0.01,rownames(ridge.mod$beta[-1,]),col=c(1:11),lty=2,lwd=3,cex=.5)
abline(v=bestlam.ridge, lty=1,col='orange',lwd=3)
abline(h=0)
# the coefficients are returned in the original scale so they are not directly comparable

# computing the standardized coefficients: 'engineHP' is, by far the most important, variable
std.coeff = ridge.mod$beta * matrix(apply(X_train,2,sd),nrow=dim(ridge.mod$beta)[1],ncol=length(lambda.grid),byrow=FALSE)
matplot(ridge.mod$lambda,t(std.coeff),type='l',lty=1,lwd=2,col=1:12,log='x',xlab='Lambda',ylab='Standradized coefficients',main='ridge regression')
legend(10**1,0.1,rownames(std.coeff),col=c(1:11),lty=2,lwd=3,cex=.53)
abline(v=bestlam.ridge, lty=1,col='orange',lwd=3)
abline(h=0)

# predict with Ridge regression
# Compute R^2 from true and predicted values
# definig the evaluation metrics function
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  MSE = SSE/nrow(df)
  
  # Model performance metrics
  data.frame(
    MSE = MSE,
    Rsquare = R_square
  )
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge.mod, s = bestlam.ridge, newx = X_train)
ridge.train.result=eval_results(y_train, predictions_train, Cars.train)
plot(y_train,predictions_train,ylim=c(4,5.5),xlim=c(4,5.5),xlab='y train',ylab='y predict',main='predict vs observed')
abline(coef = c(0,1),col='red',lwd=3)

# Prediction and evaluation on test data
predictions_test <- predict(ridge.mod, s = bestlam.ridge, newx = X_test)
ridge.test.result=eval_results(y_test, predictions_test, Cars.test)
plot(y_test,predictions_test,xlab='y test',ylab='y predict',main='predict vs observed',xlim=c(4,5.2),ylim=c(4,5.2))
# plot(10**y_test,10**predictions_test,xlab='y test',ylab='y predict',main='predict vs observed') # re-scaled price
abline(coef = c(0,1),col='red',lwd=3)

# summary of the metrics
ridge.result=data.frame('type'=c('ridge.train','ridge.test'),rbind(ridge.train.result,ridge.test.result))

#############################
#############################
#####LASSO REGRESSION########
#############################
#############################
lasso.mod = glmnet(X_train,y_train,alpha=1, lambda=lambda.grid,standardize = TRUE) 

# CV for finding optimal lambda
cv.out.lasso = cv.glmnet(X_train,y_train,alpha=1,lambda=lambda.grid)
bestlam.lasso = cv.out.lasso$lambda.1se

plot(lambda.grid,cv.out.lasso$cvm,log='x',type='l',lwd=3,xlab='Lambda',ylab='CV error',main='Optimal Lambda')
abline(v=bestlam.lasso, lty=2,col='orange')

# Plots the cross-validation curve,and upper and lower standard deviation curves, as a function of the lambda values used
plot(cv.out.lasso)
abline(v=log(bestlam.lasso), lwd=3,lty=1,col='orange')

# plot of the coefficients:
matplot(lasso.mod$lambda,t(lasso.mod$beta[-1,]),type='l',log='x',lwd=2,lty=1,col=c(1:11),ylab='Non standardized coefficients',xlab='lambda',main='Betas coef. value per lambda value in lasso regression') # the coefficients are returned in the original scale
legend(10^0,-0.01,rownames(lasso.mod$beta[-1,]),col=c(1:11),lty=1,lwd=2,cex=.55)
abline(v=bestlam.lasso, lty=1,col='orange',lwd=3)
abline(h=0)

# we also have to standardize the coefficients to compare them: 'engineHP' is, by far the most important, variable
std.coeff = lasso.mod$beta * matrix(apply(X_train,2,sd),nrow=dim(lasso.mod$beta)[1],ncol=length(lambda.grid),byrow=FALSE)

matplot(lasso.mod$lambda,t(std.coeff),type='l',lty=1,col=c(1:11),lwd=2,log='x',xlab='Lambda',ylab='Standradized coefficients',main='Lasso')
legend(10^0,0.1,rownames(std.coeff),col=c(1:11),lty=1,lwd=2,cex=.55)
abline(v=bestlam.lasso, lty=1,col='orange',lwd=3)

# After selecting the best lambda, thanks to cross validation,we refit the lasso regression model
coef.lasso <- predict(cv.out.lasso,type="coefficients",s=bestlam.lasso)

# we compare them with the LS and ridge results:
lasso.resume=cbind(ridge.resume,as.vector(round(coef.lasso,4)))
names(lasso.resume)[names(lasso.resume) == "as.vector(round(coef.lasso, 4))"] <- "lasso.coef"
lasso.resume # 'transmissionMIXED' and 'averageKMpL' have been shrunk to zero in case of Lasso.
 
# Prediction and evaluation on train data
predictions_train <- predict(lasso.mod, s = bestlam.lasso, newx = X_train)
lasso.train.result= eval_results(y_train, predictions_train, Cars.train)
plot(y_train,predictions_train,xlab='y test',ylab='y predict',main='predict vs observed',xlim=c(4,5.2),ylim=c(4,5.2))
abline(coef = c(0,1),col='red',lwd=3)
# Prediction and evaluation on test data
predictions_test <- predict(lasso.mod, s = bestlam.lasso, newx = X_test)
lasso.test.result=eval_results(y_test, predictions_test, Cars.test)
plot(y_test,predictions_test,xlab='y test',ylab='y predict',main='predict vs observed',xlim=c(4,5.2),ylim=c(4,5.2))
abline(coef = c(0,1),col='red',lwd=3)

#resume table for the metrics
lasso.result=data.frame('type'=c('lasso.train','lasso.test'),rbind(lasso.train.result,lasso.test.result))
lasso.result
#############################
#############################
##RANDOM FOREST REGRESSION###
#############################
#############################
#CV for Random Forest in order to get mtry optimal value
oob.err<-double(8)
test.err<-double(8) #define some parameters for OOB error

for(mtry in 1:8) {
  rf=randomForest(Price ~ . , data = Cars.train[,-c(3,4)] ,mtry=mtry,ntree=400) 
  oob.err[mtry] = rf$mse[length(rf$mse)] #Error of all Trees fitted
  
  pred<-predict(rf,Cars.test[,-c(3,4)]) #Predictions on Test Set for each Tree
  test.err[mtry]= with(Cars.test[,-c(3,4)], mean( (Price - pred)^2)) #Mean Squared Test Error
  
  cat(mtry," ")
  
}

matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))

#Random forest with mtry=4 because is the optimal value
rf.Cars.train <- randomForest(Price ~ ., data = Cars.train[,-c(3,4)], mtry = 4, importance = TRUE)

# Prediction and evaluation on train data
yhat.rf.train <- predict(rf.Cars.train, newdata = Cars.train[,-c(3,4)])
plot(y_train,yhat.rf.train,ylim=c(4,5.2),xlim=c(4,5.2),xlab='y train',ylab='y predicted',main='observed vs predict')
abline(coef=c(0,1),col='red',lwd=3)

SSE.train <- sum((yhat.rf.train - y_train)^2)
SST.train <- sum((y_train - mean(y_train))^2)
R_square.train <- 1 - (SSE.train / SST.train)
MSE.train = SSE.train/length(y_train)

# Prediction and evaluation on test data
yhat.rf.test <- predict(rf.Cars.train, newdata = Cars.test[,-c(3,4)])
plot(y_test,yhat.rf.test,ylim=c(4,5.2),xlim=c(4,5.2),xlab='y test',ylab='y predict',main='predict vs observed')
abline(coef=c(0,1),col='red',lwd=3)

SSE.test <- sum((yhat.rf.test - y_test)^2)
SST.test <- sum((y_test - mean(y_test))^2)
R_square.test <- 1 - (SSE.test / SST.test)
MSE.test = SSE.test/length(y_test)
# variable importance
importance(rf.Cars.train)
varImpPlot(rf.Cars.train,main = 'Variable importance')

# resume table for random forest
rf.resume = data.frame('type'=c('rf.train','rf.test'),'MSE'=c(MSE.train,MSE.test),'Rsquare'=c(R_square.train,R_square.test))
rf.resume

# final table with all regression model metrics
all.model.resume= rbind(ols.resume,ridge.result,lasso.result,rf.resume)
all.model.resume.train=all.model.resume[c(1,3,5,7),]
all.model.resume.test=all.model.resume[c(2,4,6,8),]


