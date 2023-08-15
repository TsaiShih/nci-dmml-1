install.packages(c("leaps","caret")) 
library(leaps) # best subset regsubsets() function
library(dplyr) # mutate() function can modify the column
library(car) # vif() function
library(glmnet) #glmnet() function ridge regression
library(class)# knn()function
library(caret)# confusionMatrix() function
library(tree) # decision tree() function
library(e1071) # naiveBayes() function

#get the filepath
getwd()
#set filepath
setwd("D:/dmml-1-project")

#---------beer_1 data------------
#read csv file
beer_1 = read.csv("BeerProject.csv",header=T,na.strings=c(""), stringsAsFactors = T)

#dim() check dimensions of data row = 528870 , column = 13
dim(beer_1)
#head() show the top rows of data
head(beer_1)
#show the columns
beer_1[0,]
#check structure of data
str(beer_1)
# drop columns
beer_lm<-beer_1[,-c(2:4,10,12,13)]
#names() rename the column names
names(beer_lm)<- c("abv","style","appearance","palette","overall","taste","aroma")
#sapply() check the na or missing value
sapply(beer_lm, function(x) sum(is.na(x)))
#na.omit() remove na value
beer_lm<-na.omit(beer_lm) 
sapply(beer_lm, function(x) sum(is.na(x)))
dim(beer_lm)
str(beer_lm)

par(mfrow = c(1, 2))
hist(beer_lm$overall)
boxplot(beer_lm$overall)

str(beer_lm)
n<- nrow(beer_lm)
set.seed(21)
sub_beer<- sample(seq_len(n), size = round(0.7 * n))
train.beer<-beer_lm[sub_beer,]
test.beer<-beer_lm[-sub_beer,]

#find the best subset
factor(beer_lm$style) # 104 levels
regfit.full<-regsubsets(overall~.-style,data=train.beer) # drop style column
par(mfrow = c(1, 1))
plot(regfit.full,scale="Cp")
summary(regfit.full)
coef(regfit.full,5)

#multiple linear regression 
lm.fit<-lm(overall~.-style,data=train.beer) # drop style column
summary(lm.fit)
#Multiple R-squared:  0.6591  p-value: < 2.2e-16

par(mfrow = c(2, 2)) # create 2 x 2 charts area
plot(lm.fit) ####Diagnostic Plots for Linear Regression

future <- predict(lm.fit,test.beer)
future <- as.data.frame(future)
result <- cbind(future,test.beer)
final <- mutate(result,mape=abs(future-overall)/overall)
mean(final$mape)
# Mean Absolute Percentage Error mape =  9%


#---------melb data------------
melb = read.csv("Melbourne_housing_FULL.csv",header=T,na.strings=c(""), stringsAsFactors = T)
dim(melb) # rows 34857 columns 21
head(melb)
sapply(melb,function(x) sum(is.na(x)))
melb<-na.omit(melb) #remove na value
str(melb)
#--------------linear
melb[0,]
#choose the columns we are interesting 
melb<-melb[,c(3:5,11:16)]
str(melb)

melb_lm<-melb[,-c(2)]
str(melb_lm)

par(mfrow = c(2, 2))
boxplot(melb_lm$Price)
hist(melb_lm$Price)
boxplot(log(melb_lm$Price))
hist(log(melb_lm$Price))

melb_lm["log_price"]<-log(melb_lm$Price)
plot(melb_lm[,-c(2)])
cor(melb_lm[,-c(2)]) # check correlation

set.seed(21)
n<-nrow(melb_lm)
sub_melb<- sample(seq_len(n), size = round(0.7 * n))
train<-melb_lm[sub_melb,]
test<-melb_lm[-sub_melb,]

#--------------linear regression

lm.fit<-lm(log_price~.-Price,data=train)
summary(lm.fit)

lm.fit<-lm(log_price~.-Price-Bedroom2-Landsize,data=train) # drop Price, Bedroom2 and Landsize
summary(lm.fit)
vif(lm.fit)  #Variance inflation factor, check multicollinearity
str(test)
pre.lm.fit<-predict(lm.fit,test)
mse<-mean((test$log_price-pre.lm.fit)^2)
mse 
#Mean squared error is 0.1415734

#--------------prepare data for ridge regression
str(melb_lm) 
melb_ridge<-melb_lm[,-c(2,3,6)] # copy same dataframe from melb_lm drop Price, Bedroom2, Landsize
str(melb_ridge)
n<- nrow(melb_ridge)
set.seed(21)
sub_melb<- sample(seq_len(n), size = round(0.7 * n))
train<-melb_ridge[sub_melb,]
test<-melb_ridge[-sub_melb,]
x <- model.matrix(log_price~., train)
x.test <- model.matrix(log_price~., test)
y <- train$log_price
y.test <- test$log_price

#--------------Ridge regression

ridge.melb=glmnet(x,y,alpha=0) # alpha 1 is lasso, 0 is ridge
par(mfrow = c(1, 1))
plot(ridge.melb,xvar="lambda",label = TRUE)
ridge.melb$lambda # lambda 0.029
cv.ridge<-cv.glmnet(x,y,alpha=0) # cross validation
cv.ridge$lambda.min # cross-validation lambda min 0.029
plot(cv.ridge)
coef(cv.ridge)
names(cv.ridge)
summary(cv.ridge)

cv.ridge.predict<-predict(cv.ridge,newx = x.test, s=cv.ridge$lambda.min)
cv.ridge.predict
cv.ridge.mse<-mean((y.test-cv.ridge.predict)^2)
cv.ridge.mse
# mse is 0.1414863

#---------------prepare knn 
str(melb)

n<- nrow(melb)
set.seed(21)
sub_melb<- sample(seq_len(n), size = round(0.7 * n))
std.melb <- scale(melb[, -2]) # standardize 0-1
train_type<-melb$Type[sub_melb]
train.std<-std.melb[sub_melb,]
train<-melb[sub_melb,]
test.std<-std.melb[-sub_melb,]
test<-melb[-sub_melb,]

#---------------knn

knn.pred<-knn(train.std,test.std,train_type,k=1)
confusionMatrix(knn.pred, test$Type)
cM<-confusionMatrix(knn.pred, test$Type)
plot(cM$table)
#accuracy is 86%  Kappa : 65.25%

find_k<-function(i,j){
  for(i in j){
    knn.pred<-knn(train.std,test.std,train_type,k=i)
    confusionMatrix(knn.pred, test$Type)
    cM<-confusionMatrix(knn.pred, test$Type)
    print(paste("K=",i,"accuracy is", as.numeric(cM$overall[1])))
  }
}
find_k(1,1:10) # find the best k
knn.pred<-knn(train.std,test.std,train_type,k=9)
confusionMatrix(knn.pred, test$Type)
cM<-confusionMatrix(knn.pred, test$Type)
plot(cM$table)
#accuracy is 88.22% Kappa : 68.82%

#---------estate data------------

estate = read.csv("real_estate.csv",header=T,na.strings=c(""), stringsAsFactors = T)

dim(estate) # row = 5477006 , column = 13
head(estate)
sapply(estate, function(x) sum(is.na(x)))

#------tidy data
str(estate)
summary(estate)
cor(estate[,c(1,6:13)])
estate<-estate[,-c(2:5)]
summary(estate)
estate<-estate[estate$price>0,] #filter data without minus price
summary(estate$price)

#------------decision tree
median(estate$price) # 2,990,000

estate["High"] <- factor(ifelse(estate$price < 2990000, "Low", "High")) # create binary variable
str(estate)

n <- nrow(estate)
set.seed (21)
sub_estate<- sample(seq_len(n), size = round(0.7 * n))
train <- estate[sub_estate,]
test<- estate[-sub_estate,]

tree.estate <- tree(High~.-price, data = train) #drop price because we had High column which the type show level of price
plot(tree.estate)
text(tree.estate , pretty = 0)
tree.estate

tree.pred <- predict(tree.estate ,test,type = "class") # check model
mean(tree.pred != test$High) #misclassification error 21.4%
confusionMatrix(tree.pred, test$High)
cM<-confusionMatrix(tree.pred, test$High)
plot(cM$table)
#  Accuracy  78.65%

#----Use cross-validation to check where to stop pruning before pruning

set.seed(21)
cv.estate<-cv.tree(tree.estate, FUN = prune.misclass)
names(cv.estate)
plot(cv.estate$size,cv.estate$dev, type = "b") ### we can chose the lowest RMSE

prune.estate<-prune.misclass(tree.estate,best=9 ) #9, 10 show the same
plot(prune.estate)
text(prune.estate, pretty=0)

tree.pred <- predict(prune.estate ,test ,type = "class")

mean(tree.pred != test$High) #misclassification error 21.4%


#---------Naive Bayes -----------------
estate_nb<-estate
str(estate_nb)
l.High<-estate_nb$High # label of High
n <- nrow(estate_nb)
set.seed(21)
sub_estate <- sample(seq_len(n), size = round(0.7 * n))
train <- estate_nb[sub_estate,]
train.label<-l.High[sub_estate] # create training label
test <- estate_nb[ - sub_estate,]
str(train)
nb.estate <- naiveBayes(train[,-c(10)], train.label)
nb.estate
pre.estate <- predict(nb.estate,test[,-c(10)]) # it takes a while

mean(pre.estate != test$High) #misclassification error 12%

confusionMatrix(pre.estate, test$High)
cM<-confusionMatrix(pre.estate, test$High)
plot(cM$table)
#accuracy is 87.9%

