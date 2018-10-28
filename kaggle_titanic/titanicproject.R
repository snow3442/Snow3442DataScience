### To install all packages used in the project, 
### enable the following lines of code
#install.packages("tree")
#install.packages("gbm")
#install.packages("randomForest")
#install.packages("ggplot2")
#install.packages("MASS")
#install.packages("boot")
#
#install.packages("adabag")
#install.packages("ada")
#install.packages("rpart")
#install.packages("plyr")
###loading datafiles
titanic_train<-read.csv(file.choose(),head=T)
titanic_test<-read.csv(file.choose(),head=T)
library(tree)
library(gbm)
library(randomForest)
library(ggplot2)
library(MASS)
library(boot)
library(plyr)
###view data###
dim(titanic_train)
dim(titanic_test)
head(titanic_train,n=5)
head(titanic_test, n=5)
#### cleaning  titanic_train####
attach(titanic_train)
titanic_train$Ticket<-NULL
titanic_train$PassengerId<-NULL
titanic_train$Name<-NULL
titanic_train$Embarked<-NULL
titanic_train$Cabin<-NULL
sum(is.na(titanic_train$Age))
titanic_train<-na.omit(titanic_train)
titanic_train<-within(titanic_train,{
  Survived<-as.factor(Survived)
  Pclass<-as.factor(Pclass)
  Parch<-as.factor(Parch)
})
dim(titanic_train)

#### Data Visualization ####
#survival rate 
survivalRate = mean(Survived==1)
survivalRate
#age distribution
hist(Age)
#Survived by socio class

survBySocioClassPlot<-ggplot(titanic_train,
       aes(x= Pclass, fill= Survived)) + geom_bar()
print(survBySocioClassPlot+ggtitle("Survival Status By Socio Class"))
#survived by sex
survBySexPlot<-ggplot(titanic_train, aes(x=Sex, fill=Survived))+geom_bar()
print(survBySexPlot+ggtitle("Survival Status By Gender"))
#survived by age
ggplot(titanic_train, aes(x= Survived, y= Age)) + geom_boxplot()
#####stacked barchart of survived by socio class and by sex
qplot(Pclass, main="Survival Status by gender and social class", data=titanic_train, geom="bar", fill=Survived) +
  facet_wrap(~Sex)
##### boxplots of age by social class and Survival status
ggplot(titanic_train, aes(x= Pclass, y=Age, colour= Survived)) + geom_boxplot()


#### possible Tetravariate plots ####
# if you want to plot age, survived, sex, Pclass altogether use the following
#ggplot(titanic_train, aes(x= Survived, y= Age, colour= Pclass)) + geom_boxplot() + facet_wrap(~ Sex, nrow= 1) 

#### logistic on the entire dataset ####
lean_data=titanic_train
glm.fit<-glm(data=lean_data,Survived~.,family="binomial")
pred.probs<-predict(glm.fit, newdata = lean_data, type="response")
pred.0.1<-rep(0,714)
pred.0.1[pred.probs>0.5]<-1
mean(pred.0.1==lean_data$Survived)
#7 fold cross validation
set.seed(1)
n_folds=7
folds_i<-sample(rep(1:7,length.out=nrow(titanic_train)))
cv_list<-rep(0,7)
for (k in 1:7) {
  test_i<-which(folds_i==k)
  train_logit<-titanic_train[-test_i,]
  test_logit<-titanic_train[test_i,]
  logit.fit<-glm(Survived~Pclass+Sex+Age+SibSp+Fare, data=train_logit,family=binomial)
  pred.probs<-predict(logit.fit,test_logit,type="response")
  pred.surv<-rep(0,nrow(test_logit))
  pred.surv[pred.probs>0.5]=1
  cv_list[k]=mean(pred.surv==test_logit$Survived)
}
cv_list
mean(cv_list)
#### Linear Discriminant Analysis
set.seed(10)
lean_train<-sample(1: nrow(lean_data), 500)
lean_data.train<-lean_data[lean_train,]
lean_data.test<-lean_data[-lean_train,]
lda.fit=lda(Survived~.,data=titanic_train,subset=lean_train)
lda.pred<-predict(lda.fit,lean_data.test)
lda.class<-lda.pred$class
mean(lda.class==lean_data.test$Survived)
#Linear Discriminant Analysis
set.seed(2)
n_folds=7
folds_i<-sample(rep(1:7,length.out=nrow(titanic_train)))
cv_list<-rep(0,7)
for (k in 1:7) {
  test_i<-which(folds_i==k)
  train_i<-which(folds_i!=k)
  train_lda<-titanic_train[-test_i,]
  test_lda<-titanic_train[test_i,]
  lda.fit=lda(Survived~Pclass+Sex+Age+SibSp+Fare,data=titanic_train,subset=train_i)
  lda.pred<-predict(lda.fit,test_lda)
  lda.class<-lda.pred$class
  cv_list[k]=mean(lda.class==test_lda$Survived)
}
cv_list
mean(cv_list)

library(adabag)
library(ada)
library(rpart)
lean_data<-titanic_train
###train and test using adaboost ####
lean_train<-sample(1: nrow(lean_data), 500)
lean_data.train<-lean_data[lean_train,]
lean_data.test<-lean_data[-lean_train,]
ada.titanic<- ada(Survived~., data=lean_data.train, type = "gentle", iter = 50)
ada.pred<-predict(ada.titanic,newdata=lean_data.test)
ada.pred.mean=mean(ada.pred==lean_data.test$Survived)
ada.pred.mean
##### 7 fold cross validation for adaboost #####
set.seed(3)
n_folds=7
folds_i<-sample(rep(1:7,length.out=nrow(lean_data)))
cv_list<-rep(0,7)
for (k in 1:7) {
  test_i<-which(folds_i==k)
  train_i<-which(folds_i!=k)
  train_ada<-lean_data[-test_i,]
  test_ada<-lean_data[test_i,]
  ada.titanic<- ada(Survived~., data=train_ada, type = "gentle", iter = 100)
  ada.pred<-predict(ada.titanic,newdata=test_ada)
  cv_list[k]=mean(ada.pred==test_ada$Survived)

}
cv_list
mean(cv_list)

#### train and test using random forest ####
set.seed(5)
lean_train<-sample(1: nrow(lean_data), 500)
lean_data.train<-lean_data[lean_train,]
lean_data.test<-lean_data[-lean_train,]
rf.fit=randomForest(Survived~., data=titanic_train,subset=lean_train, mtry=2, importance=TRUE)
rf.yhat = predict(rf.fit, newdata=titanic_train[-lean_train,])
result=mean(rf.yhat==lean_data.test$Survived)
result

#### 7 fold cross validation using random forest####
set.seed(6)
n_folds=7
folds_i<-sample(rep(1:7,length.out=nrow(lean_data)))
cv_list<-rep(0,7)
for (k in 1:7) {
  test_i<-which(folds_i==k)
  train_i<-which(folds_i!=k)
  train_rf<-lean_data[-test_i,]
  test_rf<-lean_data[test_i,]
  rf.fit=randomForest(Survived~., data=titanic_train,subset=train_i, mtry=3,ntree=50,importance=TRUE)
  rf.yhat = predict(rf.fit, newdata=test_rf)
  cv_list[k]=mean(rf.yhat==test_rf$Survived)
}
cv_list
mean(cv_list)


#### train and test using svm ####

#### To install package 'e1071', use the following R command
library(e1071)
set.seed(7)
lean_train<-sample(1: nrow(lean_data), 500)
lean_data.train<-lean_data[lean_train,]
lean_data.test<-lean_data[-lean_train,]
svmfit = svm(Survived~., data=lean_data.train, kernel="linear", cost=0.1, scale=FALSE)
set.seed(77)
tune.out=tune(svm, Survived~., data=lean_data.train, kernel="linear",
              ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100)))
bestmod = tune.out$best.model
svmYPreds = predict(bestmod, lean_data.test)
table(predict=svmYPreds,truth=lean_data.test$Survived)
svmResult = (104+69)/214
svmResult

