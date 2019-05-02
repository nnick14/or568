## Final Project Code ####################################

### Final Project - Nick Newman
## Logistic Regression, Random Forests, and Gradient Boosting Machines

library(tidyverse)
library(caret)
library(ggplot2)
library(scales)
library(GGally)
library(pROC)
library(RANN)

setwd("C://Users//nickn//Desktop//Grad School//OR 568")
train <- read_csv("Final Project\\application_train.csv")
test <- read_csv("Final Project\\application_test.csv")

str(train)
str(test)
dim(train)
dim(test)



## Vizualizing the data ###############################################################################

#As we can see here there is a disproportionate amount of people that repaid their loans on time
#so measuring by just accuracy would be meaningless
plotTrain <- train
plotTrain$TARGET <- as.factor(plotTrain$TARGET)

ggplot(plotTrain, aes(TARGET, fill = TARGET)) +
  geom_bar() +
  ylab("Total") +
  scale_y_continuous(labels = scales::comma) +
  scale_x_discrete(labels = c("Not Risky","Risky"))



#Checking to see the sums of NA values in each column and graphing the columns with the most NA values
NA_Count <- data.frame(Variables = names(train), Number_NA = colSums(is.na(train)))
NA_Count$Not_NA <- colSums(!is.na(train))
NA_Count$Total <- nrow(train)
NA_Count$Percentage <- round((NA_Count$Number_NA/NA_Count$Total * 100),2)

#Graph of the variables with NA Values
View(NA_Count)
NA_Count %>%
  filter(Percentage > 0) %>%
  arrange(desc(Percentage)) %>%
  ggplot(aes(reorder(Variables, Percentage), Percentage, fill = Number_NA)) +
  geom_col() +
  guides(fill = FALSE) +
  xlab("") +
  ylab("Percentage of NA Values (Color is the total number of NA Values)") +
  scale_y_continuous(breaks = c(0,10,20,30,40,50,60,70)) +
  coord_flip() 


#Graphing percentages of NA values

NA_Count %>%
  filter(Percentage > 50) %>%
  ggplot(aes(x = reorder(Variables,Percentage), Percentage, fill = Percentage)) +
  geom_bar(stat = "identity") +
  guides(fill = FALSE) +
  ylab("NA Percentage Over 50") +
  xlab(NULL) +
  coord_flip()



# Showing the discrepancy between the number of repaid loans and defaulted loans ###########################################

df1 <- as.data.frame(prop.table(table(train$TARGET)))
df2 <- as.data.frame(table(train$TARGET))
dfMerge <- merge(df1, df2, by = "Var1")

dfMerge <- dfMerge %>%
  rename(Percentage = Freq.x, Count = Freq.y, Credit_Status = Var1)

dfMerge$Credit_Status <- as.character(dfMerge$Credit_Status)
dfMerge$Credit_Status[dfMerge$Credit_Status == 0] <- "Not Risky"
dfMerge$Credit_Status[dfMerge$Credit_Status == 1] <- "Risky"

ggplot(dfMerge, aes(Credit_Status, Count, fill = Credit_Status)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(sprintf("%1.1f", Count / sum(Count) * 100),
                               "%\n", format(Count, big.mark = ",")), y = .8 * Count),
            color = "black", size = 4, position = position_stack(vjust = .75)) +
  guides(fill = FALSE) +
  scale_y_continuous(labels = scales::comma) +
  ylab("Number of Observations") +
  xlab("Loan Status")


## Feature plot with continuous variables #####################################################################

train[,c(-2,-which(sapply(train,is.character)))] %>%
  gather() %>%
  head()


#Remove all values that are characters
nonCharTrain <- train[,c(-which(sapply(train,is.character)))]

#Histograms for every continuous value in the data
graph <- list()
for (col in seq(1,ncol(nonCharTrain[-2]),21)) {
  
  graph[[col]] <- nonCharTrain[,-2] %>% 
    na.omit() %>%
    select((col:(col + 20))) %>%
    gather() %>%
    ggplot(aes(value)) +
    geom_histogram() +
    facet_wrap(~key, scales = 'free')
  
  print(graph[[col]])
  
}


# Density Plot for each value in the data

graph <- list()
for (col in seq(1,ncol(nonCharTrain) - 1,21)) {
  
  graph[[col]] <- nonCharTrain %>% 
    na.omit() %>%
    select(c(TARGET,col:(col + 20))) %>%
    gather(key, value, -TARGET) %>%
    ggplot(aes(value)) +
    geom_density(alpha = 0.5, aes(fill = factor(TARGET))) +
    facet_wrap(~key, scales = 'free')
  
  print(graph[[col]])
  
}

## Density plot for sample predictors ###

otherVar <- c("TARGET","EXT_SOURCE_3", "EXT_SOURCE_2","EXT_SOURCE_1","DAYS_ID_PUBLISH","AMT_CREDIT", "AMT_ANNUITY","AMT_GOODS_PRICE")
VarSubset <- train[,otherVar]

VarSubset %>%
  na.omit() %>%
  gather(key, value, -TARGET) %>%
  ggplot(aes(value)) +
  geom_density(alpha = 0.5, aes(fill = factor(TARGET))) +
  facet_wrap(~key, scales = 'free')



##### Variable Transformations ################################################################################

# Combine the train and test sets for variable transformations

train_test <- train %>%
  select(-TARGET) %>%
  bind_rows(test)

#Change XNA values in CODE_GENDER to Missing
train_test$CODE_GENDER[train_test$CODE_GENDER == "XNA"] <- NA

#Change XNA values in ORGANIZATION_TYPE to NA
train_test$ORGANIZATION_TYPE[train_test$ORGANIZATION_TYPE == "XNA"] <- NA

# Transform days employed variable by changing 365243 values to NA
train_test$DAYS_EMPLOYED[train_test$DAYS_EMPLOYED == 365243] <- NA

### Changing NA values in character columns to Missing #####################

train_test <- train_test %>%
  mutate_if(is.character, funs(if_else(is.na(.), "Missing", .)))

## Adding NA missing columns for variables with a high percentage of NA's
train_test[,"NA_EXT_SOURCE_1"] <- as.integer(ifelse(is.na(train_test$EXT_SOURCE_1),1,0))
train_test[,"NA_EXT_SOURCE_3"] <- as.integer(ifelse(is.na(train_test$EXT_SOURCE_3),1,0))
train_test[,"NA_OWN_CAR_AGE"] <- as.integer(ifelse(is.na(train_test$OWN_CAR_AGE),1,0))
train_test[,"NA_AMT_REQ_CREDIT_BUREAU_HOUR"] <- as.integer(ifelse(is.na(train_test$AMT_REQ_CREDIT_BUREAU_HOUR),1,0))
train_test[,"NA_AMT_REQ_CREDIT_BUREAU_DAY"] <- as.integer(ifelse(is.na(train_test$AMT_REQ_CREDIT_BUREAU_DAY),1,0))
train_test[,"NA_AMT_REQ_CREDIT_BUREAU_WEEK"] <- as.integer(ifelse(is.na(train_test$AMT_REQ_CREDIT_BUREAU_WEEK),1,0))
train_test[,"NA_AMT_REQ_CREDIT_BUREAU_MONTH"] <- as.integer(ifelse(is.na(train_test$AMT_REQ_CREDIT_BUREAU_MON),1,0))
train_test[,"NA_AMT_REQ_CREDIT_BUREAU_QUARTER"] <- as.integer(ifelse(is.na(train_test$AMT_REQ_CREDIT_BUREAU_QRT),1,0))
train_test[,"NA_AMT_REQ_CREDIT_BUREAU_YEAR"] <- as.integer(ifelse(is.na(train_test$AMT_REQ_CREDIT_BUREAU_YEAR),1,0))


## Transform with integer encoding 

## ****Our Final Models were transformed with one hot encoding****
## DO NOT RUN

train_test <- train_test %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  mutate(SOURCE_PROD = EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3,
         SOURCE_MEAN = apply(.[, c("EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3")], 1, mean),
         SOURCE_SD = apply(.[, c("EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3")], 1, sd),
         CREDIT_ANN = AMT_CREDIT / (AMT_ANNUITY + 1),
         CREDIT_INCOME = AMT_CREDIT / (AMT_INCOME_TOTAL + 1),
         INC_CREDIT = AMT_INCOME_TOTAL / (AMT_CREDIT + 1),
         CREDIT_GOODS = AMT_CREDIT / (AMT_GOODS_PRICE + 1),
         INCOME_PER_CHILD = AMT_INCOME_TOTAL / (CNT_CHILDREN + 1),
         INCOME_FAMILY_MEM = AMT_INCOME_TOTAL / (CNT_FAM_MEMBERS + 1))



## transform with one hot encoding #########################
full_rank <- dummyVars(~., data = train_test, fullRank = T)
train_oh <- predict(full_rank, train_test)
train_test <- data.frame(train_oh)

train_test <- train_test %>%
  mutate(SOURCE_PROD = EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3,
         SOURCE_MEAN = apply(.[, c("EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3")], 1, mean),
         SOURCE_SD = apply(.[, c("EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3")], 1, sd),
         CREDIT_ANN = AMT_CREDIT / (AMT_ANNUITY + 1),
         CREDIT_INCOME = AMT_CREDIT / (AMT_INCOME_TOTAL + 1),
         INC_CREDIT = AMT_INCOME_TOTAL / (AMT_CREDIT + 1),
         CREDIT_GOODS = AMT_CREDIT / (AMT_GOODS_PRICE + 1),
         INCOME_PER_CHILD = AMT_INCOME_TOTAL / (CNT_CHILDREN + 1),
         INCOME_FAMILY_MEM = AMT_INCOME_TOTAL / (CNT_FAM_MEMBERS + 1))

## Transform Back to train and test ##################

View(head(train_test))
train_test <- train_test %>%
  select(-SK_ID_CURR)

train1 <- train_test[1:nrow(train),]
test1 <- train_test[(nrow(train) + 1):nrow(train_test),]

trainNonfactor <- train
train$TARGET <- factor(ifelse(train$TARGET, "Difficult","Okay"))

## Logistic Regression ###############################################################

# make smaller dataset to tune
set.seed(101)
split <- createDataPartition(train$TARGET,p = .1)[[1]]
small_train <- train1[split,]
small_train1 <- train[split,]


set.seed(101)
ctrl <- trainControl(method = "LGOCV",
                     number = 1,
                     summaryFunction = twoClassSummary,
                     savePredictions = "final",
                     classProbs = TRUE,
                     verboseIter = TRUE)
set.seed(101)
logCredit <- train(train1,
                   train$TARGET,
                   method = "glm",
                   metric = "ROC",
                   trControl= ctrl,
                   preProc = "medianImpute")
logCredit


Prediction <- predict(logCredit, newdata = test1, type = "prob")
Prediction
write_csv(data.frame(SK_ID_CURR = as.integer(test$SK_ID_CURR), TARGET = Prediction$Difficult),"submission1.csv")


varImp(logCredit)
plot(varImp(logCredit), top = 20, main = "Logistic Regression Top 20 Predictors")

ggplot(varImp(logCredit), top = 20) +
  ggtitle("Logistic Regression Top 20 Predictors") +
  theme(axis.title.y = element_blank())

plot.roc(logCredit$pred$obs,
         logCredit$pred$Difficult, main = "Logistic Regression")

creditROCLR <- roc(logCredit$pred$obs,
                 logCredit$pred$Difficult)

auc(creditROCLR)
ci.auc(creditROCLR)

bestCoordLR <- coords(creditROCLR, x = "best", ret = "threshold",
                    best.method = "closest.topleft")

plot(creditROCLR, print.thres = c(.15,.05,bestCoordLR), type = "S",
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, legacy.axes = TRUE,
     print.auc = TRUE,
     print.auc.adj = c(3.5,-17),
     print.auc.cex = 1.2,
     main = "Logistic Regression")

## Ranger random forest ##################################

# First tune the random forest on the smaller dataset so it
# takes less time

tgrid <- expand.grid(
  .mtry = c(10,15,20,25),
  .splitrule = "gini",
  .min.node.size = c(1,2,3)
)

set.seed(101)
rfCreditsmall <- train(x = small_train, 
                  y = small_train1$TARGET,
                  method = "ranger",
                  tuneGrid = tgrid,
                  num.trees = 1500,
                  metric = "ROC",
                  trControl = ctrl,
                  importance = "impurity",
                  preProc = "medianImpute")

rfCreditsmall


## Using the full dataset
tgrid <- expand.grid(
  .mtry = c(15),
  .splitrule = "gini",
  .min.node.size = c(2)
)

set.seed(101)
rfCredit <- train(x = train1, 
                  y = train$TARGET,
                  method = "ranger",
                  tuneGrid = tgrid,
                  num.trees = 1500,
                  metric = "ROC",
                  trControl = ctrl,
                  importance = "impurity",
                  preProc = "medianImpute")

rfCredit

plot(varImp(rfCredit), top = 20, main = "Random Forest Top 20 Predictors")

ggplot(varImp(rfCredit), top = 20) +
  ggtitle("Random Forest Top 20 Predictors") +
  theme(axis.title.y = element_blank())


plot.roc(rfCredit$pred$obs,
         rfCredit$pred$Difficult, main = "Random Forest")

creditROCRF <- roc(rfCredit$pred$obs,
                 rfCredit$pred$Difficult)

auc(creditROCRF)
ci.auc(creditROCRF)

bestCoord <- coords(creditROCRF, x = "best", ret = "threshold",
       best.method = "closest.topleft")

plot(creditROCRF, print.thres = c(.15,.05,bestCoord), type = "S",
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, legacy.axes = TRUE,
     print.auc = TRUE,
     print.auc.adj = c(3.5,-17),
     print.auc.cex = 1.2,
     main = "Random Forest")


Prediction <- predict(rfCredit, newdata = test1, type = "prob")
write_csv(data.frame(SK_ID_CURR = as.integer(test$SK_ID_CURR), TARGET = Prediction$Difficult),"submission1.csv")

summary(rfFit$results)
rfFit$results

## XGBoost - Gradient Boosting #########################################
library(xgboost)


xgbTrainTest <- train_test %>% data.matrix()

xgbTrain <- xgbTrainTest[1:nrow(train),]
xgbTest <- xgbTrainTest[(nrow(train) + 1):nrow(train_test),]

testMatrix <- xgb.DMatrix(data = xgbTest)


# Creating a smaller training and valdation set in order to tune the model parameters
set.seed(101)
xgSplit <- createDataPartition(trainNonfactor$TARGET,p = .1)[[1]]
gbTrain <- xgb.DMatrix(data = xgbTrain[xgSplit,], label = trainNonfactor$TARGET[xgSplit])
train2 <- train[-xgSplit,]
set.seed(101)
xgSplit2 <- createDataPartition(train2$TARGET,p = .1)[[1]]
gbVal <- xgb.DMatrix(data = xgbTrain[xgSplit2,], label = trainNonfactor$TARGET[xgSplit2])


## Tuning over hyperparameters ############
gbmHyper <- expand.grid(
  eta = c(.05),
  max_depth = c(1,2,3,4,5,6),
  min_child_weight = c(5,7,10,15),
  subsample = c(1),
  colsample_bytree = c(.7),
  alpha = 0,
  lambda = 0,
  gamma = 0
)
nrow(gbmHyper)


for(i in 1:nrow(gbmHyper)) {
  gbmParamsshort <- list(eta = gbmHyper$eta[i],
                    max_depth = gbmHyper$max_depth[i],
                    min_child_weight = gbmHyper$min_child_weight[i],
                    subsample = gbmHyper$subsample[i],
                    colsample_bytree = gbmHyper$subsample[i],
                    objective = "binary:logistic",
                    booster = "gbtree",
                    eval_metric = "auc",
                    nrounds = 2500,
                    nthread = 4,
                    alpha = 0,
                    lambda = 0,
                    gamma = 0
                    
  )
  
  set.seed(101)
  xgb <- xgb.train(gbmParamsshort, gbTrain, gbmParamsshort$nrounds, list(val = gbVal), print_every_n = 50, early_stopping_rounds = 300)
  
  gbmHyper$best[i] <- xgb$best_score 
  
}


# Best results were 5 max depth and 5 min child weight
gbmHyper


## Full set  ########################################

gbmParams <- list(eta = .01,
                  max_depth = 5,
                  min_child_weight = 5,
                  subsample = 1,
                  colsample_bytree = .7,
                  objective = "binary:logistic",
                  booster = "gbtree",
                  eval_metric = "auc",
                  nrounds = 5000,
                  nthread = 4,
                  alpha = 0,
                  lambda = 0,
                  gamma = 0
)

set.seed(101)
split <- createDataPartition(train$TARGET,p = .8)[[1]]
gbFullTrain <- xgb.DMatrix(data = xgbTrain[split,], label = trainNonfactor$TARGET[split])
gbFullVal <- xgb.DMatrix(data = xgbTrain[-split,], label = trainNonfactor$TARGET[-split])

set.seed(101)
xgb <- xgb.train(gbmParams, gbFullTrain, gbmParams$nrounds, list(val = gbFullVal), print_every_n = 50, early_stopping_rounds = 300)

write_csv(data.frame(SK_ID_CURR = as.integer(test$SK_ID_CURR), TARGET = predict(xgb, newdata = testMatrix)),"submission1.csv")


xgb.plot.importance(xgb.importance(model = xgb), top_n = 20,
                    left_margin = 12, main = "Gradient Boosting Top 20 Predictors")


plot(xgb.importance(model = xgb))

pred <- predict(xgb, newdata = gbFullVal)
GB.ROC <- roc(predictor = pred,
              response = train[-split,]$TARGET,
              levels = rev(levels(train[split,]$TARGET)))

GB.ROC

auc(GB.ROC)
ci.auc(GB.ROC)
plot.roc(GB.ROC,
         main = "GBM")

bestCoordGB <- coords(GB.ROC, x = "best", ret = "threshold",
       best.method = "closest.topleft")

plot(GB.ROC, print.thres = c(.15,.05, bestCoordGB), type = "S",
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, legacy.axes = TRUE,
     print.auc = TRUE,
     print.auc.adj = c(3.5,-17),
     print.auc.cex = 1.2,
     main = "Gradient Boosting Machine")



xgb.importance(feature_names = colnames(train_test),xgb)




