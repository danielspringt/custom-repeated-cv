library(xgboost)
library(Matrix)
library(caret)




training <- train
#-------------------------  eval  ----------------------------#
auc_function <- function (actual, predicted) {
  
  r <- as.numeric(rank(predicted))
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
}  
#-------------------------  eval  ----------------------------#


#------------------------  params  ---------------------------#
param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.02,
                max_depth           = 5,
                subsample           = 0.9,
                colsample_bytree    = 0.6
               # min_child_weight    = 5
)

p = 5
rows = nrow(training)

mean_result <- list()
aucs <- vector()
folds <- list()

#--------------------- create folds ---------------------------#
for(s in 1:p){
  set.seed(222 * s)
  tmp.flds <- list()
  
  flds <- createFolds(train.y, k = 5, list = TRUE)
  
  folds[[s]] <- flds
}


#------------------------  params  ---------------------------#


#--------------------------  CV  -----------------------------#
for(j in 1:p){  
  result <- vector()
  round.flds <- folds[[j]]
 
  for(i in 1:length(round.flds)){
    
    test_index <- unlist(round.flds[i])
    train_index <- unlist(round.flds[-i])
    
    
    cv_train = training[train_index,]
    cv_test = training[test_index,]
    
    target_tr <- train.y[train_index]
    target_te <- train.y[test_index]
    cv_train$TARGET <- as.factor(target_tr)
    dtrain <- xgb.DMatrix(data = as.matrix(cv_train), label = target_tr)
    
    set.seed(1234)
    #--------------------------  Model  ----------------------#
    cvboost <- xgboost(   params              = param,
                          data                = dtrain,
                          nrounds             = 200, 
                          verbose             = F,
                          maximize            = FALSE
    )
    
    
    cv_test <- as.matrix(cv_test)
    dtest <- xgb.DMatrix(data=cv_test)
    ## Predict
    preds <- predict(cvboost, cv_test)
    #--------------------------  Model  ----------------------#
    result <- c(result, auc_function(target_te, preds))
    
    print(paste("round",j,"/",p ,"___ fold", i,"/",length(flds), "___ AUC =", result[i]))
  }
  aucs <- c(aucs,mean(result))
}  
cat("mean: ",mean(aucs))
cat("min: ",min(aucs))
cat("max: ",max(aucs))
cat("diff: ", max(aucs)-min(aucs))
cat("sd:", sd(aucs))
#-----------------------  CV  ------------------------------#
  
