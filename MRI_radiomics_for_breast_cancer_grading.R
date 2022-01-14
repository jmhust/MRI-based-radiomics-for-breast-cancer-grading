library(ElemStatLearn) #contains the data
library(car) #package to calculate Variance Inflation Factor
library(corrplot) #correlation plots
library(leaps) #best subsets regression
library(glmnet) #allows ridge regression, LASSO and elastic net
library(caret) #this will help identify the appropriate parameters
library(ALBERT)
library(glmnet)    # cv.glmnet
library(rms)       # nomogram
library(pROC)      # ROC
library(corrplot)  # 相关系数囿
library(ggplot2) 
library("devtools")
library("usethis")
library("DynNom")
library(devtools)
library(ALBERT)
library(glmnet)    # cv.glmnet
library(rms)       # nomogram
library(pROC)      # ROC
library(corrplot)  # 鐩稿叧绯绘暟鍥?
library(ggplot2)   
library(data.table)
library(lattice)
library(caret)

#### lasso feature selection

setwd("D:/R work/Nottingham histological grade")
train<-read.csv("train.csv") 
test<-read.csv("test.csv") 

x <- as.matrix(train[, 3:531])
y <- train[, 2]

set.seed(1234)
lasso.cv <- cv.glmnet(x,y,family = "binomial",type.measure = "auc", alpha = 1,nfolds=50) 
plot(lasso.cv)
print(lasso.cv)
fit <- glmnet(x,y,family = "binomial",type.measure = "auc", alpha = 1,nfolds=50,parallel=TRUE) 

#输出系数
coefPara <- coef(lasso.cv, s = "lambda.min")
beta <- as.matrix(coefPara[which(coefPara != 0), ])
betai_Matrix <- as.matrix(coefPara[which(coefPara != 0), ])
betai_Matrix
write.csv(betai_Matrix,"betai_Matrix.csv")


##变量收敛
get_plot<- function(the_fit,the_fit_cv,the_lamb,toplot = seq(1,50,2)){
  Coefficients <- coef(the_fit, s = the_lamb)
  Active.Index <- which(Coefficients != 0)
  coeall <- coef(the_fit, s = the_fit_cv$lambda[toplot])
  coe <- coeall[Active.Index[-1],]
  ylims=c(-max(abs(coe)),max(abs(coe)))
  sp <- spline(log(the_fit_cv$lambda[toplot]),coe[1,],n=100)
  plot(sp,type='l',col =1,lty=1, 
       ylim = c(-2,2),ylab = 'Coefficient', xlab = 'log(lambda)') #
  abline(h=0) 
  for(i in c(2:nrow(coe))){
    lines(spline(log(the_fit_cv$lambda[toplot]),coe[i,],n=1000),
          col =i,lty=i)
  }
         
}

# 传入最优lambda，从而保留相关变量
get_plot(fit,lasso.cv,exp(log(lasso.cv$lambda.min)-1))




### model assessment
library(ROCR)
get_confusion_stat <- function(pred,y_real,threshold=0.5){
  # auc
  tmp <- prediction(as.vector(pred),y_real)
  auc <- unlist(slot(performance(tmp,'auc'),'y.values'))
  # statistic
  pred_new <- as.integer(pred>threshold) 
  tab <- table(pred_new,y_real)
  if(nrow(tab)==1){
    print('preds all zero !')
    return(0)
  }
  TP <- tab[2,2]
  TN <- tab[1,1]
  FP <- tab[2,1]
  FN <- tab[1,2]
  accuracy <- round((TP+TN)/(TP+FN+FP+TN),4)
  recall_sensitivity <- round(TP/(TP+FN),4)
  precision <- round(TP/(TP+FP),4)
  specificity <- round(TN/(TN+FP),4)
  neg_rate <- round((TN+FN)/(TP+TN+FP+FN),4)
  re <- list('AUC' = auc,
             'Confusion_Matrix'=tab,
             'Statistics'=data.frame(value=c('accuracy'=accuracy,
                                             'recall_sensitivity'=recall_sensitivity,
                                             'precision'=precision,
                                             'specificity'=specificity,
                                             'neg_rate'=neg_rate)))
  return(re)
}

## Performance on the training cohort
get_eval <- function(train,theta=0.5,the_fit=fit,the_lamb=lasso.cv$lambda.min){
  
  y <- train[, 2]
  x <- as.matrix(train[, 3:531])
  pred <- predict(the_fit,newx=x,s=the_lamb,type = 'response')
  print(get_confusion_stat(pred,y, theta))
}

get_eval(train)


## performance on the test cohort
get_eval <- function(test,theta=0.5,the_fit=fit,the_lamb=lasso.cv$lambda.min){
  
  y <- test[, 2]
  x <- as.matrix(test[, 3:531])
  pred <- predict(the_fit,newx=x,s=the_lamb,type = 'response')
  print(get_confusion_stat(pred,y, theta))
}

get_eval(test)


###Building the MRI radiomics signature for breast cancer grading

## training cohort
train_new<-transform(train, rad_score=-2.545154383
                  -0.246038256*BEDR1_Tumor
                  -0.981198836*BEDR2_Tumor
                  +1.953559302*Inf_mea_of_corr2_Tumor
                  +1.482906551*Grouping_based_proportion_of_tumor_voxels_3D_tumor_Group_1
                  -1.093604877*Grouping_based_mean_of_washout_slope_3D_tissue_PostCon_Group_2
                  -3.01E-05*X2nd_DFT_CoeffMap_Momment_Invariant_3_3D_tumor
                  -3.34E-06*X2nd_DFT_CoeffMap_Momment_Invariant_4_3D_tumor
                  -0.06602994*X2nd_DFT_CoeffMap_Momment_Invariant_3_2D_tumorSlice
                  +9.96E-05*Mean_norm_DLBP_max_timepoint_binsize_256_with_filling_Tumor
                  -3.74E-06*Cluster_Prominence_tissue_PostCon
                  -0.000709476*Cluster_Shade_tissue_PostCon
                  -0.030595658*sum_entropy_tissue_PostCon
                  +9.98E-07*SER_Partial_tumor_vol_cu_mm
                  +0.000323005*SER_map_sum_average_tumor
                  -0.807372532*PE_map_information_measure_correlation2_tumor
                  -7.72E-09*WashinRate_map_Cluster_Prominence_tumor
                  +0.527481296*WashinRate_map_Entropy_tumor
                  -0.413569223*WashinRate_map_information_measure_correlation2_tumor
                  +0.587715431*WashinRate_map_skewness_tumor
                  -0.000100483*SER_map_kurtosis_tissue_T1
                  -1.18E-05*WashinRate_map_Autocorrelation_tissue_PostCon
                  )

write.csv(train_new,"train_new.csv")

## test cohort
test_new<-transform(test, rad_score=-2.545154383
                 -0.246038256*BEDR1_Tumor
                 -0.981198836*BEDR2_Tumor
                 +1.953559302*Inf_mea_of_corr2_Tumor
                 +1.482906551*Grouping_based_proportion_of_tumor_voxels_3D_tumor_Group_1
                 -1.093604877*Grouping_based_mean_of_washout_slope_3D_tissue_PostCon_Group_2
                 -3.01E-05*X2nd_DFT_CoeffMap_Momment_Invariant_3_3D_tumor
                 -3.34E-06*X2nd_DFT_CoeffMap_Momment_Invariant_4_3D_tumor
                 -0.06602994*X2nd_DFT_CoeffMap_Momment_Invariant_3_2D_tumorSlice
                 +9.96E-05*Mean_norm_DLBP_max_timepoint_binsize_256_with_filling_Tumor
                 -3.74E-06*Cluster_Prominence_tissue_PostCon
                 -0.000709476*Cluster_Shade_tissue_PostCon
                 -0.030595658*sum_entropy_tissue_PostCon
                 +9.98E-07*SER_Partial_tumor_vol_cu_mm
                 +0.000323005*SER_map_sum_average_tumor
                 -0.807372532*PE_map_information_measure_correlation2_tumor
                 -7.72E-09*WashinRate_map_Cluster_Prominence_tumor
                 +0.527481296*WashinRate_map_Entropy_tumor
                 -0.413569223*WashinRate_map_information_measure_correlation2_tumor
                 +0.587715431*WashinRate_map_skewness_tumor
                 -0.000100483*SER_map_kurtosis_tissue_T1
                 -1.18E-05*WashinRate_map_Autocorrelation_tissue_PostCon)

write.csv(test_new,"test_new.csv")


## NHG2 prognostic cohort
setwd("D:/R work/Nottingham histological grade")
NHG2<-read.csv("NHG2.csv") 

NHG2_new<-transform(NHG2, rad_score=-2.545154383
                      -0.246038256*BEDR1_Tumor
                      -0.981198836*BEDR2_Tumor
                      +1.953559302*Inf_mea_of_corr2_Tumor
                      +1.482906551*Grouping_based_proportion_of_tumor_voxels_3D_tumor_Group_1
                      -1.093604877*Grouping_based_mean_of_washout_slope_3D_tissue_PostCon_Group_2
                      -3.01E-05*X2nd_DFT_CoeffMap_Momment_Invariant_3_3D_tumor
                      -3.34E-06*X2nd_DFT_CoeffMap_Momment_Invariant_4_3D_tumor
                      -0.06602994*X2nd_DFT_CoeffMap_Momment_Invariant_3_2D_tumorSlice
                      +9.96E-05*Mean_norm_DLBP_max_timepoint_binsize_256_with_filling_Tumor
                      -3.74E-06*Cluster_Prominence_tissue_PostCon
                      -0.000709476*Cluster_Shade_tissue_PostCon
                      -0.030595658*sum_entropy_tissue_PostCon
                      +9.98E-07*SER_Partial_tumor_vol_cu_mm
                      +0.000323005*SER_map_sum_average_tumor
                      -0.807372532*PE_map_information_measure_correlation2_tumor
                      -7.72E-09*WashinRate_map_Cluster_Prominence_tumor
                      +0.527481296*WashinRate_map_Entropy_tumor
                      -0.413569223*WashinRate_map_information_measure_correlation2_tumor
                      +0.587715431*WashinRate_map_skewness_tumor
                      -0.000100483*SER_map_kurtosis_tissue_T1
                      -1.18E-05*WashinRate_map_Autocorrelation_tissue_PostCon)

write.csv(NHG2_new,"NHG2_new.csv")

