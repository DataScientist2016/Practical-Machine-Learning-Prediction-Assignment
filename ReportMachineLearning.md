


# Practical Machine Learning / Prediction Assignment
## Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, and to predict the manner in which they did the exercise (they were asked to perform barbell lifts correctly and incorrectly in 5 different ways).

## Loading and reading data
More information about the data is available from the website here: http://groupware.les.inf.puc-rio.br/har.

On the first step we do setwd() and then load the data. After that we load the required R libraries.


```r
if(!file.exists("project")){
  dir.create("project")
}

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile="./project/pml-training.csv",method="auto")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile="./project/pml-testing.csv",method="auto")
list.files("./project")

training <- read.csv("./project/pml-training.csv")
testing <- read.csv("./project/pml-testing.csv")

dim(training); dim(testing)

library(caret)
library(corrplot)
library(rattle)
```

The *training* dataset has 19622 rows and 160 columns, while the *testing* dataset has 20 rows and 160 columns.

## Cleaning Data
The number of variables is too big. We should explore them and clear our datasets. This we do in three steps:

### Removing Near Zero Variables

```r
near0var <- nearZeroVar(training, saveMetrics = TRUE)
head(near0var, 15)
training1 <- training[, !near0var$nzv]
testing1 <- testing[, !near0var$nzv]
dim(training1); dim(testing1)
```
After removing Near Zero Variables both datasets contain each 100 columns/variables.

### Removing NA

```r
training2 <- training1[, (colSums(is.na(training1)) == 0)]
testing2 <- testing1[, (colSums(is.na(training1)) == 0)]
dim(training2); dim(testing2)
```
After removing variables containing NAs both datasets have each 59 columns/variables.

### Removing other useless variables

```r
training3 <- training2[, !(grepl("^X|user_name|timestamp|num_window", names(training2)))]
testing3 <- testing2[, !(grepl("^X|user_name|timestamp|num_window", names(training2)))]
dim(training3); dim(testing3)
```
After removing some other useless variables both datasets contain each 53 columns/variables.

### Removing highly correlated variables.

```r
corrM <- cor(training3[, -length(names(training3))])
corrplot(corrM, method = "circle", tl.cex = 0.7)
```

![plot of chunk corr](figure/corr-1.png)

```r
rmcorr <- findCorrelation(corrM, cutoff = .90, verbose = TRUE)
training4 <- training3[,-rmcorr]
testing4 <- testing3[,-rmcorr]
dim(training4); dim(testing4)
```

After removing highly correlated variables both datasets contain each 46 columns/variables.

## Partitioning training set for Cross Validation

For validation purposes we will split the cleaned training dataset in two sets, one for training and one for cross validation. The smaller *final training dataset* will contain 70% of the cleaned training dataset and the *cross validation dataset* will contain 30% of the dataset. The reason for such procedure is that after we obtain our model, we will use the cross validation data set to test the accuracy of our model.


```r
set.seed(23232) # reproducibility
partition <- createDataPartition(y = training4$classe, p = 0.7, list = FALSE)
training.final = training4[partition, ]
validation = training4[-partition, ]
dim(training.final); dim(validation)
```

```
## [1] 13737    46
```

```
## [1] 5885   46
```

## Choosing a model

### Predicting with DECISION TREE

First we try DESISION TREE. 


```r
set.seed(23232)
modFitTREE <- train(classe ~ .,method="rpart",data=training.final)
fancyRpartPlot(modFitTREE$finalModel)
```

![plot of chunk TREE](figure/TREE-1.png)

```r
predTREE <- predict(modFitTREE,newdata=validation)
cfTREE <- confusionMatrix(predTREE,validation$classe)
```


```r
cfTREE$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1068  210   26   59   18
##          B   25  491   18   94  202
##          C  421  373  977  416  436
##          D  160   65    4  311   72
##          E    0    0    1   84  354
```

```r
cfTREE$overall[1]
```

```
##  Accuracy 
## 0.5439252
```
The results with DECISION TREE are very poor. We can exclude this model already on this stage.

### Predicting with GBM

```r
set.seed(23232)
modelFitGBM <- train(classe ~.,data=validation, method="gbm")
predGBM <- predict(modelFitGBM,newdata=validation)
cfGBM <- confusionMatrix(predGBM,validation$classe)
```


```r
cfGBM$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1662   22    0    1    1
##          B    6 1102   20    1    3
##          C    4   12  997   20    7
##          D    2    0    6  941   10
##          E    0    3    3    1 1061
```

```r
cfGBM$overall[1]
```

```
##  Accuracy 
## 0.9792693
```

We have got very good results. But let's try one more model.

### Predicting with RANDOM FOREST

And now we try Random Forest algorithm. It selects automatically  important variables and is robust to correlated covariates and outliers in general.


```r
set.seed(23232)
modelFitRF <- train(classe ~.,data=validation, method="rf")

predRF <- predict(modelFitRF,newdata=validation)

cfRF <- confusionMatrix(predRF,validation$classe)
```


```r
cfRF$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    0    0    0
##          C    0    0 1026    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
```

```r
cfRF$overall[1]
```

```
## Accuracy 
##        1
```

The results are perfect. We choose this model for final predictions.

### Out of Sample Error


```r
control <- trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(23232)
modelRF <- train(classe ~ ., data=training.final, method="rf", trControl=control)
M <- confusionMatrix(predict(modelRF, newdata=validation), validation$classe)
ose <- 1 - M$overall[1] 
names(ose)<- "Out of Sample Error"
ose
```

```
## Out of Sample Error 
##          0.00815633
```

## Final predictions / RESULTS

The Random Forest model produces the best (perfect) results. So we use it to predict our 20 values in the testing set.

```r
predRFtest <- predict(modelFitRF,newdata=testing4)
predRFtest
```

```
##  [1] B A B A A E D D A A B C B A E E A B B B
## Levels: A B C D E
```

