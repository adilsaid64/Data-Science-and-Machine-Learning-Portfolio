#import data set
dataset = read.csv("Data.csv")
View(dataset)


#missing data, replacing by mean of columns
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), 
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

#encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1,2,3))


dataset$Purchased = factor(dataset$Purchased,
                         levels = c("No", "Yes"),
                         labels = c(0,1))

#splitting into test and training data sets.

#install.packages("caTools")

library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
#To avoid certain features/variables dominating others. Putting everything on the same scale.

training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])




