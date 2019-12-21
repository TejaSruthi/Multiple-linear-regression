#load the dataset
datset = read.csv('50_Startups.csv')

# encoding categorical data
datset$State = factor(datset$State, levels = c('California','New York','Florida'),
                      labels = c(2,1,3))

library('caTools')
set.seed(123)
split = sample.split(datset$Profit, SplitRatio = 0.8)
train_set = subset(datset, split == TRUE)
test_set = subset(datset, split == FALSE)

# fitting the linear regression to train_set
regressor = lm(formula = Profit ~ R.D.Spend,data = train_set)

#predict test_set results
y_pred = predict(regressor, newdata = test_set)

# Building the optinal model using backward elimination
regressor = lm(formula = Profit ~ R.D.Spend ,
               data = datset)
summary(regressor)