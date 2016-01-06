### Notice
# This is NOT my work !!
# This is copy from https://www.kaggle.com/yildirimarda/titanic/titanic-test3

library("rpart")
library("rpart.plot")

#####################################################
# Read data
#####################################################
train <- read.csv("./data/train.csv")
test <- read.csv("./data/test.csv")
test$Survived <- 0

#####################################################
# Clean data
#####################################################
combi <- rbind(train,test)

# Drop cabin because it has so much of null data, and it is irrelavant to learning
combi$Cabin <- NULL

# extract Mr, Mrs, Miss from name and assign to Title
combi$Name <- as.character(combi$Name)
combi$Title <- sapply(combi$Name, FUN=function(x){
  strsplit(x,split='[,.]')[[1]][2]
})
# handling of other values other than Mr, Mrs, Miss
combi$Title <- sub(' ', '', combi$Title) # trim
combi$Title[combi$PassengerId==797] <- 'Mrs' # female doctor
combi$Title[combi$PassengerId==370] <- 'Miss' # female
combi$Title[combi$Title %in% c('Lady', 'the Countess', 'Mlle', 'Mee', 'Ms')] <- 'Miss'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Col', 'Jonkheer', 'Rev', 'Dr', 'Master')] <- 'Mr'
combi$Title[combi$Title %in% c('Dona')] <- 'Mrs'
# factorize title
combi$Title <- factor(combi$Title)

# Embarkment : from which area passengers board on
# Passenger on row 62 and 830 do not have a value for embarkment. 
# Since many passengers embarked at Southampton, we give them the value S.
# We code all embarkment codes as factors.
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)

# Passenger on row 1044 has an NA Fare value. Let's replace it with the median fare value.
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

# Create new column -> family_size
combi$family_size <- combi$SibSp + combi$Parch + 1

# How to fill in missing Age values?
# We make a prediction of a passengers Age using the other variables and a decision tree model. 
# This time you give method="anova" since you are predicting a continuous variable.
predicted_age <- rpart(Age ~ Pclass+Sex+SibSp+Parch+Fare+Embarked+Title+family_size,
                       data=combi[!is.na(combi$Age),],method="anova")
combi$Age[is.na(combi$Age)] <- predict(predicted_age, combi[is.na(combi$Age),])

#####################################################
# Create decision tree model for response variable : Survived
#####################################################
train_new <- combi[1:891,]
test_new <- combi[892:1309,]
test_new$Survied <- NULL

my_tree <- rpart(Survived ~ Age + Sex + Pclass + family_size, data = train_new, 
                 method="class", control=rpart.control(cp=0.0001))
summary(my_tree)

# Visualize decision tree
prp(my_tree, type=4, extra = 100)

# prediction
my_prediction <- predict(my_tree, test_new, type="class")
head(my_prediction)

# solution
vector_passengerid <- test_new$PassengerId
my_solution <- data.frame(PassengerId = vector_passengerid, Survived = my_prediction)
head(my_solution)





