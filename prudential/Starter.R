
### Notice
# This is NOT my work !!
# This is copy from https://www.kaggle.com/pruadmin/prudential-life-insurance-assessment/starter-script/code


###############################################################################
#Step 1: Read in the data and define variables as either a factor or numeric
#########################################################################################
train <- read.csv("./data/train.csv",stringsAsFactors = TRUE)
test <- read.csv("./data/test.csv", stringsAsFactors = TRUE)

# flag whether data belongs to train set or not
train$Train_Flag <-1 
test$Train_Flag <- 0
test$Response <- NA  # add Response column to test data and initializa NA

# concatenate train and test set
All_Data <- rbind(train,test)

# Define additional variables das either numeric or factor
Data_1 <- All_Data[,names(All_Data) %in% c("Product_Info_4",	"Ins_Age",	"Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep=""))]
Data_2 <- All_Data[,!(names(All_Data) %in% c("Product_Info_4",	"Ins_Age",	"Ht",	"Wt",	"BMI",	"Employment_Info_1",	"Employment_Info_4",	"Employment_Info_6",	"Insurance_History_5",	"Family_Hist_2",	"Family_Hist_3",	"Family_Hist_4",	"Family_Hist_5",	"Medical_History_1",	"Medical_History_15",	"Medical_History_24",	"Medical_History_32",paste("Medical_Keyword_",1:48,sep="")))]
Data_2<- data.frame(apply(Data_2, 2, as.factor))

# All_Data have newly formatted columns
All_Data <- cbind(Data_1, Data_2)

# remove unnecessary ones
rm(Data_1,Data_2,train,test)

str(All_Data)

##############################################################
#Step 2: Feature Creation - create some features which we want to test in a predictive model
##########################################################

# TODO

##############################################################
#Step 3: Now that we are finished with feature creation lets recreate train and test
##########################################################

train <- All_Data[All_Data$Train_Flag==1,]
test <- All_Data[All_Data$Train_Flag==0,]

set.seed(1234)
train$random <- runif(nrow(train))  # assign uniform dist r.v. for later random selection

##############################################################
#Step 4: Model building - Build a GBM on a random 70% of train and validate on the other 30% of train.
#        This will be an iterative process where you should add/refine/remove features
##########################################################
# random split on train set, 7:3
train_70 <- train[train$random <= 0.7,]
train_30 <- train[train$random >= 0.7,]

# make sure that response dist holds up well across the random split
round(table(train_70$Response)/nrow(train_70),2)
round(table(train_70$Response)/nrow(train_70),2)

#Lets build a very simple GBM on train_70 and calculate the performance on train_30
#To make the GBM run faster I will subset train_70 to only include the variables I want in the model

#train_70 <- train_70[,c("Response","BMI","Wt","Ht","Ins_Age","Number_medical_keywords","Wt_quintile")]
train_70 <- train_70[,c("Response","BMI","Wt","Ht","Ins_Age")]

library("gbm")
GBM_train <- gbm(Response ~ .,
                 data=train_70,
                 n.trees=50,
                 distribution = "multinomial",
                 interaction.depth = 5,
                 n.minobsinnode = 40,
                 shrinkage = 0.1,
                 cv.folds = 0,
                 n.cores = 1,
                 train.fraction = 1,
                 bag.fraction = 0.7,
                 verbose = TRUE)

#Use the OOB method to determine the optimal number of trees
GBM_train$opt_tree <- gbm.perf(GBM_train, method="OOB")
summary(GBM_train, n.trees=GBM_train$opt_tree)

#It looks like Ht and Wt_quintile have little impact in this model, however the number of medical keywords and BMI variables seem to have significant impact on the GBM
#We should investigate what this impact is and ensure we are not overfitting. The train_30 data would help for this, for example we could look at some actual versus predicted plots across these variables on the train_30 etc.

#Now that we have a model, lets score train_30 and calculate the quadratic kappa
Prediction_Object <- predict(GBM_train,train_30,GBM_train$opt_tree,type="response")

#an array with the probability of falling into each class for each observation (the probability will add up to 1 across all classes)
#We want to classify each application, a trivial approach would be to take the class with the highest predicted probability for each application
train_30$Prediction <- apply(Prediction_Object, 1, which.max)
round(table(train_30$Prediction)/nrow(train_30),2)
round((table(train_30$Prediction,train_30$Response)/nrow(train_30))*100,1)





