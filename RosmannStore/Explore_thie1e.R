
### Notice
# This is NOT my work !!
# This is copy from https://www.kaggle.com/chechir/prudential-life-insurance-assessment/features-predictive-power

library(data.table)
library(zoo)
library(forecaset)
library(ggplot2)

# load data
train <- fread("./data/train.csv")
test <- fread("./data/test.csv")
store <- fread("./data/store.csv")

#str(train)
#str(test)
#str(store)
summary(train)
summary(test)

# Date format and sort by date
train[, Date := as.Date(Date)]
test[, Date := as.Date(Date)]
train <- train[order(Date)]
test <- test[order(Date)]

# imputation
test[is.na(test$Open),]
test[test$Store==622]
test[is.na(test)] <- 1  # may be open 

# unique values per column
train[, lapply(.SD, function(x) length(unique(x)))]
test[, lapply(.SD, function(x) length(unique(x)))]

# All test stores are also in the train data
# test whether there will be unseen data
# 259 train stores are not in the test data
sum(unique(test$Store) %in% unique(train$Store))
sum(!unique(train$Store) %in% unique(test$Store))

# Simple facts
table(train$Open)/nrow(train) # Percent of Open Train
table(test$Open)/nrow(test) # Percent of Open Test
table(train$Promo)/nrow(train)
table(test$Promo)/nrow(test)
table(train$StateHoliday)/nrow(train)
table(test$StateHoliday)/nrow(test)
table(train$SchoolHoliday)/nrow(train)
table(test$SchoolHoliday)/nrow(test)

plot(train$Date, type="l")
plot(test$Date, type="l")

# As expected all 856 stores to be predicted daily
all(table(test$Date)==856)

# look at dist of sales train
hist(train$Sales,100)
hist(aggregate(train[Sales!=0]$Sales,
               by = list(train[Sales!=0]$Store),mean
               )$x,
     100,
     main="Mean sales per store when store was not closed"
     )

# look at dist of customers train
hist(train$Customers,100)
hist(aggregate(train[Sales!=0]$Customers,
               by = list(train[Sales!=0]$Store),mean
               )$x,
      100,
      main="Mean customers per store when store was not closed"
     )

# look at dist of Scool holiday train
ggplot(train[Sales!=0], aes(x=factor(SchoolHoliday),y=Sales)) +
  geom_jitter(alpha=0.1) +
  geom_boxplot(color="yellow",outlier.colour = NA, fill=NA)

# number of custormer vs sales per each day
ggplot(train[ train$Sales!=0 & train$Customers!=0 ],
        aes(x=log(Customers), y=log(Sales))) +
  geom_point(alpah=0.2) + geom_smooth()

# look at dist of Promo train
ggplot(train[Sales!=0 & train$Customers !=0], 
       aes(x=factor(Promo),y=Sales)) +
  geom_jitter(alpha=0.1) +
  geom_boxplot(color="yellow",outlier.colour = NA, fill=NA)
ggplot(train[Sales!=0 & train$Customers !=0], 
       aes(x=factor(Promo),y=Customers)) +
  geom_jitter(alpha=0.1) +
  geom_boxplot(color="yellow",outlier.colour = NA, fill=NA)

#This would mean that the promos are not mainly attracting more customers but make customers spend more
with(train[train$Sales!=0 & train$Promo==0],mean(Sales/Customers))
with(train[train$Sales!=0 & train$Promo==1],mean(Sales/Customers))

# no sales even open : 54
table(ifelse(train$Open==1, "Opened", "Closed"),
      ifelse(train$Sales>0, "Sales>0", "Sales=0"))  
train[Open==1 & Sales == 0]

# hist of zero sale per store
zerosPerStore <- sort(tapply(train$Sales,list(train$Store), function(x) sum(x==0)))
hist(zerosPerStore,100)

# store with the most zeros in their sales
tail(zerosPerStore,10)

# Some stores were closed for some time, some of those were closed multiple times
plot(train[Store==972,Sales],ylab="Sales",xlab="Days",main="Store 972")
plot(train[Store == 103, Sales], ylab = "Sales", xlab = "Days", main = "Store 103")
plot(train[Store == 708, Sales], ylab = "Sales", xlab = "Days", main = "Store 708")

# Sales high when opened on sundays/holidays
ggplot(train[Store==85],
       aes(x=Date, y=Sales,
           color=factor(DayOfWeek==7),
           shape=factor(DayOfWeek==7))) +
  geom_point(size=3) +
  ggtitle("Sales of store 85 (True if sunday)")
ggplot(train[Store==262],
       aes(x=Date, y=Sales,
           color=factor(DayOfWeek==7),
           shape=factor(DayOfWeek==7))) +
  geom_point(size=3) +
  ggtitle("Sales of store 85 (True if sunday)")


# That is not true in general. The variability of sales on sundays is quite high while the median is not:
ggplot(train[Sales!=0],
       aes(x=factor(DayOfWeek), y=Sales)) +
  geom_jitter(alpha=0.1) +
  geom_boxplot(color="yellow",outlier.colour = NA, fill = NA)


############## Store
table(store$StoreType)
table(store$Assortment)
# There is a connection between store type and type of assortment
table(data.frame(Assortment = store$Assortment, StoreType=store$StoreType))

# hist of compete distance
hist(store$CompetitionDistance,100)

# hist of compete open since
# Convert the CompetitionOpenSince... variables to one Date variable
store$CompetitionOpenSince <- as.yearmon(paste(store$CompetitionOpenSinceYear,
                                               store$CompetitionOpenSinceMonth,sep="-"))
hist(as.yearmon("2015-10") - store$CompetitionOpenSince,100,
     main="Years since opening of nearest competition")  

# Convert the Promo2Since... variables to one Date variable
# Assume that the promo starts on the first day of the week
store$Promo2Since <- as.POSIXct(paste(store$Promo2SinceYear,
                                      store$Promo2SinceWeek,1,sep="-"),
                                format = "%Y-%U-%u")
hist(as.numeric(as.POSIXct("2015-10-01",format="%Y-%m-%d") - store$Promo2Since),
     100, main="Days since start of promo2")

table(store$PromoInterval)

# merge store and train
train_store <- merge(train,store, by="Store")

# effect of promoInterval on sales
ggplot(train_store[Sales!=0], aes(x=factor(PromoInterval),y=Sales)) +
  geom_jitter(alpha=0.1) +
  geom_boxplot(color="yellow", outlier.colour = NA, fill = NA )

# effect of store type on sales
ggplot(train_store[Sales!=0],
       aes(x=as.Date(Date), y=Sales, color=factor(StoreType))) +
  geom_smooth(size=2)
# effect of store type on customers
ggplot(train_store[Customers!=0],
       aes(x=as.Date(Date),y=Customers,color=factor(StoreType))) +
  geom_smooth(size=2)
# effect of store assortment on sales
ggplot(train_store[Sales!=0],
       aes(x=as.Date(Date),y=Sales,color=factor(Assortment))) +
  geom_smooth(size=2)
# effect of store type on customers

# The effect of the distance to the next competitor is a little counterintuitive. Lower distance to the next competitor implies (slightly, possibly not significantly) higher sales. This may occur (my assumption) because stores with a low distance to the next competitor are located in inner cities or crowded regions with higher sales in general. Maybe the effects of being in a good / bad region and having a competitor / not having a competitor cancel out:
salesByDist <- aggregate(train_store[Sales!=0 & !is.na(CompetitionDistance)]$Sales,
                         by = list(train_store[Sales != 0 & !is.na(CompetitionDistance)]$CompetitionDistance), mean)
colnames(salesByDist) <- c("CompetitionDistance", "MeanSales")
ggplot(salesByDist, aes(x=log(CompetitionDistance),y=log(MeanSales))) +
  geom_point() + geom_smooth()


