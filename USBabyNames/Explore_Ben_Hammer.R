
### Notice
# This is NOT my work !!
# This is copy from https://www.kaggle.com/benhamner/d/kaggle/us-baby-names/exploring-the-us-baby-names-data
# I did not upload data file because it is to huge to upload

library(DBI)
library(RSQLite)
library(dplyr)
library(ggvis)

# First, let's see what tables we have to work with.
db <- dbConnect(dbDriver("SQLite"), "./data/database.sqlite")
tables <- dbGetQuery(db, "select name from sqlite_master where type='table'")
colnames(tables) <- c("TableName")
tables <- tables %>%
  rowwise() %>% 
  mutate(RowCount=dbGetQuery(db,paste0("select count(id) RowCount from ", TableName))$RowCount[1])
print.table(tables)

# As we see above, we have two tables: NationalNames and StateNames. 
# Let’s look at the corresponding schemas.
print.table(dbGetQuery(db,"PRAGMA table_info('NationalNames')")[c("name","type")])
print.table(dbGetQuery(db,"PRAGMA table_info('StateNames')")[c("name","type")])

# what years does this dataset cover?
# how many babies are represented in this dataset?
yearCounts <- dbGetQuery(db,"select year,sum(Count) NumBabies from NationalNames group by Year order by Year")
yearCounts %>%
  ggvis(~Year, ~NumBabies) %>%
  layer_lines(stroke:="#20beff",strokeWidth:=3) %>%
  add_axis("y", title="Number of Babies", title_offset=80)

# what are the most popular baby names in this yeae(2014)?
print.table(dbGetQuery(db,"
SELECT * FROM NationalNames
WHERE Year=2015
ORDER BY Count desc
LIMIT 10
"))


# what are the most popular baby names in this yeae(2014) in states?
commonGirlNames <- dbGetQuery(
  db, "
  WITH CommonGirlCounts AS (
    SELECT State,
            MAX(Count) Count,
            SUM(Count) NumBabies
    FROM StateNames
    WHERE GENDER='F'
    AND Year=2014
    GROUP BY State
  )
  SELECT s.State,
         s.Name MostCommonGirlName,
         ROUND(100.0*s.Count/g.NumBabies,1) PercentGirlsWithName
  FROM StateNames s,
       CommonGirlCounts g
  WHERE s.State=g.State
        AND s.Count=g.Count
        AND s.Gender='F'
        AND s.Year=2014"
)

commonBoyNames <- dbGetQuery(
  db, "
  WITH CommonBoyCounts AS (
    SELECT State,
      MAX(Count) Count,
      SUM(Count) NumBabies
    FROM StateNames
    WHERE GENDER='M'
      AND Year=2014
    GROUP BY State
  )
  SELECT s.State,
    s.Name MostCommonBoyName,
    ROUND(100.0*s.Count/b.NumBabies,1) PercentBoysWithName
  FROM StateNames s,
    CommonBoyCounts b
  WHERE s.State=b.State
    AND s.Count=b.Count
    AND s.Gender='M'
    AND s.Year=2014"
)

print.table(merge(commonGirlNames, commonBoyNames, by="State"))


