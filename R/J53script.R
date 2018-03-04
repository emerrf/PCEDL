library(kernplus)
library(dplyr)

### Quick dataset check
# head(windpw)
# summary(windpw)
# plot(y ~ V, data = windpw)
# hist(windpw$y[windpw$V < 5])

rm(list=ls())


### J53 Data reader

read_J53_data <- function(path) {

  wt_types <- c("WT1"="shore", "WT2"="shore", "WT3"="shore", "WT4"="shore",
                "WT5"="offshore", "WT6"="offshore")

  J53_varnames <- c("Sequence.No."="seqNo",
                    "V"="V",
                    "D"="D",
                    "air.density"="rho",
                    "Humidity"="H",
                    "I"="I",
                    "S_a"="Sa",
                    "S_b"="Sb",
                    "y....relative.to.rated.power."="y"
  )

  J53_data_list <- list()
  for (fname in list.files(path, full.names = TRUE)) {
    cat(paste("Loading:", fname, "\n"))
    fdata <- read.table(fname, header = TRUE, stringsAsFactors = FALSE)
    colnames(fdata) <- J53_varnames[colnames(fdata)]
    split_path <- strsplit(fname, "\\.|_")[[1]]
    id <- split_path[grep("WT", split_path)]
    fdata$WT <- id
    fdata$Type <- wt_types[id]

    J53_data_list[[fname]] <- fdata
  }

  J53_data <- dplyr::bind_rows(J53_data_list)
  J53_data$WT <- factor(J53_data$WT)
  J53_data$Type <- factor(J53_data$Type)

  return(J53_data)
}


setwd("C:/Users/emerrf/Documents/GitHub/PCEDL_local/R")
dirpath <- "../PCEDL/data/J53/"
J53_data <- read_J53_data(dirpath)

# WT1 Dataset
WT1_data <- subset(J53_data, subset = WT == "WT1", select = -c(H, Sa, WT, Type))
head(WT1_data)
plot(y ~ V, data = WT1_data)

# Randomise
set.seed(1)
WT1_data <- WT1_data[sample(nrow(WT1_data)),]
#write.csv(WT1_data["seqNo"], file="WT1_seqNo_order.csv", row.names = FALSE)
train_size <- floor(nrow(WT1_data) * 0.9)  # 42787
test_size <- nrow(WT1_data) - train_size # 4755

df.tr <- WT1_data[1:train_size, ]
df.ts <- WT1_data[(train_size+1):(train_size+test_size), ]
id.cov <- c('V', 'D', 'rho', 'I', 'Sb')
pred <- kp.pwcurv(df.tr$y, df.tr[, id.cov], df.ts[, id.cov], id.dir = 2)
rmse <- sqrt(mean((df.ts$y - pred)^2))
mse = rmse^2
# Assuming Vestas V80/1800 with rated power of 1.8MW
rmse_paper = rmse/100*1800 # W
cat(paste("MSE:", mse, "RMSE:", rmse, "RMSE Paper:", rmse_paper))

# KFold cross validation
WT1_data <- subset(J53_data, subset = WT == "WT1", select = -c(H, Sa, WT, Type))
k <- 10
rmse_vec <- vector(length=k)
for(i in 1:k){
  set.seed(i+1000)
  WT1_data <- WT1_data[sample(nrow(WT1_data)),]
  train_size <- floor(nrow(WT1_data) * 0.9)
  test_size <- nrow(WT1_data) - train_size
  df.tr <- WT1_data[1:train_size, ]
  df.ts <- WT1_data[(train_size+1):(train_size+test_size), ]

  id.cov <- c('V', 'D', 'rho', 'I', 'Sb')
  pred <- kp.pwcurv(df.tr$y, df.tr[, id.cov], df.ts[, id.cov], id.dir = 2)

  mse <- mean((df.ts$y - pred)^2)
  rmse <- sqrt(mse)
  cat(paste("Iter:", i, "MSE:", mse, "RMSE:", rmse))
  cat("\n")
  rmse_vec[i] <- rmse
}
cat(rmse_vec)
cat(mean(rmse_vec))  # 7.432424
cat(sd(rmse_vec))    # 0.1190444

## WINDPW

# Suppose only 90% of data are available and use the rest 10% for prediction.
use_zscores = FALSE

df.tr <- windpw[1:900, ]
df.ts <- windpw[901:1000, ]
id.cov <- c('V', 'D', 'rho', 'I', 'Sb')

if(use_zscores){
  # df.tr <- as.data.frame(scale(df.tr, scale=TRUE))
  # df.ts <- as.data.frame(scale(df.ts, scale=TRUE))
  df.tr <- cbind(as.data.frame(scale(df.tr[, id.cov], scale=TRUE)), df.tr[,"y"])
  df.ts <- cbind(as.data.frame(scale(df.ts[, id.cov], scale=TRUE)), df.ts[,"y"])
}

pred <- kp.pwcurv(df.tr$y, df.tr[, id.cov], df.ts[, id.cov], id.dir = 2)
sqrt(mean((df.ts$y - pred)^2))
plot(pred, df.ts$y - pred)

# Train
pred <- kp.pwcurv(df.tr$y, df.tr[, id.cov], df.tr[, id.cov], id.dir = 2)
sqrt(mean((df.tr$y - pred)^2))


plot(pred, df.ts$y)
abline(lm(df.ts$y~pred))

predy <- (pred*sd(windpw[901:1000, "y"]))+mean(windpw[901:1000, "y"])
sqrt(mean((windpw[901:1000, "y"]-predy)^2))
