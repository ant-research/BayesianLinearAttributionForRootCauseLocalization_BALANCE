library('fsMTS')

data <- scale(read.csv('Xy.csv')[,-1], center = FALSE, scale = FALSE)
p <- ncol(data)
max_lag = 3

mIndep<-fsMTS(data, max.lag=max_lag, method="ownlags")
mCCF<-fsMTS(data, max.lag=max_lag, method="CCF")
# mDistance<-fsMTS(data, max.lag=max_lag, method="distance", shortest = traffic.mini$shortest, step = 5)
mGLASSO<-fsMTS(data, max.lag=max_lag,method="GLASSO", rho = 0.5)
#mLARS<-fsMTS(data, max.lag=max_lag,method="LARS")
mRF<-fsMTS(data, max.lag=max_lag,method="RF")
# mMI<-fsMTS(data, max.lag=max_lag,method="MI")
mlist <- list(Independent = mIndep,
              # Distance = mDistance,
              CCF = mCCF,
              GLASSO = mGLASSO,
              # LARS = mLARS,
              RF = mRF) #,MI = mMI)

thr_start = 0
thr_end = 1
max_iter = 1000
for (iter in 1:max_iter) {
  thr <- (thr_start + thr_end) / 2
  res <- fsEnsemble(mlist, threshold = thr, method="ranking")
#   res <- fsEnsemble(mlist, threshold = thr, method="majority")
  max_res <- res[1:(p - 1), p - 1]
  for (i in 2:max_lag) {
    max_res <- max_res + res[(p * (i - 1) + 1) : (p * i - 1), p]
  }
  if (sum(max_res > 0) > 3) {
    thr_end <- thr
  } else if (sum(max_res > 0) < 3) {
    thr_start <- thr
  } else {
    break
  }
}

write.csv(max_res, file ="coef.csv", row.names=FALSE)


