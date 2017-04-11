library(jpeg)

setwd(getwd())

goby <- readJPEG("images/RobertMixed03.jpg")
strlitzia <- readJPEG("images/smallstrelitzia.jpg")
sunset <- readJPEG("images/smallsunset.jpg")

em <- function(img, clusters){ 
  height <- dim(img)[1]
  width <- dim(img)[2]
  
  # Create an empty matrix with dimension (width*hight) X 3
  img_2d <- matrix(0,height*width,3)
  for(i in seq(height)){
    for(j in seq(width)){
      img_2d[((i-1)*width)+j,] <- img[i,j,]
    }
  }
  
  # scale() will calculate the mean and std of the vector and then normalize the vector.
  img_2d <- scale(img_2d)

  pis <- matrix(1/clusters,1,clusters)
  # Genrate random compositions with a uniform distribution.
  random_vals <- runif(3*clusters)
  means <- matrix(random_vals, nrow=clusters)
  
  #EM steps 
  stop_criteria <- 0.000001
  old_Q <- 0
  flag <- 0
  while(TRUE){
    #E Step
    inner <- matrix(0,height*width, clusters)
    for(i in seq(clusters)){
      dist <- t(t(img_2d)-means[i,])
      inner[,i] <- (-.5) * rowSums(dist^2)
    }
    #calculate wijs
    top <- exp(inner) %*% diag(pis[1:clusters])
    bottom <- rowSums(top)
    wijs <- top/bottom
    #calculate Q
    Q <- sum(inner*wijs)
    
    #M step
    for(j in seq(clusters)){
      # smoothing_constant
      top <- colSums(img_2d * wijs[,j])
      bottom <- sum(wijs[,j]) 
      means[j,] <-top/bottom
      
      #update pis
      pis[j] <- sum(wijs[,j]) / height*width
    }
    
    #stopping rule
    if(flag == 0){
    	flag <- 1
    } else{
    	if(Q - old_Q < stop_criteria){
    		break
    	}
    }
    old_Q <- Q    
  }
  
  # Construct return image
  ans <- array(0,c(height, width,3))
  for(i in seq(height)){
    for(j in seq(width)){
      index <- (i-1)*width + j
      mean_segment <- which(wijs[index,] == max(wijs[index,]))
      ans[i,j,] <- means[mean_segment,]*attr(img_2d, 'scaled:scale') + attr(img_2d, 'scaled:center')
    }
  }
  return(ans)
}

writeJPEG(em(goby,10), "results/goby_10.jpg",quality = 1)
writeJPEG(em(goby,20), "results/goby_20.jpg",quality = 1)
writeJPEG(em(goby,50), "results/goby_50.jpg",quality = 1)

writeJPEG(em(strlitzia,10), "results/strlitzia_10.jpg",quality = 1)
writeJPEG(em(strlitzia,20), "results/strlitzia_20.jpg",quality = 1)
writeJPEG(em(strlitzia,50), "results/strlitzia_50.jpg",quality = 1)

writeJPEG(em(sunset,10), "results/sunset_10.jpg",quality = 1)
writeJPEG(em(sunset,20), "results/sunset_20.jpg",quality = 1)
writeJPEG(em(sunset,50), "results/sunset_50.jpg",quality = 1)

# Special test image
for(i in seq(5)){
	writeJPEG(em(sunset,20), paste("results/sunset_20", "_", i, ".jpg", sep=""), quality = 1)
}

