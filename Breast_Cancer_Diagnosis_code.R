#packages
# install.packages('rmarkdown') 
# install.packages("NbClust")
# install.packages("cluster")
# install.packages("fpc")


#read the csv file
library(readr)
data <- read.csv("C:/Users/bella/Desktop/Breast_cancer_data.csv", 
                   sep=";" )

#fast analysis
summary(data) #general informations
str( data )
dim( data) #we have 569 records and 32 variables
colnames(data) #diagnosis is M or B, the others are continuous

attach(data) 

#analysis of histograms of non-normalized variables
#breaks = sqrt( length( radius_mean ) )

hist( radius_mean, breaks = sqrt( length( radius_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Radius mean', xlab = 'Radius mean' ) 

hist( texture_mean, breaks = sqrt( length( texture_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Texture mean', xlab = 'Texture mean' )

hist( perimeter_mean, breaks = sqrt( length( perimeter_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Perimeter mean', xlab = 'Perimeter mean' )

hist( area_mean, breaks = sqrt( length( area_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Area mean', xlab = 'Area mean' )

hist( smoothness_mean, breaks = sqrt( length( smoothness_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Smoothness mean', xlab = 'Smoothness mean' )

hist( compactness_mean, breaks = sqrt( length( compactness_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Compactness mean', xlab = 'Compactness mean' )

hist( concavity_mean, breaks = sqrt( length( concavity_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Concavity mean', xlab = 'Concavity mean' )

hist( concave_points_mean, breaks = sqrt( length( concave_points_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Concave points mean', xlab = 'Concave points mean' )

hist( symmetry_mean, breaks = sqrt( length( symmetry_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Symmetry mean', xlab = 'Symmetry mean' )

hist( fractal_dimension_mean, breaks = sqrt( length( fractal_dimension_mean ) ), probability = TRUE,
      col = 'lavender', main = 'Fractal dimension mean', xlab = 'Fractal dimension mean' )

hist( radius_se, breaks = sqrt( length( radius_se ) ), probability = TRUE,
      col = 'lavender', main = 'Radius_se', xlab = 'Radius_se' ) #standard error


hist( texture_se , breaks = sqrt( length( texture_se) ), probability = TRUE,
      col = 'lavender', main = 'Texture_se', xlab = 'Texture_se')

hist( perimeter_se , breaks = sqrt( length( perimeter_se ) ), probability = TRUE,
      col = 'lavender', main = 'Perimeter_se', xlab = 'Perimeter_se')

hist( area_se , breaks = sqrt( length( area_se ) ), probability = TRUE,
      col = 'lavender', main = 'Area_se', xlab = 'Area_se')

hist( smoothness_se , breaks = sqrt( length( smoothness_se ) ), probability = TRUE,
      col = 'lavender', main = 'Smoothness_se', xlab = 'Smoothness_se')

hist( compactness_se , breaks = sqrt( length( compactness_se ) ), probability = TRUE,
      col = 'lavender', main = 'Compactness_se', xlab = 'Compactness_se')

hist( concavity_se , breaks = sqrt( length( concavity_se ) ), probability = TRUE,
      col = 'lavender', main = 'Concavity_se', xlab = 'Concavity_se')

hist( concave_points_se , breaks = sqrt( length( concave_points_se ) ), probability = TRUE,
      col = 'lavender', main = 'Concave points_se', xlab = 'Concave points_se')

hist( symmetry_se , breaks = sqrt( length( symmetry_se ) ), probability = TRUE,
      col = 'lavender', main = 'Symmetry_se', xlab = 'Symmetry_se')

hist( fractal_dimension_se , breaks = sqrt( length( fractal_dimension_se ) ), probability = TRUE,
      col = 'lavender', main = 'Fractal dimension_se', xlab = 'Fractal dimension_se')

hist( radius_worst , breaks = sqrt( length( radius_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Radius worst', xlab = 'Radius worst')

hist( texture_worst , breaks = sqrt( length( texture_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Texture worst', xlab = 'Texture worst')

hist( perimeter_worst , breaks = sqrt( length( perimeter_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Perimeter worst', xlab = 'Perimeter worst')

hist( area_worst, breaks = sqrt( length( area_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Area worst', xlab = 'Area worst')

hist( smoothness_worst , breaks = sqrt( length( smoothness_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Smoothness worst', xlab = 'Smoothness worst')

hist( compactness_worst , breaks = sqrt( length( compactness_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Compactness worst', xlab = 'Compactness worst')

hist( concavity_worst , breaks = sqrt( length( concavity_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Concavity worst', xlab = 'Concavity worst')

hist( radius_worst , breaks = sqrt( length( radius_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Radius worst', xlab = 'Radius worst')

hist( concave_points_worst , breaks = sqrt( length( concave_points_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Concave points worst', xlab = 'Concave points worst')

hist( symmetry_worst , breaks = sqrt( length( symmetry_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Symmetry worst', xlab = 'Symmetry worst')

hist( fractal_dimension_worst , breaks = sqrt( length( fractal_dimension_worst ) ), probability = TRUE,
      col = 'lavender', main = 'Fractal dimension worst', xlab = 'Fractal dimension worst')


x=cbind(id, radius_mean    ,       
    texture_mean  ,       perimeter_mean ,         area_mean ,          
  smoothness_mean,        compactness_mean,       concavity_mean  ,      
     concave_points_mean ,   symmetry_mean     ,    fractal_dimension_mean,
    radius_se   ,           texture_se ,            perimeter_se ,         
  area_se   ,             smoothness_se  ,         compactness_se ,      
    concavity_se ,       concave_points_se   ,    symmetry_se     ,     
    fractal_dimension_se,   radius_worst      ,      texture_worst,          
    perimeter_worst   ,     area_worst     ,         smoothness_worst ,     
    compactness_worst    ,  concavity_worst  ,       concave_points_worst   ,
    symmetry_worst     ,   fractal_dimension_worst) #without diagnosis that is categorical, but with id

V=cor(x, method="pearson") #Pearson coefficients of correlation, square and sym positive matrix
V 
#see the matrix in the V in Data

Vwithoutdiag= V-diag(V) #to delete the 1 in the diagonal and after apply the filter

##CORRELATION HEAT MAP
library(dplyr)
cor_var <- subset(Breast_cancer_data, select=(-c(id,diagnosis)))
library(ggplot2)
library(reshape2)
corvar2 <- round(cor(cor_var),2)

#Reorder
reorder_corvar2 <- function(corvar2){
  # Use correlation between variables as distance
  dd <- as.dist((1-corvar2)/2)
  hc <- hclust(dd)
  corvar2 <-corvar2[hc$order, hc$order]
}
corvar2 <- reorder_corvar2(corvar2)

#get only the lower part
get_upper_tri<-function(corvar2){
  corvar2[lower.tri(corvar2)] <- NA
  return(corvar2)
}

#PLOT FOR ORDERED COLOURS
upper_tri <- get_upper_tri(corvar2)
melted_corvar <- melt(upper_tri, na.rm = TRUE)
ggheatmap <- ggplot(melted_corvar, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
# Print the heatmap
print(ggheatmap)

##PUT NUMBER IN THE GRAPH
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 2.5) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

plot(texture_mean, texture_worst) #0.912044589 coeff of correlation, we can see they are high linear correlated


###BOXPLOTS###

boxplot (radius_mean, texture_mean, perimeter_mean,area_mean,smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, 
         names = c("Radius mean", "Texture mean", "Perimeter mean", "Area mean", "Smoothness mean", "Compactness mean", "Concavity mean","Concave point mean", "Symmetry mean",
                   "Fractal dimension mean"))

##NORMALIZED MEAN
boxplot (normalized_radius_mean, normalized_texture_mean, normalized_perimeter_mean,normalized_area_mean,normalized_smoothness_mean, normalized_compactness_mean, normalized_concavity_mean, normalized_concave_points_mean, normalized_symmetry_mean, normalized_fractal_dimension_mean, 
         names = c("Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity","Concave point", "Symmetry",
                   "Fractal dimension"), main = "Mean Boxplot")
##NORMALIZED SE
boxplot (normalized_radius_se, normalized_texture_se, normalized_perimeter_se,normalized_area_se,normalized_smoothness_se, normalized_compactness_se, normalized_concavity_se, normalized_concave_points_se, normalized_symmetry_se, normalized_fractal_dimension_se, 
         names = c("Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity","Concave point", "Symmetry",
                   "Fractal dimension"), main = "Se Boxplot")
##NORMALIZED WORST
boxplot (normalized_radius_worst, normalized_texture_worst, normalized_perimeter_worst,normalized_area_worst,normalized_smoothness_worst, normalized_compactness_worst, normalized_concavity_worst, normalized_concave_points_worst, normalized_symmetry_worst, normalized_fractal_dimension_worst, 
         names = c("Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity","Concave point", "Symmetry",
                   "Fractal dimension"), main = "worst Boxplot")

##NORMALIZATION USING Z-SCORE####
####Var <- (myVar - mean(myVar)) / sd(myVar)
#MEAN
normalized_radius_mean <- (radius_mean - mean (radius_mean))/(sd(radius_mean))
normalized_texture_mean <- (texture_mean - mean (texture_mean))/sd (texture_mean)
normalized_perimeter_mean <- (perimeter_mean - mean (perimeter_mean))/sd (perimeter_mean)
normalized_area_mean <- (area_mean - mean (area_mean))/sd (area_mean)
normalized_smoothness_mean <- (smoothness_mean - mean (smoothness_mean))/sd (smoothness_mean)
normalized_compactness_mean <- (compactness_mean - mean (compactness_mean))/sd (compactness_mean)
normalized_concavity_mean <- (concavity_mean - mean (concavity_mean))/sd (concavity_mean) 
normalized_concave_points_mean <- (concave_points_mean - mean (concave_points_mean))/sd (concave_points_mean) 
normalized_symmetry_mean <- (symmetry_mean - mean (symmetry_mean))/sd (symmetry_mean) 
normalized_fractal_dimension_mean <- (fractal_dimension_mean - mean (fractal_dimension_mean))/sd (fractal_dimension_mean) 

#SE
normalized_radius_se <- (radius_se - mean (radius_se))/(sd(radius_se))
normalized_texture_se <- (texture_se - mean (texture_se))/sd (texture_se)  
normalized_perimeter_se <- (perimeter_se - mean (perimeter_se))/sd (perimeter_se) 
normalized_area_se <- (area_se - mean (area_se))/sd (area_se) 
normalized_smoothness_se <- (smoothness_se - mean (smoothness_se))/sd (smoothness_se) 
normalized_compactness_se <- (compactness_se - mean (compactness_se))/sd (compactness_se) 
normalized_concavity_se <- (concavity_se - mean (concavity_se))/sd (concavity_se) 
normalized_concave_points_se <- (concave_points_se - mean (concave_points_se))/sd (concave_points_se) 
normalized_symmetry_se <- (symmetry_se - mean (symmetry_se))/sd (symmetry_se) 
normalized_fractal_dimension_se <- (fractal_dimension_se - mean (fractal_dimension_se))/sd (fractal_dimension_se) 

#WORST
normalized_radius_worst = (radius_worst - mean (radius_worst))/(sd(radius_worst))
normalized_texture_worst = (texture_worst - mean (texture_worst))/sd (texture_worst)  
normalized_perimeter_worst = (perimeter_worst - mean (perimeter_worst))/sd (perimeter_worst) 
normalized_area_worst = (area_worst - mean (area_worst))/sd (area_worst) 
normalized_smoothness_worst = (smoothness_worst - mean (smoothness_worst))/sd (smoothness_worst) 
normalized_compactness_worst = (compactness_worst - mean (compactness_worst))/sd (compactness_worst) 
normalized_concavity_worst = (concavity_worst - mean (concavity_worst))/sd (concavity_worst) 
normalized_concave_points_worst = (concave_points_worst - mean (concave_points_worst))/sd (concave_points_worst)
normalized_symmetry_worst = (symmetry_worst - mean (symmetry_worst))/sd (symmetry_worst)
normalized_fractal_dimension_worst = (fractal_dimension_worst - mean (fractal_dimension_worst))/sd (fractal_dimension_worst)

### ANALIZE OF SHAPIRO TO CHECK FOR NORMAL DISTRIBUTION FOR THE NORMALIZED VARIABLES ###
shapiro.test(normalized_radius_mean)
shapiro.test(normalized_radius_se)
shapiro.test(normalized_radius_worst)
shapiro.test(normalized_texture_mean)
shapiro.test(normalized_texture_se)
shapiro.test(normalized_texture_worst)
shapiro.test(normalized_perimeter_mean)
shapiro.test(normalized_perimeter_se)
shapiro.test(normalized_perimeter_worst)
shapiro.test(normalized_area_mean)
shapiro.test(normalized_area_se)
shapiro.test(normalized_area_worst)
shapiro.test(normalized_smoothness_mean)
shapiro.test(normalized_smoothness_se)
shapiro.test(normalized_smoothness_worst)
shapiro.test(normalized_compactness_mean)
shapiro.test(normalized_compactness_se)
shapiro.test(normalized_compactness_worst)
shapiro.test(normalized_concavity_mean)
shapiro.test(normalized_concavity_se)
shapiro.test(normalized_concavity_worst)
shapiro.test(normalized_concave_points_mean)
shapiro.test(normalized_concave_points_se)
shapiro.test(normalized_concave_points_worst)
shapiro.test(normalized_symmetry_mean)
shapiro.test(normalized_symmetry_se)
shapiro.test(normalized_symmetry_worst)
shapiro.test(normalized_fractal_dimension_mean)
shapiro.test(normalized_fractal_dimension_se)
shapiro.test(normalized_fractal_dimension_worst)

xnorm=cbind(normalized_radius_mean,normalized_texture_mean,normalized_perimeter_mean,normalized_area_mean, normalized_smoothness_mean, normalized_compactness_mean,
            normalized_concavity_mean, normalized_concave_points_mean, normalized_symmetry_mean, normalized_fractal_dimension_mean, normalized_radius_se,
            normalized_texture_se,normalized_perimeter_se,normalized_area_se,normalized_smoothness_se,normalized_compactness_se,
            normalized_concavity_se,normalized_concave_points_se,normalized_symmetry_se,normalized_fractal_dimension_se,
            normalized_radius_worst,normalized_texture_worst,normalized_perimeter_worst,normalized_area_worst,
            normalized_smoothness_worst,normalized_compactness_worst,normalized_concavity_worst,
            normalized_concave_points_worst, normalized_symmetry_worst,normalized_fractal_dimension_worst
) #without id and diagnosis, 30 variables

#apply pca
xnorm.pca <- princomp(xnorm, scores=TRUE,cor=TRUE)
xnorm.pca
plot(xnorm.pca, type = "l") #the x axis stop at comp 10 because it has (95%) of variability, we chose 90%. Thus, 7 principal components
summary(xnorm.pca)
xnorm.pca$loadings
xnorm.pca$center
xnorm.pca$scale
xnorm.pca$n.obs
mat=xnorm.pca$scores
mat
matcomp7=mat[, 1:7] #matrix with 569 rows and 7 columns

#clustering
#number of clusters

library(NbClust)
library(cluster)

set.seed(2365)
# Compute and plot wss for k = 2 to k = 15 ---> Elbow plot
k.max <- 15 # Maximal number of clusters
wss <- sapply(1:k.max, 
              function(k){kmeans(matcomp7, k, nstart=10 )$tot.withinss})

plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")
abline(v = 3, lty =2)

#another way of Elbow plot, but use Elbow plot
km=NbClust(matcomp7, diss = NULL, distance = "euclidean",
           min.nc = 2, max.nc =15,
           method = "kmeans", index = "all", alphaBeale = 0.1)
#kmeans

#The Hartigan-Wong algorithm generally does a better job 
#trying nstart different, >1 is often suggested
clk=kmeans(matcomp7, 3, iter.max = 100, nstart = 1,    
           algorithm = c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"),  trace=FALSE)  #change everytime , between two, wtf

clk=kmeans(matcomp7, 3, iter.max = 100, nstart =2365 ,    
           algorithm = c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"),  trace=FALSE)

clk

bandm= table(diagnosis,clk$cluster )
bandm


tab1=matrix(c(0,0,0,37,321,321+37, 37,321, 321+37), ncol=3, byrow=TRUE)
colnames(tab1)=c("Real Benign", "Real Malign", " ")
rownames(tab1)=c("Cluster Benign", "Cluster Malign", " ")
tab1

#corr_miss<-(tab1[1,1]+tab1[2,2])/(tab1[1,1]+tab1[1,2]+tab1[2,1]+tab1[2,2])
#err_miss <- 1-corr_miss
sens<-tab1[1,1]/(tab1[1,1]+tab1[2,1])
sens

spec=tab1[2,2]/(tab1[1,2]+tab1[2,2]) 
spec


tab2=matrix(c(65,36,36+65,0,0,0, 65,36,36+65) , ncol=3, byrow=TRUE)   # B=36, M=65
colnames(tab2)=c("Real Malign", "Real Benign", " ")
rownames(tab2)=c("Cluster Malign", "Cluster Benign", " ")
tab2

sens<-tab2[1,1]/(tab2[1,1]+tab2[2,1]) #ok
sens

spec=tab2[2,2]/(tab2[1,2]+tab2[2,2]) #ok
spec

tab3=matrix(c(110,0,110,0,0,0, 110,0,0) , ncol=3, byrow=TRUE)
colnames(tab3)=c("Real Benign", "Real Malign", " ")
rownames(tab3)=c("Cluster Benign", "Cluster Malign", " ")
tab3

sens<-tab3[1,1]/(tab3[1,1]+tab3[2,1])
sens

spec=tab3[2,2]/(tab3[1,2]+tab3[2,2]) 
spec


tabtot=matrix(c(65+110,36,36+65+110,37,321,321+37, 65+110+37,36+321,65+110+37+36+321) , ncol=3, byrow=TRUE)   
colnames(tabtot)=c("Real Malign", "Real Benign", " ")
rownames(tabtot)=c("Cluster Malign", "Cluster Benign", " ")
tabtot

sens<-tabtot[1,1]/(tabtot[1,1]+tabtot[2,1])
sens

spec=tabtot[2,2]/(tabtot[1,2]+tabtot[2,2]) 
spec

prevalence=(tabtot[1,1]+tabtot[2,1])/ (tabtot[1,1]+tabtot[2,1]+tabtot[1,2]+tabtot[2,2])
prevalence #malign are lower than benign
1-prevalence

accuracy=(tabtot[1,1]+tabtot[2,2])/(tabtot[1,1]+tabtot[2,1]+tabtot[1,2]+tabtot[2,2])
accuracy
1-accuracy  #=errtot


#error calcolation

clus1err= 37/(321+37)

clus2err= 36/(65+36)

clus3err=0


errtot= (((321+37)*clus1err)+((65+36)*clus2err))/569
errtot #12%


#kmedoids
library(fpc)
clm=pamk(matcomp7, krange=2:10, criterion="asw", usepam=TRUE,
         scaling=FALSE, alpha=0.001, diss=inherits(data, "dist"),
         critout=FALSE, ns=10, seed=NULL)  #check if 2 is not a parameter, anyway it is perfect

clm #we have the centroids values
#visualization
library(cluster)
plot(pam(matcomp7, clm$nc))


bandmmedoids= table(diagnosis,clm$pamobject$clustering )
bandmmedoids
#calculate spec and sens
#  M     B    true
#M 169  43
#B 6    351
#model

specmedoids=351/(321+43)
specmedoids
sensmedoids=169/(169+6)
sensmedoids

#errors
err1=6/(6+169)
err2=43/(351+43)
errf=(err1*(6+169)+err2*(351+43))/569 #perfeito 8% 1-accuracy


#transform them into the initial variables


#importance of variables
#calculate, for example, the variance of the centroids for every dimension. The dimensions with the highest variance are most important in distinguishing the clusters.


centroids=clm$pamobject$medoids
xnorm.pca <- princomp(xnorm, scores=TRUE,cor=TRUE)
xnorm.pca
summary(xnorm.pca)
xnorm.pca$loadings
xnorm.pca$center
xnorm.pca$scale
xnorm.pca$n.obs
mat=xnorm.pca$scores
mat
matmatcomp7=mat[,1:7]
matmatcomp7
var1=matmatcomp7[1,]%*%centroids[1,]
var2=matmatcomp7[2,]%*%centroids[1,]
var3=matmatcomp7[3,]%*%centroids[1,]
var4=matmatcomp7[4,]%*%centroids[1,] #ecc until 33 for the cluster 1

var1=matmatcomp7[1,]%*%centroids[2,] #for the cluster 2
