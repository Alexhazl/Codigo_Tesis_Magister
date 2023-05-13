rm(list=ls())
setwd('C:/Codigo_Tesis_Magister/Referencia')
library(ggplot2)
library(abind)
library(scales)
library(ggrepel)
source('PFuncion.R')

# Proceso de generar matrices para el entrenamiento (Intesidad y desplazamiento)

##############
# Intensidad #
##############

load('daily.intensity2010.2017.RData')

tot.mat2016.2017<-tot.mat2016.2017[(2:dim(tot.mat2016.2017)[1]),,]
tot.mat2010.2017 <- abind(tot.mat2010.2013, tot.mat2014.2015, tot.mat2016.2017, along= 1)
rm(tot.mat2010.2013,tot.mat2014.2015,tot.mat2016.2017)
matrices<-matrix(tot.mat2010.2017,nrow=2922*20,ncol=20)

##################
# Desplazamiento #
##################

Catearth<-read.csv('CatalogoVelAng2007-2017.csv')

Catearth<-Catearth[Catearth$Lat>= -38,]
Catearth<-Catearth[Catearth$Lat<= -31,] 
Catearth<-Catearth[Catearth$Lon>= -75,]
Catearth<-Catearth[Catearth$Lon<= -71,]
Catearth<-Catearth[Catearth$year>=2009,]
Catearth<-Catearth[Catearth$year<=2017,]
Catearth$Fecha<-as.Date(Catearth$Fecha)

Mmd<-GrillaDes(Catearth)
Mmd<-matrix(Mmd,nrow=2922*20,ncol=20)
summary(Mmd)
# Se guardan los datos Desplazamientos observados
write.table(Mmd, file='DEOB.txt', row.names=FALSE, col.names=FALSE)

# Se guardan intensidades estimadas
write.table(matrices, file='INOB.txt', row.names=FALSE, col.names=FALSE)
