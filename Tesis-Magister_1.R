rm(list=ls())
setwd('C:/Codigo_Tesis_Magister/Referencia')
source('PFuncion.R')
library(tidyverse)
library(imputeTS)
library(data.table)
library(berryFunctions)

# Puntos de referencia de la placa de Nazca
Nasca<-read.csv('Nazcapts.txt',sep=' ',header = F)
colnames(Nasca)<-c('Lat','Lon')
Nasca<-Nasca[Nasca$Lat>-49,]

# Catalogo de eventos sismicos
Catearth<-read.csv('data_earthquake_2000-2017.csv',sep=";")
colnames(Catearth)[c(8,9)]<-c("Lat","Lon")

Catearth<-Catearth[Catearth$year>=2007,]
Catearth<-Catearth[Catearth$Lon>=-80,]
Catearth<-Catearth[Catearth$Lat>=-45,]

Lonp<-ProyNazc(Catearth,Nasca)
Catearth<-(cbind(Catearth,Lonp))
HDist<-Harvdist(Catearth)

Catearth<-(cbind(Catearth,HDist))

# Prepara lista de nombres para ingresarlos a un data.frame

grdecil<-list.files('C:/Codigo_Tesis_Magister/Referencia/GDChile')
grdecil<-grdecil[!grdecil%in% 'CUVIgd.csv']# Se extrae estacion CUVI
grdecil<-grdecil[!grdecil%in% 'HLNEgd.csv']# Se extrae estacion HLNE

setwd("C:/Codigo_Tesis_Magister/Referencia/GDChile")
grdecil<-str_sub(grdecil,1,4)

for(i in grdecil){ # Ingresa series de tiempo de las estaciones de GPS en grados decimales
  assign(i,read.csv(paste0(i,"gd.csv"),sep=","))
  assign(i,data.frame(as.Date(get(i)$Fecha),get(i)$Lon,get(i)$Lat))
  assign(i, `colnames<-`(get(i), c("Fecha","Lon","Lat")))            # Cambia nombres
  
}

# La 'data_actualizada.csv' contiene coordenadas de referencia para todas 
# las estaciones de GPS del mundo

setwd('C:/Codigo_Tesis_Magister/Referencia')
catGPS<-read.csv("data_actualizada.csv",header = F,sep=";")

colnames(catGPS)<-c('Estacion','Lat','Lon','Sis','X','Ciudad','Pais')

catGPS<-catGPS[!catGPS$Estacion%in%'CUVI',] # Extraigo CUVI
catGPS<-catGPS[!catGPS$Estacion%in%'HLNE',] # Extraigo HLNE
Gpchile<-data.frame()

# Filtra conjunto de datos para solo estaciones de Chile 
for(i in 1:nrow(catGPS)){
  ifelse(catGPS$Pais[i]==' Chile',Gpchile<- rbind(Gpchile,catGPS[i,]),FALSE)
}

rm(catGPS)

Gpchile<-Gpchile[Gpchile$Lat>-45,]
Gpchile<-Gpchile[Gpchile$Lon>-80,]

# Se genera df para visualizar fechas de inicio y culmino de cada estacion
# con su proporcion de valores faltantes por longitud y latitud
DesData<-data.frame()
for(i in 1:nrow(Gpchile)){
    DesData<-rbind(DesData,data.frame(as.character(Gpchile$Estacion[i]),
                                    Gpchile$Lon[i],
                                    Gpchile$Lat[i],
                                    get(as.character(Gpchile$Estacion[i]))$Fecha[1],
                                    get(as.character(Gpchile$Estacion[i]))$Fecha[nrow(get(as.character(Gpchile$Estacion[i])))],
                                    (sum(is.na(get(as.character(Gpchile$Estacion[i]))$Lon))/nrow(get(as.character(Gpchile$Estacion[i])))),
                                    (sum(is.na(get(as.character(Gpchile$Estacion[i]))$Lat))/nrow(get(as.character(Gpchile$Estacion[i]))))
  ))
}

colnames(DesData)<-c('Estacion','Lon','Lat','FechaI','FechaF','LonPmiss','LatPmiss')
DesData<-DesData[order(-DesData$LonPmiss),]
DesData$Estacion<-as.character(DesData$Estacion)
rownames(DesData)<-NULL

# Muestra Top 5 De estaciones mas cercanas a la de referencia 

Rankesta("PB07",DesData)
# Estima valores faltantes a traves de na_interpolation 
# tanto para latitud y longitud

for(i in 1:nrow(DesData)){
  set(get(DesData$Estacion[i]), j = 'Lon', value = Reemplazo(get(DesData$Estacion[i]),1))
  set(get(DesData$Estacion[i]), j = 'Lat', value = Reemplazo(get(DesData$Estacion[i]),2))
}

# Quita el ruido a las series de tiempo

for(i in 1:nrow(DesData)){
  set(get(DesData$Estacion[i]), j = 'Lon', value = DnoisEst(get(DesData$Estacion[i])$Lon))
  set(get(DesData$Estacion[i]), j = 'Lat', value = DnoisEst(get(DesData$Estacion[i])$Lat))
}

DesData<-data.frame()
for(i in 1:nrow(Gpchile)){ 
  # Se genera df para visualizar fechas de inicio y culmino de cada estacion
  # con su proporcion de valores faltantes por longitud y latitud

  DesData<-rbind(DesData,data.frame(as.character(Gpchile$Estacion[i]),
                                    Gpchile$Lon[i],
                                    Gpchile$Lat[i],
                                    get(as.character(Gpchile$Estacion[i]))$Fecha[1],
                                    get(as.character(Gpchile$Estacion[i]))$Fecha[nrow(get(as.character(Gpchile$Estacion[i])))],
                                    (sum(is.na(get(as.character(Gpchile$Estacion[i]))$Lon))/nrow(get(as.character(Gpchile$Estacion[i])))),
                                    (sum(is.na(get(as.character(Gpchile$Estacion[i]))$Lat))/nrow(get(as.character(Gpchile$Estacion[i]))))
  ))
}

colnames(DesData)<-c('Estacion','Lon','Lat','FechaI','FechaF','LonPmiss','LatPmiss')
DesData$Estacion<-as.character(DesData$Estacion)
DesData$FechaI<-as.Date(DesData$FechaI)
DesData$FechaF<-as.Date(DesData$FechaF)

Fecha<-as.Date(paste0(Catearth$year,'-',Catearth$month,'-',Catearth$day))
Catearth<-cbind(Fecha,Catearth)

# CAV almacena catalogo final con los calculos

CAV<-data.frame()
dfna<-data.frame(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
colnames(dfna)<-c('Estacion','ELon','ELat','Angulo','Velocidad','HVQuk','Mt1D','Mt2D','Mt3D','Mt4D','Mt5D')

#Esto se demora, si quiere probar, ocupe menos datos

start.time <- Sys.time()
# Se obtiene estacion, angulo, velocidad y distancia con respecto al evento sismico

for(i in 1:nrow(Catearth)){
  if(!is.error(CatAngVel(DesData,Catearth[i,]))){
    CAV<-rbindlist(list(CAV,CatAngVel(DesData,Catearth[i,])))
  }else{
    CAV<-rbindlist(list(CAV,dfna))
  }
}
end.time <- Sys.time()
time.takenB1 <- end.time - start.time
time.takenB1

DFinal<-cbind(Catearth,CAV)
summary(DFinal)
write.csv(DFinal,'CatalogoVelAng2007-2017.csv',row.names = FALSE)

DFinal2<-read.csv('CatalogoVelAng2007-2017.csv')
names(DFinal2)