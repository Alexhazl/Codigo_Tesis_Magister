rm(list=ls())
setwd('C:/Codigo_Tesis_Magister/Referencia')
library(ggrepel)
library(ggplot2)
library(scales)
library(rgdal)
source('PFuncion.R')


# Cargar catalogo sismico
Catearth<-read.csv('CatalogoVelAng2007-2017.csv')

Catearth<-Catearth[Catearth$Lat>= -38,]
Catearth<-Catearth[Catearth$Lat<= -31,] 
Catearth<-Catearth[Catearth$Lon>= -75,]
Catearth<-Catearth[Catearth$Lon<= -71,]
Catearth<-Catearth[Catearth$year>=2009,]
Catearth<-Catearth[Catearth$year<=2017,]
Catearth$Fecha<-as.Date(Catearth$Fecha)

# Se cargan los ShapeFile
setwd('C:/Codigo_Tesis_Magister/Referencia/Shapes')# Abro las delimitaciones
SHP1<-readOGR('CQM.shp')
SHP2<-readOGR('LBO.shp')
SHP3<-readOGR('RMT.shp')
SHP4<-readOGR('VLP.shp')
SHP5<-readOGR('BIO.shp')
SHP6<-readOGR('EML.shp')
SHP<-rbind(SHP1,SHP2,SHP3,SHP4,SHP5,SHP6) # Uno regiones

setwd('C:/Codigo_Tesis_Magister/Referencia')
# Cargar datos predicted vs observed

caso1<-c("ffnn_2_p1","lstm_2_p1","convlstm_2_p1","mconvlstm_p1")
caso2<-c("ffnn_1_p1","lstm_1_p1","convlstm_1_p1")


OBE<-read.csv('ObsTest.txt',sep=' ',header=FALSE)
OBE<-as.matrix(OBE)
OBE<-array(OBE,c(dim(OBE)[1],20,20))

Df1<-PrePros(caso1[1])
Df2<-PrePros(caso1[2])
Df3<-PrePros(caso1[3])
Df4<-PrePros(caso1[4])

Df5<-PrePros(caso2[1])
Df6<-PrePros(caso2[2])
Df7<-PrePros(caso2[3])

ndate<-Period(1)
ndate1<-ndate[21:as.integer(length(ndate)*0.83)]
ndate2<-ndate[(as.integer(length(ndate)*0.83)+21):length(ndate)]

# Matriz de ceros para realizar la redimension
# de los datos observed vs predicted

# dos tipos de secuencias de coordenadas para los pixels

seqlon <-seq(-75,-71, length.out =20)# Secuencia de coordenadas
seqlat <-seq(-38,-31, length.out=20)

seqlon2<-seq(-75,-71, length.out =21)# Secuencia de coordenadas
seqlat2<-seq(-38,-31, length.out=21)

# idate sirve para obtener id de un dia especifico
# este cambia segun el periodo
ifech<-'2015-09-16'
idate<-match(as.Date(ifech),ndate2)


# aux auxiliar que representa el punto que se va a
# plotear en los graficos considerando fecha 
# y magnitud del evento (En este caso el evento principal)

aux<-Catearth[Catearth$Fecha %in%
                 as.Date(ifech) &
                 Catearth$magn1 > 8 ,]

DFPT<-data.frame(aux$long,aux$lat,aux$magn1)
colnames(DFPT)<-c('long','lat','ident')

###################
# Ploteo de mapas #
###################

ifech<-'2015-09-16'
idate<-match(as.Date(ifech),ndate2)

# Intesidad con desplazamiento
DfRaster<-MatoDf(OBE[idate,,])
GGP1<-PltMap(DfRaster,DFPT,2,SHP)
GGP1


DfRaster<-MatoDf(Df1[idate,,])
GGP2<-PltMap(DfRaster,DFPT,caso1[1],SHP)
GGP2

DfRaster<-MatoDf(Df2[idate,,])
GGP3<-PltMap(DfRaster,DFPT,caso1[2],SHP)
GGP3

DfRaster<-MatoDf(Df3[idate,,])
GGP4<-PltMap(DfRaster,DFPT,caso1[3],SHP)
GGP4

DfRaster<-MatoDf(Df4[idate,,])
GGP5<-PltMap(DfRaster,DFPT,caso1[4],SHP)
GGP5

#Solo intensidad

DfRaster<-MatoDf(Df5[idate,,])
GGP6<-PltMap(DfRaster,DFPT,caso2[1],SHP)
GGP6

DfRaster<-MatoDf(Df6[idate,,])
GGP7<-PltMap(DfRaster,DFPT,caso2[2],SHP)
GGP7

DfRaster<-MatoDf(Df7[idate,,])
GGP8<-PltMap(DfRaster,DFPT,caso2[3],SHP)
GGP8

####################
# Series de tiempo #
####################

#Intensidad y desplazamiento

TsNet1<-FindCoord(log(OBE),ndate2,DFPT$long,DFPT$lat)
TsNet2<-FindCoord(log(Df1),ndate2,DFPT$long,DFPT$lat)
TsNet3<-FindCoord(log(Df2),ndate2,DFPT$long,DFPT$lat)
TsNet4<-FindCoord(log(Df3),ndate2,DFPT$long,DFPT$lat)
TsNet5<-FindCoord(log(Df4),ndate2,DFPT$long,DFPT$lat)

TsNet<-cbind(TsNet1,TsNet2$Net,TsNet3$Net,TsNet4$Net,TsNet5$Net)
colnames(TsNet)<-c("Date","Obs","FFNN","LSTM","ConvLSTM","MConvLSTM")
cols<-c('Observed'='grey0','FFNN'='green','LSTM'='purple','ConvLSTM'='blue','M-ConvLSTM'='red')

TSPL1 <- ggplot(TsNet, aes(x=Date)) +
  geom_line(aes(y = FFNN, color="FFNN"), size = 1)+
  geom_line(aes(y = LSTM, color="LSTM"), size = 1)+
  geom_line(aes(y = ConvLSTM, color="ConvLSTM"), size = 1)+
  geom_line(aes(y = MConvLSTM, color="M-ConvLSTM"), size = 1)+
  geom_line(aes(y = Obs, color = "Observed") ,size = 1) +
  xlab("")+
  ylab("Intensity function")+
  labs(color='')+
  scale_color_manual( values = cols)+
  theme(plot.title = element_text(size = 10)
        ,axis.title =element_text(size=10)
        ,axis.text  =element_text(size=10)
        ,legend.text=element_text(size =10)
        ,legend.key.width = unit(0.5, "cm"))

TSPL1

# Solo Intensidad

TsNet6<-FindCoord(log(Df5),ndate2,DFPT$long,DFPT$lat)
TsNet7<-FindCoord(log(Df6),ndate2,DFPT$long,DFPT$lat)
TsNet8<-FindCoord(log(Df7),ndate2,DFPT$long,DFPT$lat)


TsNet<-cbind(TsNet1,TsNet6$Net,TsNet7$Net,TsNet8$Net)
colnames(TsNet)<-c("Date","Obs","FFNN","LSTM","ConvLSTM")
cols<-c('Observed'='grey0','FFNN'='green','LSTM'='purple','ConvLSTM'='blue','M-ConvLSTM'='red')

TSPL2 <- ggplot(TsNet, aes(x=Date)) +
  geom_line(aes(y = FFNN, color="FFNN"), size = 1)+
  geom_line(aes(y = LSTM, color="LSTM"), size = 1)+
  geom_line(aes(y = ConvLSTM, color="ConvLSTM"), size = 1)+
  geom_line(aes(y = Obs, color = "Observed") ,size = 1) +
  xlab("")+
  ylab("Intensity function")+
  labs(color='')+
  scale_color_manual( values = cols)+
  theme(plot.title = element_text(size = 10)
        ,axis.title =element_text(size=10)
        ,axis.text  =element_text(size=10)
        ,legend.text=element_text(size =10)
        ,legend.key.width = unit(0.5, "cm"))

TSPL2