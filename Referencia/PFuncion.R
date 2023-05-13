library(geosphere)
library(ggplot2)
library(scales)

ProyNazc<-function(df,nzcpt){
  #Proyecta df a un conjunto de datos de la placa de nasca
  Lonp<-c()
  for(i in 1:nrow(df)){
    Find<-0
    for(j in 1:(nrow(nzcpt)-1)){
      if(df$Lat[i]<=  nzcpt$Lat[j]  & nzcpt$Lat[j+1]<= df$Lat[i]){
        Find<-proyx(nzcpt$Lon[j],nzcpt$Lat[j],
                    nzcpt$Lon[j+1],nzcpt$Lat[j+1],
                    df$Lat[i])
      }
    }
    if(Find != 0){
      Lonp<-c(Lonp,Find)
    }else{
      Lonp<-c(Lonp,0)
    }
  }
  return(Lonp)
}


Harvdist<-function(df){
  lon1<-df$Lonp
  lat<-df$Lat
  lon2<-df$Lon
  HDist<-c()
    
  for(i in 1:nrow(df)){
    if(lon1[i]!=0){
      v<-distHaversine(c(lon1[i],lat[i]),c(lon2[i],lat[i]),r=6378137)
      HDist<-c(HDist,v)
    }else{
      HDist<-c(HDist,0)
    }
  }
  return(HDist)
  
  #R = 6378000
  #delat = (lat2*(pi/180))-(lat1*(pi/180))
  #delon = (lon2*(pi/180))-(lon1*(pi/180))
  #a = ((1-cos(delat))/2) + (cos(lat1)*cos(lat2)*((1-cos(delon))/2))
  #c = 2 * atan2(sqrt(a),sqrt(1-a))
  #d = R * c
  #return(d)  
}

disteuc<-function(x1,y1,x2,y2){
  d<-sqrt((x1-x2)^2+(y1-y2)^2)
  return(d)
}

proyx<-function(x1,y1,x2,y2,Yn){
  # x1,y1,x2,y2 son el intervalo de la placa
  # Yn es la latitud del evento sismico, este 
  # se proyectara en el intervalo que sera Xn
  #  if(Yn<=y1 & y2<=Yn){
  m<-(y2-y1)/(x2-x1)
  Xn<-(Yn-y1+(x1*m))/m
  return(Xn)
  #  }
}

Visual<-function(A,D){ 
  # Se visualizan dos plots 
  # uno conrespecto a sus datos originales
  # otro con la imputacion Kalman vs un ruido 
  # Esto para verificar candidato
  
  if(D==1){
    return(list(
      plot(get(A)$Lon),
      plot(na_interpolation(get(A)$Lon))
    ))}
  else{
    return(list(
      plot(get(A)$Lat),
      plot(na_interpolation(get(A)$Lat))
    ))
  }
}

Reemplazo<-function(A,D){
  # Reemplaza valores visualizados 
  # en la funcion anterior
  
  if(D==1){
    a<-(na_interpolation(A$Lon))
    for(i in 1:nrow(A)){
      if(is.na(A$Lon[i])){
        A$Lon[i]<-a[i]
      }
    }
    return(A$Lon)
  }else{
    a<-(na_interpolation(A$Lat))
    for(i in 1:nrow(A)){
      if(is.na(A$Lat[i])){
        A$Lat[i]<-a[i]
      }
    }
    return(A$Lat)
  }
}

Rankesta<-function(Est,df){
  # Calcula distancia Harversine de una 
  # estacion con respecto a las otras en el frame y muestra el Top 5
  
  x1<-(df[df$Estacion==Est,]$Lon)
  y1<-(df[df$Estacion==Est,]$Lat)
  HVS<-c()
  for(i in 1:nrow(df)){
    x2<-(df[i,]$Lon)    
    y2<-(df[i,]$Lat)
    yep<-distHaversine(c(x1,y1),c(x2,y2),r=6378137)
    HVS<-c(HVS,yep)
  }
  df<-cbind(df,HVS)
  df<-df[order(df$HVS),]
  rownames(df)<-NULL
  return(df[1:5,])
}

caldis<-function(est,dts){
  # Calcula los desplazamientos 1
  # dia anterior y 5 dias antes
  
  df<-get(est)
  n<-nrow(df)
  
  nms<-c()
  for(i in 0:6){
    assign(paste0('m',i+1),df[df$Fecha %in% (dts-i),])
  }
  
  for(i in 2:6){
    assign(paste0('euc',i-1),disteuc(get(paste0('m',i-1))$Lon,get(paste0('m',i-1))$Lat,
                                     get(paste0('m',i))$Lon,get(paste0('m',i))$Lat)
           *((2*pi*6371000)/360))
  }
  Mt1D<-euc1
  Mt2D<-sum(euc1,euc2)
  Mt3D<-sum(euc1,euc2,euc3)
  Mt4D<-sum(euc1,euc2,euc3,euc4)
  Mt5D<-sum(euc1,euc2,euc3,euc4,euc5)
  
  df2<-data.frame(Mt1D,Mt2D,Mt3D,Mt4D,Mt5D)
  colnames(df2)<-c('Mt1D','Mt2D','Mt3D','Mt4D','Mt5D')
  rownames(df2)<-NULL
  return(df2)
}

QuakHVS<-function(df,lon,lat,dts){
  # Calcula la distancia Harversine de terremotos VS estaciones de GPS (Top 5)
  # lon lat son del evento sismico, df estaciones de GPS
  # y ordena en funcion del mayor desplazamiento, dts es la fecha del evento
  
  x1<-lon
  y1<-lat
  HVS<-c()
  for(i in 1:nrow(df)){
    x2<-(df[i,]$Lon)    
    y2<-(df[i,]$Lat)
    yep<-distHaversine(c(x1,y1),c(x2,y2),r=6378137)
    HVS<-c(HVS,yep)
  }
  df<-cbind(df,HVS)
  df<-df[order(df$HVS),]
  df<-df[!is.na(df$Estacion),]# Si hay NA's no los considera
  df<-df[1:5,]
  
  # Comienza a calcular los desplazamientos
  
  nms<-c()
  for(i in 1:nrow(df)){
    nms[i]<-(df[i,]$Estacion)
  }
  
  aux<-data.frame()
  for(i in 1:length(nms)){
    aux<-rbind(aux,caldis(nms[i],dts))
  }
  
  df<-cbind(df,aux)
  df<-df[rev(order(df$Mt5D)),] # Ordena en funcion del mayor desplazamiento
  rownames(df)<-NULL
  return(df)
}

AV5D<-function(Est,Dt){
  # Calcula anagulo y velocidad
  # para 5 dias de una estacion
  # en una fecha especifica
  
  Est$Fecha<-as.Date(Est$Fecha)
  seqdate<-seq(as.Date(Dt)-5,as.Date(Dt),1)
  ex1<-Est[Est$Fecha %in% seqdate,]
  
  van1<-lm(ex1$Lon~ex1$Fecha)$coefficients
  van1<-van1[2]*5 # Multiplica por 5 por los 5 d?as (mm/5dias)
  
  van2<-lm(ex1$Lat~ex1$Fecha)$coefficients
  van2<-van2[2]*5
  
  ansp<-c(van1,van2) # Longitud, Latitud
  v<-sqrt((ansp[1])^2+(ansp[2])^2)
  a<-atan(ansp[2]/ansp[1]) # (N/E)
  f<-c(a,v)
  names(f)<-c('Angulo','Velocidad')
  return(f)
}

QuakINT<-function(dte,df){ 
  # Intervalo de tiempo GPS vs Evento
  # dte: Fecha del envento sismico
  # df: df del catalogo de GPS ya acotado por distancias
  # (el catalogo ya tiene la fechas de inicio y final)
  
  dte<-as.Date(dte)
  df2<-data.frame()
  for(i in 1:nrow(df)){
    if(df[i,]$FechaI <= (dte-5) && dte <= df[i,]$FechaF){  
      df2<-rbind(df2,df[i,])
    }
  }
  return(df2)
}

CatAngVel<-function(CtGPS,CtSIS){
  # Funcion para realizar el catalogo de sismo con el angulo y la velocidad
  # CtGPS: catalogo de GPS preprocesado
  # CtSis: evento sismico de catalogo (debe contener toda la fila)
  
  # Identifica las estaciones que estan en el intervalo de tiempo
  IFE<-QuakINT(CtSIS$Fecha,CtGPS) 
  
  # Calula el top 5 de distancias de la estaciones vs el
  # terremoto especificado mas los desplazamientos
  
  QvsT<-QuakHVS(IFE[,1:5],CtSIS$Lon,CtSIS$Lat,CtSIS$Fecha)
  
  # Calcula alngulo y velocidad de la mejor estacion por 5 dias
  av<-AV5D(get(QvsT$Estacion[1]),CtSIS$Fecha) 
  
  # Calcula coordenadas de la estacion en el momento del evento sismico
  ncords<-get(QvsT$Estacion[1])[get(QvsT$Estacion[1])$Fecha
                                == as.character(CtSIS$Fecha),]
  
  df<-data.frame(QvsT$Estacion[1],ncords[,2:3],t(av),QvsT$HVS[1],QvsT$Mt1D[1],QvsT$Mt2D[1],QvsT$Mt3D[1],QvsT$Mt4D[1],QvsT$Mt5D[1])
  colnames(df)<-c('Estacion','ELon','ELat','Angulo','Velocidad','HVQuk','Mt1D','Mt2D','Mt3D','Mt4D','Mt5D')
  
  return(df)
}

DnoisEst<-function(coord){# Comprueba y ejecuta cual dnoise tiene menor MSE
  library(wmtsa)
  mse<-c()
  ref<-c()
  for(i in seq(1,70,5)){
    assign(paste0('pred',i),wavShrink(coord,wavelet ='haar',thresh.scale=i))
    mse<-c(mse,(sum((coord-get(paste0('pred',i)))^2)/length(coord)))
    ref<-c(ref,paste0('pred',i))
  }
  pred75<-wavShrink(coord,wavelet ='d6')
  pred85<-wavShrink(coord,wavelet ='s8')
  
    
  MSE1=sum((coord-pred75)^2)/length(coord)
  MSE3=sum((coord-pred85)^2)/length(coord)
  
  
  mse<-c(mse,MSE1,MSE3)
  ref<-c(ref,'pred75','pred85')
  
  minv<-which.min(mse)
  
  fpred<-get(ref[minv])
  #print(ref[minv])
  #print(mse)
  return(fpred)
}

GrillaDes<-function(df){
  # Crea Grilla en función del desplazamiento para el procesamiento
  # en las redes neuronales
  
  seqlon<-seq(-75,-71, length.out=21)# Secuencia de coordenadas para intervalo
  seqlat<-seq(-38,-31, length.out=21)
  
  seqlon2<-seq(-75,-71, length.out=20)# Secuencia de coordenadas que se asignara
  seqlat2<-seq(-38,-31, length.out=20)
  
  auxlon<-c()
  auxlat<-c()
  
  Mmd<- array(0, dim=c(2922,20,20)) # Matriz de ceros con total de matrices
  
  seqdate<-seq(as.Date('2009-12-31'),as.Date('2017-12-30'),1)
  
  for(i in 1:nrow(df)){ 
    # Estima logitudes y latitdues del catalogo
    # en funcion de la grilla que se ocupara
    
    for(j in 1:(length(seqlon2))){
      if(seqlon[j]<= df$Lon[i] & df$Lon[i]<=seqlon[j+1]){
        auxlon<-c(auxlon,seqlon2[j])
        break
      }
    }
    
    for(k in 1:(length(seqlat2))){
      if(seqlat[k]<= df$Lat[i] & df$Lat[i]<=seqlat[k+1]){
        auxlat<-c(auxlat,seqlat2[k])
        break
      }
    }
  }
  df<-cbind(df,auxlon,auxlat) # Agrega las logitudes y latitudes estimadas al catalogo
  
  for(i in 1:length(seqdate)){
    # Reemplaza desplazamientos en las correspondientes coordenadas
    
    aux<-df[df$Fecha%in%seqdate[i],]                     # Identifica df en funcion de una fecha
    aux<-aux[order(-aux$magn1),]                         # Ordena de forma asendete con respecto a la magnitud
    dorp_id<-duplicated(aux[,c(30,31)],fromLast = FALSE) # Identifica dupicados en funcion de la longitud y la latitud
    aux<-aux[!dorp_id,]                                  # Extrae los duplicados, considerando validos solo las magnitudes maximas
    for(j in 1:nrow(aux)){                               # Reemplaza los desplazamiento en la grilla correspondiente al dia del suceso 
      Mmd[i,aux$auxlon[j]==seqlon2,aux$auxlat[j]==seqlat2]<-aux$Mt5D[j]
    }
  }
  return(Mmd)
}


Tcaso<-function(cs){
  # Identifica caso de redes
  if(cs=="ffnn_2_p1" ||cs=="ffnn_2_p2" ||cs=="ffnn_2_p3" ||cs=="ffnn_1_p1" ||cs=="ffnn_1_p2" ||cs=="ffnn_1_p3"){
    return("FFNN")
  }else if(cs=="lstm_2_p1" ||cs=="lstm_2_p2" ||cs=="lstm_2_p3" ||cs=="lstm_1_p1" ||cs=="lstm_1_p2" ||cs=="lstm_1_p3"){
    return("LSTM")  
  }else if(cs=="convlstm_2_p1" ||cs=="convlstm_2_p2" ||cs=="convlstm_2_p3" ||cs=="convlstm_1_p1" ||cs=="convlstm_1_p2" ||cs=="convlstm_1_p3"){
    return("ConvLSTM")
  }else if(cs=="mconvlstm_p1" ||cs=="mconvlstm_p2" ||cs=="mconvlstm_p3"){
    return("MConvLSTM")
  }else{
    return("Observed")
  }
}


PrePros<-function(cs){
  # Preprosesa los datos de test
  Df<-read.csv(paste0('TestP_',cs,'.txt'),sep=' ',header=FALSE)
  Df<-as.matrix(Df)
  Df<-array(Df,c(dim(Df)[1],20,20))
  return(Df)
}

Period<-function(p){
  # Identifica periodos 
  if(p==1){
    date<-seq(as.Date('2009-12-31'),as.Date('2015-12-31'),1)# intervalo de fechas
    return(date)  
  }else if(p==2){
    date<-seq(as.Date('2011-01-01'),as.Date('2016-12-31'),1)# intervalo de fechas
    return(date)
  }else if(p==3){
    date<-seq(as.Date('2012-01-01'),as.Date('2017-12-31'),1)# intervalo de fechas
    return(date)
  }
}


MatoDf<-function(mat){
  # TranSforma matriz a un data.frame 
  aux<-c()# Sitema para ordenar raster (observed)
  auxlon<-c()
  auxlat<-c()
  
  seqlon<-seq(-75,-71, length.out =20)# Secuencia de coordenadas
  seqlat<-seq(-38,-31, length.out=20)
  
  for(i in 1:20){
    for(j in 1:20){
      aux<-c(aux,mat[i,j])
      auxlon<-c(auxlon,seqlon[i])
      auxlat<-c(auxlat,seqlat[j])
    }
  }
  # Dataframe del raster
  DFraster<-data.frame(auxlon,auxlat,aux)
  colnames(DFraster)<-c('Lon','Lat','FI')
  return(DFraster)
}

PltMap<-function(df,dfpt,cs,shp){
  # Plotea graficos de mapa en funcion de alguna red 
  print(class(dfpt))
  seqlon <-seq(-75,-71, length.out =20)# Secuencia de coordenadas
  seqlat <-seq(-38,-31, length.out=20)
  
  ggp<-ggplot(df, aes(Lon, Lat) )+
    coord_map()+
    ggtitle(Tcaso(cs))+
    geom_tile(aes(fill=FI))+
    scale_fill_gradient(name=" Intensity \n function ",breaks = seq(min(df$FI),max(df$FI),length.out = 4),labels = comma) +
    coord_fixed(ylim = c(min(seqlat),max(seqlat)), xlim = c(min(seqlon),max(seqlon)))+
    geom_polygon(data = shp, aes(x = long, y = lat, group = group), #Se traza el shp
                 color = "black", 
                 alpha = 0.2)+
    theme(panel.grid.major = element_blank()# Estructura bonita 
          ,panel.grid.minor = element_blank()
          ,axis.title.x=element_blank()
          ,axis.ticks.x=element_blank()
          ,plot.title = element_text(size = 20)
          ,axis.title.y=element_blank()
          ,axis.ticks.y=element_blank()
          ,axis.text  =element_text(size=20)
          ,legend.text=element_text(size =20)
          ,legend.title=element_text(size=20)
          ,plot.margin=grid::unit(c(0,0,0,0), "mm")
          ,legend.key.width = unit(1, "cm")
          ,legend.key.height = unit(1, "cm"))+
    geom_text_repel(data=dfpt, aes(long, lat, label = ident), color= ("red"), show.legend = FALSE ,hjust = 0, nudge_x =-0.5 ,nudge_y = -0.5, size=9)+
    geom_point(data=dfpt, aes(long, lat), color= ("red"),inherit.aes = FALSE, size = 2 )
  return(ggp)
}

FindCoord<-function(net,time,long,lat){
  # Funcion que encuentra posicion de pixel en funcion de un evento sismico 
  seqlon2<-seq(-75,-71, length.out =21)# Secuencia de coordenadas
  seqlat2<-seq(-38,-31, length.out=21)
  
  for(i in 1:length(seqlon2)){ 
    if(seqlon2[i]<long & seqlon2[i+1]>=long){
      clon<-i
      break
    }
  }
  
  for(i in 1:length(seqlat2)){
    if(seqlat2[i]<=lat & seqlat2[i+1]>=lat){
      clat<-i
      break
    }
  }
  df<-data.frame(time,net[,clon,clat])
  colnames(df)<-c('Date','Net')
  return(df)
}

