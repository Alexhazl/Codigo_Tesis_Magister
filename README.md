# Tesis-Magíster
Código utilizado en la siguiente tesis de magíster.

- [Redes Neuronales ConvLSTM para la predicción de eventos sísmicos en Chile](https://repositorio.unab.cl/xmlui/handle/ria/36157)

Dentro de este repositorio se encontrará el código utilizado para la presente tesis.

Cuenta con:
- Carpeta con los datos por cada estación de GPS
- Carpeta con contorno de las regiones de Chile (Shapes)
- Script con funciones creadas solamente para este proyecto, cada función trae su respectiva descripción
- Script de los principales cálculos abordados dentro de esta tesis

Procedimiento
1.- Tesis-Magister_1: Es todo el pre-procesamiento para calcular el desplazamineto asociado a cada evento sísmico
2.- Tesis-Magister_2: Define la grilla que se van a procesar en las diferentes redes nueronales (Intensidad y desplazamiento)
3.- Entrenamiento_Modelos: Contiene todos los modelos utilizados en el estudio, contiene tambien el proceso de optimización para cada red con keras-tuner
4.- Graficos_Tesis_Magister: Son los graficos para visualizar los resultados de cada red
