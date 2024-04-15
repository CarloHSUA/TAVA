# TAVA

![annotated_image](https://github.com/CarloHSUA/TAVA/assets/99215566/0333662e-3421-4ed9-bdef-8de360085f71)

# Resumen


El __reconocimiento de emociones por voz y por imagen__ son campos esenciales en el desarrollo de espejos inteligentes, ya que permiten que estos dispositivos comprendan y respondan mejor a las necesidades y estados emocionales de los usuarios. Al detectar las emociones de una persona a través de su voz o imagen reflejada, el espejo inteligente puede adaptar su interfaz y funcionalidades para proporcionar apoyo emocional, sugerir actividades para mejorar el estado de ánimo o incluso alertar a los cuidadores si se detectan signos de angustia o malestar.

El __reconocimiento de actividades físicas__ también es crucial, ya que los espejos inteligentes pueden actuar como entrenadores personales al monitorear y analizar el rendimiento durante el ejercicio. Esta capacidad puede ayudar a los usuarios a mantenerse motivados, establecer metas realistas y mejorar su estado físico general. Además, el reconocimiento de actividades físicas puede ser especialmente beneficioso para las personas mayores o aquellas con condiciones médicas crónicas, ya que puede ayudar a prevenir lesiones y promover un estilo de vida activo y saludable. 
  
# Requisitos
## Instalar dependencias:
```
pip install -r requirements.txt
```

# Aplicación
## Reconocimiento de emociones por imagen.
```
cd ./app
python app.py
```

## Reconocimiento de emociones por voz.
```
Cambiar e introducir el directorio y la aplicación de voz ......
cd ./app
python pose.py
```

## Reconocimiento de actividades físicas.
```
cd ./app
```
Para obtener la predicción de una imagen:
```
python pose.py image
```
o de un vídeo:
```
python pose.py video
```

# Estructura del repositorio
- `app/` : Contiene dos aplicaciones, *app.py* para el reconocimiento de emociones en tiempo real y *pose.py* para el reconocimiento de actividades físicas en tiempo real.
- `datasets/` : Carpeta donde se almacenan los datasets generados.
- `emotion_recognition/` : Contiene un *notebook* descargado de Kaggle con la carga de dos datasets (FER-2013 y AffectNet) con su correspondiente predicción y resultados.
- `scripts_data/` : Contiene fichero que descarga los conjuntos de datos y los almacena en la carpeta *datasets/*.
- `vital_signs/` : 
