# Análisis Exploratorio de Datos (EDA) para Datos Petrofísicos

## Propósito

Esta metodología describe el enfoque sistemático para el análisis exploratorio de datos (EDA) de archivos LAS (Log ASCII Standard) en aplicaciones de machine learning para la industria petrolera. El EDA asegura que los datos estén comprendidos, validados y preparados optimalmente para el pipeline de machine learning.

## Flujo de Trabajo de Análisis

### 1. Descarga y Preparación de Datos

El primer paso es descargar los archivos de datos necesarios del sitio web de KGS:

- **`ks_wells.zip`**: Contiene datos de pozos.
- **`ks_wells.txt`**: Proporciona URLs para archivos LAS.

**Acciones**:

1. Crear una carpeta llamada `data` en el directorio del proyecto.
2. Guardar los archivos descargados en la carpeta `data`.
3. Descomprimir `ks_wells.zip` para acceder a los datos crudos.

### 2. Selección de Campo

Después de preparar los datos, seleccionar el campo petrolero específico con el que deseas trabajar. Esto ayuda a enfocar el análisis en un área particular de interés.

**Propósito**:

- Reducir el conjunto de datos para análisis dirigido.
- Gestionar el volumen de datos para procesamiento eficiente.

### 3. Exploración de Variables y Curvas

Entender las variables (curvas) en los archivos LAS es crucial.

**Pasos**:

1. **Identificar Curvas Disponibles en Cada Pozo**: Determinar qué curvas de medición están presentes en cada pozo dentro del campo seleccionado.
2. **Obtener Descripciones de Curvas**: Acceder a descripciones detalladas para cada curva para entender el tipo de datos (ej., registros de rayos gamma, resistividad).
3. **Agrupar Curvas para Análisis**: Organizar curvas en grupos basados en herramientas o tipos de medición. La agrupación ayuda en el análisis comparativo y simplifica el procesamiento.

**Propósito**:

- Familiarizarse con el contenido del conjunto de datos.
- Planificar análisis subsecuentes basados en datos disponibles.

### 4. Análisis Estadístico por Curva

Realizar análisis estadísticos en curvas individuales para entender sus distribuciones.

**Pasos**:

1. **Análisis Monovariable**: Analizar cada curva individualmente.
2. **Generar Boxplots e Histogramas**: Visualizar la distribución de datos para cada grupo de curvas.

**Propósito**:

- Identificar patrones, tendencias y anomalías en variables individuales.
- Detectar valores atípicos dentro de tipos de medición individuales.

### 5. Análisis Estadístico por Campo

Realizar análisis multivariables para explorar relaciones entre diferentes curvas a través del campo.

**Pasos**:

1. **Análisis Multivariable**: Examinar cómo diferentes curvas se relacionan entre sí.
2. **Visualización de Datos Faltantes**: Usar gráficos de datos faltantes (ej., `missingno`) para identificar vacíos en el conjunto de datos.
3. **Gráficos de Registros de Pozos**: Visualizar mediciones a través de profundidades para obtener insights sobre formaciones geológicas.

**Propósito**:

- Entender la interacción entre diferentes mediciones geológicas.
- Evaluar la completitud y calidad de los datos.

### 6. Detección de Valores Atípicos

Identificar y manejar valores atípicos para asegurar la integridad de los datos.

**Métodos**:

- **Z-Score**: Identifica puntos de datos que están a más de n desviaciones estándar de la media
- **Rango Intercuartílico (IQR)**: Detecta valores fuera del rango Q1 - 1.5*IQR a Q3 + 1.5*IQR
- **Isolation Forest**: Algoritmo de machine learning para detección de anomalías
- **DBSCAN**: Clustering basado en densidad para identificar puntos de ruido
- **Local Outlier Factor (LOF)**: Mide la desviación local de densidad de un punto

**Propósito**:

- Detectar puntos de datos atípicos que pueden sesgar el análisis.
- Mejorar la confiabilidad del modelado subsecuente.

### 7. Limpieza de Datos

Finalizar el conjunto de datos limpiándolo y preparándolo para machine learning.

**Pasos**:

1. **Filtrar Valores Atípicos**: Remover o corregir valores atípicos identificados.
2. **Manejar Datos Faltantes**: Imputar o remover valores faltantes.
3. **Estandarizar Datos**: Asegurar consistencia en formatos de datos y unidades.

**Propósito**:

- Producir un conjunto de datos de alta calidad adecuado para modelado predictivo.
- Minimizar errores y sesgos en aplicaciones de machine learning.

## Consideraciones Técnicas

### Optimización de Rendimiento

El proyecto ha sido optimizado para computación de alto rendimiento con aceleración GPU y capacidades de procesamiento paralelo:

- **Aceleración GPU**: TensorFlow configurado con colocación suave de dispositivos y entrenamiento de precisión mixta
- **Velocidad de Preprocesamiento**: Integración de cuML para transformaciones aceleradas por GPU
- **Validación Cruzada Paralelizada**: Utilización de múltiples núcleos de CPU para validación cruzada K-fold

### Calidad de Datos

La metodología enfatiza la calidad de datos a través de:

- **Validación de Consistencia**: Verificación de integridad de datos entre pozos
- **Estandarización de Formaciones**: Unificación de nomenclatura geológica
- **Control de Calidad Estadístico**: Aplicación de múltiples métodos de detección de anomalías

### Reproducibilidad

Todos los pasos están diseñados para ser:

- **Documentados**: Cada paso incluye justificación y propósito
- **Automatizados**: Scripts y notebooks proporcionan ejecución consistente
- **Versionados**: Control de versiones para seguimiento de cambios en metodología

## Resultados Esperados

Al completar esta metodología, tendrás:

1. **Conjunto de Datos Limpio**: Datos petrofísicos estandarizados y validados
2. **Documentación Comprensiva**: Registro completo del proceso de limpieza
3. **Insights Geológicos**: Entendimiento profundo de las características del campo
4. **Base para Modelado**: Datos preparados optimalmente para aplicaciones de machine learning

Esta metodología sirve como base para el pipeline completo de redes neuronales documentado en otros archivos del proyecto. 