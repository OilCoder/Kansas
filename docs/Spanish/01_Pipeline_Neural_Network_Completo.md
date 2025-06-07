# Pipeline Completo de Redes Neuronales para Análisis Petrofísico

## Propósito

Este documento describe el pipeline completo de redes neuronales para análisis petrofísico, desde el procesamiento inicial de datos hasta la evaluación final del modelo. El pipeline integra técnicas avanzadas de ingeniería de datos, optimización de hiperparámetros, y validación rigurosa para producir modelos predictivos robustos de propiedades petrofísicas.

**¿Qué problema estamos resolviendo?** La predicción precisa de propiedades petrofísicas (como CNLS y formaciones geológicas) a partir de registros de pozos requiere un procesamiento sofisticado de datos y modelado que considera las complejidades geológicas únicas de cada campo petrolero.

**¿Por qué necesitamos un pipeline integral?** Cada paso del proceso (desde el preprocesamiento hasta la evaluación) afecta la calidad final del modelo. Un enfoque sistemático asegura reproducibilidad, robustez, y calidad en la predicción.

**¿Cómo logra esto el pipeline?** Mediante la integración de ocho pasos secuenciales que transforman datos crudos de pozos en modelos predictivos calibrados y validados.

## Descripción del Flujo de Trabajo

El pipeline se ejecuta en ocho pasos secuenciales, cada uno construyendo sobre los resultados del anterior:

```
Datos Crudos → [Step 0] → [Step 0.5] → [Step 1] → [Step 2] → [Step 3] → [Step 4] → [Step 5] → [Step 6] → [Step 7] → Modelo Final
```

### Visión General de los Pasos

| Paso | Nombre | Propósito | Referencia |
|------|---------|-----------|------------|
| 0 | Configuración del Entorno | Preparación de directorios y configuración TensorFlow | Este documento |
| 0.5 | Preprocesamiento de Datos | Corrección de consistencia y estandarización | Este documento |
| 1 | División de Datos | Partición en conjuntos de entrenamiento/validación y prueba externa | Este documento |
| 2 | Ingeniería de Características | Generación de características petrofísicas avanzadas | [02_Ingeniería de Características](./02_Ingeniería%20de%20Características%20Petrofísicas%20De%20Registros%20Crudos%20a%20Características%20Predictivas.md) |
| 3 | Normalización | Escalado y transformación de datos | [03_Normalización](./03_Normalización%20en%20el%20procesamiento%20de%20datos%20petrofísicos.md) |
| 4 | Optimización de Hiperparámetros | Exploración inteligente con Optuna | [04_Exploración Inicial Hiperparámetros](./04_Exploracion_Inicial_Hyperparametros_Optuna.md) |
| 5 | Validación Cruzada | Selección rigurosa del mejor modelo | [05_Validación Rigurosa Cross-Validation](./05_Validacion_Rigurosa_Cross_Validation.md) |
| 6 | Entrenamiento Final | Entrenamiento de producción con división 90/10 | [06_Entrenamiento Final Producción](./06_Entrenamiento_Final_Produccion.md) |
| 7 | Evaluación | Predicción y análisis en conjunto externo | Este documento |

## Entradas y Salidas

### Entradas del Pipeline
- **data**: Diccionario con datos de pozos petroleros (registros LAS)
- **selected_curves**: Lista de curvas de entrada para entrenamiento
- **curves_to_predict**: Lista de curvas objetivo a predecir
- **train_task**: Tarea de entrenamiento ('regression', 'classification', 'both')

### Salidas del Pipeline
- **Modelo Final**: Red neuronal entrenada y optimizada
- **Predictores**: Conjunto completo de transformadores y escaladores
- **Resultados de Evaluación**: Predicciones en conjunto externo con métricas
- **Documentación Completa**: Historial de entrenamiento y configuraciones

## Pasos Detallados

### Step 0: Configuración del Entorno

**¿Qué problema resuelve este paso?** La configuración adecuada del entorno computacional es crítica para el entrenamiento exitoso de redes neuronales en GPU.

**¿Por qué no configurar ad-hoc?** Una configuración sistemática asegura reproducibilidad, optimización de memoria, y compatibilidad entre componentes.

#### Implementación

```python
# Referencia: pipeline.py, líneas 107-130
# Configuración de directorios específicos por tarea
task_base_dir = os.path.join(current_dir, train_task)
directories = [
    os.path.join(task_base_dir, 'files'),      # Logs y estudios Optuna
    os.path.join(task_base_dir, 'model'),      # Modelos y preprocesadores
    os.path.join(task_base_dir, 'results'),    # Gráficos y métricas
]
```

#### Configuración TensorFlow

El sistema configura automáticamente TensorFlow para optimización en RTX 4080:

```python
# Referencia: pipeline.py, _configure_tensorflow()
# Precisión mixta para Tensor Cores
mixed_precision.set_global_policy('mixed_float16')
```

**¿Qué son los Tensor Cores?** Unidades de procesamiento especializadas en RTX 4080 que aceleran operaciones de redes neuronales usando precisión mixta.

### Step 0.5: Preprocesamiento de Datos

**¿Qué inconsistencias existen en datos petrofísicos?** Los datos de pozos petroleros frecuentemente contienen:
- Formaciones con nombres inconsistentes entre pozos
- Valores atípicos debido a errores de instrumentación
- Registros incompletos o con gaps significativos

**¿Por qué es crítico este paso?** Inconsistencias en los datos se propagan a través del pipeline, resultando en modelos inestables o sesgados.

#### Corrección de Consistencia

```python
# Referencia: pipeline.py, líneas 147-155
data_with_standardized_formations, preprocessing_report = preprocess_data_comprehensive(data)
```

**¿Qué hace la corrección comprensiva?**
1. **Estandarización de Formaciones**: Unifica nombres de formaciones entre pozos
2. **Corrección de Outliers**: Identifica y corrige valores extremos
3. **Validación de Integridad**: Verifica consistencia de tipos de datos

### Step 1: División de Datos

**¿Por qué dividir datos antes del entrenamiento?** Una evaluación honesta del modelo requiere datos que nunca haya visto durante cualquier etapa del desarrollo.

**¿Cuál es la diferencia entre validación y prueba externa?** 
- **Validación**: Usada durante optimización de hiperparámetros y selección de modelo
- **Prueba Externa**: Evaluación final completamente independiente

#### Estrategia de División

```python
# Referencia: pipeline.py, líneas 157-163
train_validation_data, external_test_data, discarded_wells, _ = split_wells_by_prediction(
    data_with_standardized_formations, 
    curves_to_predict, 
    min_curves=MIN_CURVES, 
    random_seed=RANDOM_SEED
)
```

**¿Por qué división por pozos y no por muestras?** Evita fuga de datos (data leakage) donde información del mismo pozo aparece en entrenamiento y prueba.

#### Criterios de Exclusión

Los pozos se descartan si:
- Tienen menos de `MIN_CURVES` registros válidos
- Faltan las curvas objetivo especificadas
- Contienen datos inconsistentes irrecuperables

### Step 2: Ingeniería de Características

Este paso se encuentra completamente documentado en:
**📄 [02_Ingeniería de Características Petrofísicas](./02_Ingeniería%20de%20Características%20Petrofísicas%20De%20Registros%20Crudos%20a%20Características%20Predictivas.md)**

#### Resumen del Proceso

```python
# Referencia: pipeline.py, líneas 190-201
engineered_data, feature_info, final_cols = generate_features(
    train_validation_data,
    selected_curves, 
    curves_to_predict, 
    window_size=ROLLING_WINDOW_SIZE, 
    num_clusters=DEFAULT_NUM_CLUSTERS, 
    preserve_master=False
)
```

**Características generadas incluyen:**
- Descriptores estadísticos (media, varianza, asimetría)
- Análisis multiescala (ventanas deslizantes)
- Clustering geológico (K-means en propiedades petrofísicas)
- Métricas de entropía y complejidad

### Step 3: Normalización de Datos

Este paso se encuentra completamente documentado en:
**📄 [03_Normalización en el procesamiento de datos petrofísicos](./03_Normalización%20en%20el%20procesamiento%20de%20datos%20petrofísicos.md)**

#### Resumen del Proceso

```python
# Referencia: pipeline.py, líneas 210-221
X_scaled, y_scaled, per_well_strategies, global_feature_scalers, 
categorical_encoders, column_types, feature_columns, global_columns, 
well_descriptors, target_scalers, formation_encoder, unknown_index, 
normalizers, fit_errors = prepare_and_normalize_data(...)
```

**Estrategias implementadas:**
- Normalización por pozo vs. global automática
- Transformaciones específicas basadas en sesgo estadístico
- Codificación robusta de variables categóricas
- Gestión de clases desconocidas en formaciones

### Step 4: Optimización de Hiperparámetros

Este paso se encuentra completamente documentado en:
**📄 [04_Exploración Inicial Hiperparámetros Optuna](./04_Exploracion_Inicial_Hyperparametros_Optuna.md)**

#### Configuración de Clases

```python
# Referencia: pipeline.py, líneas 259-265
all_classes = y_scaled['Formation'].unique()
valid_classes = [cls for cls in all_classes if cls != unknown_index]
classification_output_shape = len(valid_classes) + 1  # +1 para clase unknown
```

**¿Por qué calcular la forma de salida dinámicamente?** Diferentes campos petroleros tienen diferentes números de formaciones, requiriendo adaptación automática de la arquitectura.

### Step 5: Validación Cruzada

Este paso se encuentra completamente documentado en:
**📄 [05_Validación Rigurosa Cross-Validation](./05_Validacion_Rigurosa_Cross_Validation.md)**

#### Integración con Optuna

```python
# Referencia: pipeline.py, líneas 289-297
best_config, cv_results, best_model = cross_validation(
    X=X_scaled, 
    y=y_scaled, 
    top_configs=top_configs, 
    unknown_index=unknown_index, 
    classification_output_shape=classification_output_shape,
    train_task=train_task,
    save_path=os.path.join(task_base_dir, 'model', 'cross_validation'),
)
```

### Step 6: Entrenamiento Final

Este paso se encuentra completamente documentado en:
**📄 [06_Entrenamiento Final Producción](./06_Entrenamiento_Final_Produccion.md)**

#### Reconstrucción de Hiperparámetros

```python
# Referencia: pipeline.py, líneas 309-312
# Reconstruye el dict de hiperparámetros
best_config = reconstruct_full_hyperparams(best_config['config'])
```

**¿Por qué reconstruir hiperparámetros?** La validación cruzada guarda configuraciones en formato comprimido que debe expandirse para el entrenamiento final.

### Step 7: Evaluación en Conjunto Externo

**¿Qué problema resuelve la evaluación externa?** Proporciona una estimación no sesgada del rendimiento del modelo en datos completamente nuevos.

**¿Por qué no es suficiente la validación cruzada?** La validación cruzada puede estar sesgada por la selección de hiperparámetros y configuraciones de modelo.

#### Predicción en Pozos Externos

```python
# Referencia: pipeline.py, líneas 332-349
predictions = predict_wells(
    wells_data=external_test_data,
    model_path=model_path,
    selected_curves=selected_curves,
    curves_to_predict=curves_to_predict,
    per_well_strategies=per_well_strategies,
    global_feature_scalers=global_feature_scalers,
    categorical_encoders=categorical_encoders,
    # ... más parámetros de normalización
)
```

**¿Qué incluye el proceso de predicción?**

1. **Ingeniería de Características**: Aplica las mismas transformaciones usadas en entrenamiento
2. **Normalización**: Usa los escaladores entrenados para transformar datos
3. **Predicción**: Ejecuta el modelo final en datos transformados
4. **Desnormalización**: Convierte predicciones a escala original

#### Generación de Visualizaciones

```python
# Referencia: pipeline.py, líneas 365-378
plot_files = save_prediction_plots(
    predictions_data=predictions,
    task_base_dir=task_base_dir,
    create_summary=True,
    max_wells=None  # Graficar todos los pozos
)
```

**¿Qué visualizaciones se generan?**

1. **Gráficos Individuales**: Track-style plots para cada pozo con:
   - Registros originales vs. predicciones
   - Intervalos de confianza
   - Métricas de error por profundidad

2. **Gráficos Resumen**: Comparaciones agregadas mostrando:
   - Distribución de errores entre pozos
   - Métricas de rendimiento globales
   - Análisis de incertidumbre

#### Guardado de Resultados

```python
# Referencia: pipeline.py, líneas 354-361
if predictions:
    results_dir = os.path.join(task_base_dir, 'results')
    save_predictions_to_csv(predictions, results_dir)
    logger.info(f"✅ Prediction CSVs saved to: {results_dir}")
```

**¿Qué contienen los archivos CSV?**
- Predicciones punto por punto para cada pozo
- Métricas de confianza y incertidumbre
- Datos originales para comparación
- Metadatos del modelo y configuración

## Fundamentos Técnicos

### Gestión de Memoria

**¿Por qué es crítica la gestión de memoria en GPU?** El entrenamiento de redes neuronales puede agotar rápidamente la memoria de GPU, especialmente durante optimización de hiperparámetros con múltiples trials.

**¿Cómo maneja esto el pipeline?**

```python
# Referencia: Múltiples puntos en pipeline.py
clean_memory_for_trial()  # Ejecutado después de cada paso mayor
```

La función `clean_memory_for_trial()` incluye:
- Liberación explícita de variables de TensorFlow
- Limpieza de caché de GPU
- Garbage collection de Python
- Reinicio de sesiones de TensorFlow cuando necesario

### Logging y Trazabilidad

**¿Qué información es crítica para auditoría?** Todo el proceso debe ser trazable para reproducibilidad científica y depuración.

**¿Cómo logra esto el pipeline?**

```python
# Referencia: pipeline.py, configuración de logging
log_file = os.path.join(task_base_dir, 'files', 'neural_network.log')
configure_logging(log_file)
```

**Información registrada incluye:**
- Configuración completa de hiperparámetros
- Detalles de división de datos
- Métricas de rendimiento en cada paso
- Errores y advertencias
- Tiempos de ejecución
- Uso de recursos del sistema

### Estructura de Directorios

**¿Por qué organización específica por tarea?** Diferentes tareas (regresión, clasificación, multi-task) requieren modelos y configuraciones distintas.

```
neural_network/
├── regression/          # Para train_task='regression'
│   ├── files/          # Logs, estudios Optuna
│   ├── model/          # Modelos entrenados
│   │   ├── optuna_trials/
│   │   ├── cross_validation/
│   │   └── final_train/
│   ├── results/        # Predicciones y gráficos
│   └── scalers/        # Transformadores guardados
├── classification/      # Para train_task='classification'
└── both/               # Para train_task='both'
```

## Formulación Matemática

### Pipeline como Función Compuesta

El pipeline completo puede expresarse como una función compuesta:

```
M = f₇(f₆(f₅(f₄(f₃(f₂(f₁(f₀(D))))))) 
```

Donde:
- D: Datos crudos de entrada
- f₀: Configuración del entorno
- f₁: Preprocesamiento y división
- f₂: Ingeniería de características  
- f₃: Normalización
- f₄: Optimización de hiperparámetros
- f₅: Validación cruzada
- f₆: Entrenamiento final
- f₇: Evaluación
- M: Modelo final y resultados

### Optimización Multi-Objetivo

Para tareas 'both', el pipeline optimiza:

```
θ* = argmin [λ₁ · L_reg(θ) + λ₂ · L_cls(θ) + R(θ)]
     θ∈Θ
```

Donde:
- L_reg: Loss de regresión (CNLS prediction)
- L_cls: Loss de clasificación (Formation prediction)  
- R(θ): Término de regularización
- λ₁, λ₂: Pesos de balance entre tareas

### Validación Estadística

La evaluación final proporciona estimadores no sesgados:

```
E[L_test] = E[L(M(X_test), Y_test)]
```

Con intervalo de confianza:
```
CI = E[L_test] ± t_{α/2,n-1} · (s/√n)
```

## Referencia de Código

**Implementación Principal**: `code/src/neural_network/pipeline.py`

**Funciones Clave**:
- `pipeline()`: Orquestación principal del pipeline completo
- `_configure_tensorflow()`: Configuración optimizada de TensorFlow
- Múltiples llamadas a módulos especializados para cada paso

**Archivos de Configuración**:
- `hyperparameters.py`: Parámetros globales y configuraciones de tareas
- `initialize_gpu.py`: Configuración inicial de GPU

**Utilidades de Soporte**:
- `utils/core/utils.py`: Logging y configuración general
- `utils/neural_network/memory_management/`: Gestión de memoria GPU
- `utils/neural_network/visualization/`: Generación de gráficos

## Resultados Esperados

Después de ejecutar el pipeline completo, tendrás:

1. **Modelo de Producción**: Red neuronal completamente entrenada y validada
2. **Sistema de Preprocesamiento**: Conjunto completo de transformadores reproducibles
3. **Documentación Completa**: Logs detallados de todo el proceso de entrenamiento
4. **Evaluación Rigurosa**: Métricas de rendimiento en datos completamente externos
5. **Visualizaciones Comprensivas**: Gráficos de track-style para análisis petrofísico
6. **Estructura Reproducible**: Organización clara que facilita futuras iteraciones

El pipeline representa un sistema end-to-end robusto para convertir datos crudos de pozos petroleros en modelos predictivos calibrados, validados, y listos para despliegue en aplicaciones petrofísicas. 