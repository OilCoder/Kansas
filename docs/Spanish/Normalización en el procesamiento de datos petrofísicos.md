# Normalización en el Procesamiento de Datos Petrofísicos

## Introducción
En la ingeniería petrolera, trabajamos con conjuntos de datos complejos derivados de registros petrofísicos. Estos registros proporcionan información esencial sobre formaciones subterráneas, guiando decisiones en exploración y producción. Sin embargo, los datos brutos de estos registros pueden variar ampliamente en escala y distribución, lo que plantea desafíos para el análisis y el modelado. La normalización es un paso de preprocesamiento crucial que aborda estos desafíos, asegurando que nuestros datos sean adecuados para modelos de aprendizaje automático y otros métodos analíticos.

Este documento explica la importancia de la normalización en el procesamiento de datos petrofísicos, destaca cómo las consideraciones operativas en la industria petrolera afectan nuestro enfoque, describe la estrategia integrada que empleamos y proporciona explicaciones matemáticas de los métodos de escalado utilizados.

## Importancia de la Normalización
La normalización asegura que todas las características de entrada en nuestro conjunto de datos estén en una escala similar. Esto es vital por varias razones:

- **Entrenamiento Estable del Modelo**: Los modelos de aprendizaje automático, particularmente las redes neuronales, funcionan mejor cuando las características de entrada tienen escalas comparables. Grandes diferencias en las escalas de las características pueden conducir a un entrenamiento inestable y modelos subóptimos.

- **Mejor Rendimiento del Modelo**: Los datos normalizados pueden mejorar la velocidad de convergencia de los algoritmos de entrenamiento y conducir a un mejor rendimiento general.

- **Contribución Equitativa de Características**: Evita que las características con escalas más grandes dominen el proceso de aprendizaje, permitiendo que todas las características contribuyan por igual.

En el procesamiento de datos petrofísicos, tratamos con mediciones que pueden variar ampliamente en escala. Por ejemplo, los registros de resistividad pueden tener valores en miles, mientras que otros registros como el potencial espontáneo (SP) pueden tener valores pequeños o incluso negativos. Sin normalización, estas disparidades pueden impactar negativamente nuestros análisis y modelos.

## Características de los Datos y Consideraciones Operativas
Nuestro conjunto de datos comprende registros petrofísicos recolectados de múltiples pozos en diferentes ubicaciones. Las características clave y consideraciones operativas incluyen:

- **Diversas Escalas de Medición**: Registros como la resistividad pueden tener valores altos (por ejemplo, miles de ohm-metros), mientras que otros como los registros de calibre pueden variar entre 5 y 12 pulgadas. Algunos registros, como el SP, a menudo tienen valores negativos.

- **Variabilidad Entre Pozos**: Diferentes pozos podrían haber sido registrados usando varias herramientas y bajo diferentes condiciones operativas. Esto puede llevar a discrepancias en las distribuciones de datos entre pozos.

- **Variabilidad de Herramientas de Registro**: Diferentes herramientas, métodos de calibración y proveedores pueden introducir inconsistencias. Por ejemplo, dos herramientas de resistividad de diferentes fabricantes podrían registrar valores ligeramente diferentes bajo las mismas condiciones.

- **Diversidad Geológica**: Las formaciones subterráneas varían en composición, porosidad, permeabilidad y contenido de fluidos. Estas variaciones afectan las mediciones registradas por nuestros logs.

- **Condiciones Operativas**: Cambios en las propiedades del fluido de perforación, condiciones del pozo y velocidades de registro pueden influir en las lecturas de los registros.

Estos factores requieren un enfoque cuidadoso para la normalización. Necesitamos una estrategia que se adapte a la variabilidad inherente en nuestros datos debido a realidades operativas, mientras los prepara efectivamente para análisis y modelado.

## Estrategia de Normalización
Nuestra estrategia de normalización está diseñada para equilibrar la necesidad de un preprocesamiento efectivo de datos con las consideraciones prácticas de las operaciones petroleras. Integramos nuestras decisiones en un enfoque cohesivo de la siguiente manera:

### Estrategia de Detección Automática de Columnas Basada en Varianza
Nuestro pipeline de normalización implementa un **sistema de detección automática** que elimina la necesidad de especificación manual de columnas globales. Cada columna en el conjunto de datos se evalúa utilizando un análisis sofisticado de varianza por pozo para determinar el enfoque de normalización más apropiado.

Este enfoque asegura que cada curva sea tratada consistentemente en todos los pozos, respetando las diferencias operativas sin comprometer la consistencia del modelo. Al analizar las características de cada feature a través de evaluación individual de pozos, mantenemos la integridad de los patrones específicos del pozo mientras permitimos comparaciones entre pozos.

### Evaluación de Varianza por Pozo con Ratio de Pozos Fallidos
La innovación central de nuestra estrategia es el **sistema de evaluación de varianza por pozo** que determina el tratamiento de columnas basado en el rendimiento individual de pozos:

**Resumen del Algoritmo:**
1. **Evaluación Individual de Pozos**: Para cada columna, evaluamos la varianza dentro de cada pozo individualmente después de aplicar la transformación apropiada
2. **Conteo de Pozos Fallidos**: Los pozos donde la columna transformada tiene varianza por debajo del umbral (`var_threshold_perwell`) se cuentan como "fallidos"
3. **Cálculo del Ratio de Pozos Fallidos**: Se calcula el ratio de pozos fallidos respecto al total de pozos
4. **Decisión Basada en Umbral**: Si el ratio de pozos fallidos excede `min_failed_wells_ratio` (por defecto: 0.5), la columna se convierte en candidata global

**Parámetros Clave:**
- `min_failed_wells_ratio`: Por defecto 0.5 (50% de los pozos deben fallar para que la columna sea global)
- `var_threshold_perwell`: Umbral de varianza para evaluación individual de pozos
- `var_threshold_global`: Umbral de varianza para validación de columnas globales

Este enfoque basado en varianza asegura que cada característica sea tratada de acuerdo a sus propiedades estadísticas a través de la población de pozos, optimizando la información que puede proporcionar al modelo.

### Reglas de Decisión e Implementación
La siguiente tabla resume las reglas de decisión implementadas en nuestro sistema de detección automática:

| Condición | Acción Aplicada | Implementación |
|-----------|----------------|----------------|
| Varianza global < 1e-3 | 'drop' | La característica se excluye del conjunto de datos |
| Ratio de pozos fallidos ≥ min_failed_wells_ratio | 'global' | Escalado una vez con escalador global |
| Ratio de pozos fallidos < min_failed_wells_ratio | 'per-well' | Escalado individual por pozo usando tipo de transformación almacenado |
| Contiene valores negativos con asimetría | 'power_robust' | Pipeline Yeo-Johnson + RobustScaler |
| Valores positivos con alta asimetría | 'boxcox_robust' | Pipeline Box-Cox + RobustScaler |
| Distribución similar a la normal | 'robust' | Solo RobustScaler |
| Datos categóricos | 'categorical' | Se aplica OrdinalEncoder |

Estos tipos de transformadores corresponden directamente a la implementación en la función `fit_feature_scalers()` de nuestro código base y la lógica de transformación. Esta alineación entre la documentación y el código asegura que la estrategia descrita refleje con precisión la implementación real.

### Nueva Estrategia por Pozo: Tipos de Transformación vs Escaladores Ajustados
Una mejora crítica en nuestra nueva estrategia es cómo manejamos las columnas por pozo:

**Enfoque Anterior (Obsoleto):**
- Almacenaba escaladores ajustados para cada combinación pozo-columna
- Requería memoria extensiva para almacenamiento de escaladores
- Flexibilidad limitada para predicción de pozos nuevos

**Enfoque Nuevo (Actual):**
- Almacena solo el **tipo de transformación** ('robust', 'power_robust', 'boxcox_robust') para columnas por pozo
- Para cada pozo (entrenamiento o nuevo), ajusta escaladores frescos usando el tipo de transformación almacenado con los datos de ese pozo
- Reduce dramáticamente los requerimientos de memoria y mejora la precisión de predicción de pozos nuevos

**Beneficios:**
1. **Eficiencia de Memoria**: Solo se almacenan estrategias de transformación, no objetos ajustados
2. **Mejor Predicción de Pozos Nuevos**: Los escaladores frescos se ajustan a la distribución real de datos del pozo nuevo
3. **Precisión Mejorada**: Cada pozo obtiene escaladores optimizados para sus características específicas de datos
4. **Pipeline Simplificado**: Separación más limpia entre determinación de estrategia y ajuste de escaladores

### Escalado Global vs Local
Nuestro pipeline de normalización implementa un enfoque de escalado dual con detección automática:

- **Escalado Global**: Aplicado a columnas automáticamente detectadas como teniendo ratios altos de pozos fallidos (≥ min_failed_wells_ratio). El escalador global se entrena una vez usando datos de todos los pozos, asegurando un tratamiento consistente de estas características en todo el conjunto de datos.

- **Escalado Local**: Aplicado pozo por pozo a columnas con ratios bajos de pozos fallidos usando tipos de transformación almacenados. Se ajustan escaladores frescos para cada pozo usando la estrategia de transformación apropiada, preservando patrones específicos del pozo mientras estandariza escalas dentro de cada pozo.

Esta separación nos permite equilibrar entre mantener el contexto global (importante para características geográficas y de metadatos) y respetar características específicas del pozo (críticas para mediciones petrofísicas).

### Resultados de Detección Automática
El sistema de detección automática típicamente identifica los siguientes patrones:

**Comúnmente Detectados como Globales:**
- Coordenadas de Latitud/Longitud (100% de los pozos usualmente fallan el umbral de varianza)
- Identificadores de pozos y metadatos
- Marcadores geológicos constantes o casi constantes

**Comúnmente Detectados como Por Pozo:**
- Mediciones petrofísicas (GR, SP, RHOB, registros de resistividad, etc.)
- Características ingenieriles derivadas de curvas de registros de pozos
- Mediciones específicas de formación

### Estrategia de Codificación de Formaciones
La columna 'Formation' recibe un tratamiento especial en nuestro pipeline:

- Se codifica globalmente usando LabelEncoder, ignorando valores etiquetados como 'unknown'.
- Las formaciones no vistas durante el entrenamiento se asignan a un unknown_index especial.
- Este unknown_index se propaga a lo largo del pipeline y se respeta en métricas personalizadas e inferencia.

Este enfoque asegura una codificación de formación consistente entre pozos mientras maneja adecuadamente formaciones no vistas previamente durante la predicción, lo cual es esencial para el despliegue práctico en nuevos pozos.

### Manejo de Valores Negativos y Asimetría
Algunos registros petrofísicos contienen valores negativos o exhiben distribuciones asimétricas. Nuestro enfoque integrado para abordar estos problemas incluye:

- **Valores Negativos**: Para registros como SP con lecturas negativas, aplicamos transformaciones matemáticas que pueden manejar valores negativos, como la transformación Yeo-Johnson, para ajustar los datos.

- **Asimetría**: Muchos registros tienen distribuciones asimétricas. Usamos transformaciones como Yeo-Johnson (para datos con valores negativos) o transformación Box-Cox (para datos positivos) para reducir la asimetría y hacer que los datos sean más normalmente distribuidos.

- **Decisión Combinada**: Al abordar los valores negativos y la asimetría dentro de nuestro proceso de normalización, mejoramos la idoneidad de los datos para algoritmos de aprendizaje automático, que a menudo funcionan mejor con entradas normalmente distribuidas.

### Controles de Calidad y Manejo de Errores
Nuestro pipeline de normalización incluye varios mecanismos de control de calidad:

- **Validación de NaN e Inf**: Las columnas con valores NaN o Inf después de la transformación son identificadas y abordadas. El pipeline asegura que todos los datos normalizados estén limpios y utilizables.

- **Documentación de Pozos Fallidos**: El sistema rastrea y registra qué pozos fallan los umbrales de varianza para cada columna, proporcionando transparencia en el proceso de toma de decisiones.

- **Alineación de Columnas**: El pipeline verifica que todos los pozos tengan las mismas columnas finales antes de la concatenación, asegurando consistencia en el conjunto de datos normalizado.

- **Manejo de Datos Faltantes**: El sistema maneja adecuadamente curvas faltantes, asegurando que el pipeline no se rompa cuando ciertas mediciones estén ausentes en algunos pozos.

- **Rastreo de Errores de Ajuste**: Todas las fallas de transformación se registran con información detallada de errores para depuración y aseguramiento de calidad.

Estos controles de calidad mejoran la robustez de nuestro proceso de normalización, haciéndolo más confiable en entornos de producción.

### Integración de Decisiones en un Pipeline de Procesamiento
Para asegurar consistencia y eficiencia, integramos nuestros pasos de normalización y decisiones en un pipeline de procesamiento. Este enfoque:

- **Asegura Consistencia**: Cada muestra de datos pasa por los mismos pasos de preprocesamiento en el mismo orden, manteniendo uniformidad en todo el conjunto de datos.

- **Mejora la Reproducibilidad**: Los pipelines ayudan a mantener la reproducibilidad, que es esencial para validar modelos y comparar resultados entre diferentes ejecuciones o conjuntos de datos.

- **Mejora la Eficiencia Operativa**: En entornos operativos, los pipelines simplifican el flujo de trabajo, facilitando el procesamiento de grandes conjuntos de datos típicos en la ingeniería petrolera.

- **Elimina la Intervención Manual**: El sistema de detección automática elimina la necesidad de que expertos de dominio especifiquen manualmente qué columnas deben tratarse globalmente.

Al combinar estas decisiones en una estrategia integrada, abordamos efectivamente los desafíos planteados por las características de nuestros datos y consideraciones operativas.

## Explicación Matemática de los Métodos de Escalado
Para implementar nuestra estrategia de normalización, utilizamos varios métodos de escalado y transformaciones matemáticas. Cada uno tiene ventajas específicas y es adecuado para diferentes características de datos.

### Estandarización (StandardScaler)
**Fórmula**:

Para cada característica $x$, el valor estandarizado $z$ se calcula como:

$z = \frac{x - \mu}{\sigma}$

- $\mu$: Media de los valores de la característica en los datos de entrenamiento.
- $\sigma$: Desviación estándar de los valores de la característica en los datos de entrenamiento.

**Ventajas y Utilidad**:

- Centra los Datos: Transforma los datos para tener una media de cero.
- Escala la Varianza: Ajusta los datos para tener una desviación estándar de uno.
- Preserva Valores Atípicos: No limita ni restringe valores extremos.
- Adecuado Para: Características que se distribuyen aproximadamente de forma normal.

**Aplicación en Nuestros Datos**:

La estandarización se aplica después de abordar la asimetría y los valores negativos para asegurar que todas las características contribuyan por igual al entrenamiento del modelo.

### Escalado Min-Max (MinMaxScaler)
**Fórmula**:

Para cada característica $x$, el valor escalado $x_{scaled}$ se calcula como:

$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$

- $x_{min}$: Valor mínimo de la característica en los datos de entrenamiento.
- $x_{max}$: Valor máximo de la característica en los datos de entrenamiento.

**Ventajas y Utilidad**:

- Escala los Datos a un Rango Fijo: Típicamente [0, 1].
- Preserva la Distribución Original: No cambia la forma de la distribución de datos.
- Sensible a Valores Atípicos: Los valores extremos pueden sesgar el escalado.

**Aplicación en Nuestros Datos**:

No usamos principalmente el escalado min-max debido a su sensibilidad a los valores atípicos y la presencia de valores extremos en datos petrofísicos.

### Escalado Robusto (RobustScaler)
**Fórmula**:

Para cada característica $x$, el valor escalado $x_{scaled}$ se calcula como:

$x_{scaled} = \frac{x - median(x)}{IQR(x)}$

- $median(x)$: Mediana de los valores de la característica en los datos de entrenamiento.
- $IQR(x)$: Rango intercuartílico (percentil 75 - percentil 25) de los valores de la característica.

**Ventajas y Utilidad**:

- Robusto a Valores Atípicos: Usa mediana e IQR, que no se ven afectados por valores extremos.
- Preserva la Distribución de Datos: Mantiene el espaciado relativo de valores.

**Aplicación en Nuestros Datos**:

Usamos escalado robusto para características numéricas que no requieren transformación de distribución pero necesitan protección contra valores atípicos.

### Transformaciones de Potencia
Las transformaciones de potencia tienen como objetivo estabilizar la varianza y hacer que los datos se distribuyan más normalmente.

#### Transformación Yeo-Johnson
**Fórmula**:

Para cada valor $x$, el valor transformado $T(x;\lambda)$ es:

$T(x;\lambda) = \begin{cases}
    \frac{(x + 1)^\lambda - 1}{\lambda}, & \text{si } \lambda \neq 0, x \geq 0 \\
    \ln(x + 1), & \text{si } \lambda = 0, x \geq 0 \\
    \frac{-((-x + 1)^{2-\lambda} - 1)}{2-\lambda}, & \text{si } \lambda \neq 2, x < 0 \\
    -\ln(-x + 1), & \text{si } \lambda = 2, x < 0
\end{cases}$

- $\lambda$: Parámetro estimado para maximizar la normalidad de los datos transformados.

**Ventajas y Utilidad**:

- Maneja Valores Negativos: A diferencia de Box-Cox, puede transformar datos con valores cero o negativos.
- Reduce la Asimetría: Hace que los datos sean más simétricos y similares a una distribución normal.
- Flexible: Se adapta a los datos estimando el $\lambda$ óptimo.

**Aplicación en Nuestros Datos**:

Usamos la transformación Yeo-Johnson (implementada como 'power_robust') para registros como SP que tienen valores negativos y distribuciones asimétricas.

#### Transformación Box-Cox
**Fórmula**:

Para cada valor positivo $x$, el valor transformado $T(x;\lambda)$ es:

$T(x;\lambda) = \begin{cases}
    \frac{x^\lambda - 1}{\lambda}, & \text{si } \lambda \neq 0 \\
    \ln(x), & \text{si } \lambda = 0
\end{cases}$

- $x > 0$: Requiere que todos los datos sean positivos.
- $\lambda$: Parámetro estimado para maximizar la normalidad de los datos transformados.

**Ventajas y Utilidad**:

- Reduce la Asimetría: Transforma variables dependientes no normales en una forma normal.
- Mejora la Linealidad: Puede ayudar a modelar relaciones de manera más efectiva.

**Aplicación en Nuestros Datos**:

Aplicamos la transformación Box-Cox (implementada como 'boxcox_robust') a registros con valores positivos que exhiben alta asimetría para reducir su asimetría y aproximar la normalidad.

### Transformación Logarítmica (FunctionTransformer)
**Fórmula**:

Para cada valor positivo $x$, el valor transformado es:

$x_{transformed} = \ln(x)$

**Ventajas y Utilidad**:

- Reduce la Asimetría a la Derecha: Efectiva para datos con patrones de crecimiento exponencial.
- Maneja Amplio Rango de Valores: Comprime los valores grandes más que los pequeños.

**Aplicación en Nuestros Datos**:

Podemos usar la transformación logarítmica para características donde una relación logarítmica es apropiada, pero se debe tener cuidado ya que no puede manejar valores cero o negativos.

## Conclusión
La normalización es un paso vital en la preparación de datos petrofísicos para análisis y aprendizaje automático. Al adoptar una **estrategia de detección automática basada en varianza** con evaluación por pozo y umbrales de ratio de pozos fallidos, hemos eliminado la necesidad de especificación manual de columnas globales mientras mejoramos el rendimiento y la capacidad de generalización de nuestros modelos.

Nuestro enfoque está influenciado por las realidades prácticas de las operaciones petroleras, reconociendo la variabilidad inherente en los datos recolectados de diferentes pozos bajo condiciones variables. Al implementar análisis sofisticado de varianza por pozo y almacenar estrategias de transformación en lugar de escaladores ajustados, desarrollamos modelos que son robustos y aplicables a través de una variedad de escenarios operativos.

A través de esta estrategia integrada, aseguramos que nuestros datos estén listos para un análisis efectivo, apoyando en última instancia una mejor toma de decisiones en la exploración y producción petrolera.

## Puntos Clave
- **Detección Automática de Columnas**: El nuevo sistema elimina la especificación manual de columnas globales a través de evaluación sofisticada de varianza por pozo con umbrales configurables de ratio de pozos fallidos.

- **Evaluación de Varianza por Pozo**: La evaluación individual de pozos proporciona clasificación de columnas más precisa comparada con análisis de varianza agregado, asegurando tratamiento óptimo para cada característica.

- **Almacenamiento de Estrategias de Transformación**: Almacenar tipos de transformación en lugar de escaladores ajustados mejora dramáticamente la eficiencia de memoria y la precisión de predicción de pozos nuevos.

- **Umbral de Ratio de Pozos Fallidos**: El parámetro configurable `min_failed_wells_ratio` (por defecto: 0.5) permite ajuste fino del límite de decisión global vs por pozo basado en características del conjunto de datos.

- **Predicción Mejorada de Pozos Nuevos**: Los escaladores frescos ajustados a datos de pozos nuevos usando estrategias de transformación almacenadas proporcionan mejor precisión de predicción que escaladores pre-ajustados.

- **Manejo Robusto de Errores**: Rastreo comprehensivo de errores y mecanismos de respaldo aseguran confiabilidad del pipeline en entornos de producción.

- **Las Realidades Operativas Importan**: El sistema se adapta automáticamente a la variabilidad de herramientas de registro, diversidad geológica y condiciones operativas sin intervención manual.

- **Consistencia a Través de Automatización**: El sistema de detección automática asegura aplicación consistente de estrategias de normalización a través de diferentes conjuntos de datos y escenarios operativos.

Al entender e implementar estos principios y técnicas matemáticas, podemos aprovechar al máximo nuestros datos petrofísicos, llevando a modelos más precisos y mejores conocimientos en nuestros esfuerzos de ingeniería petrolera. 