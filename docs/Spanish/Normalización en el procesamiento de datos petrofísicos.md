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

### Estrategia de Normalización Basada en Columnas
Cada columna en el conjunto de datos se evalúa globalmente para determinar la técnica de normalización más apropiada. Esta técnica luego se aplica localmente pozo por pozo, excepto en casos donde se detecta baja varianza global o cuando una señal actúa como identificador de pozo.

Este enfoque asegura que cada curva sea tratada consistentemente en todos los pozos, respetando las diferencias operativas sin comprometer la consistencia del modelo. Al analizar las características de cada feature globalmente pero aplicando transformaciones localmente, mantenemos la integridad de los patrones específicos del pozo mientras permitimos comparaciones entre pozos.

### Reglas de Decisión Basadas en Varianza
Para cada columna en nuestro conjunto de datos, aplicamos reglas específicas basadas en el análisis de varianza:

- **Baja Varianza Global (< 1e-3)**: Las columnas con varianza global extremadamente baja se eliminan completamente ('drop'). Estas características proporcionan poca o ninguna información para el modelo y pueden introducir ruido.

- **Baja Varianza por Pozo pero Variando Entre Pozos**: Las columnas que muestran baja varianza dentro de pozos individuales pero cambian significativamente entre pozos se marcan como 'global_static' y se normalizan una vez usando un global_scaler. Esto preserva las diferencias entre pozos mientras estandariza la escala general.

- **Variabilidad Suficiente**: Las columnas con variabilidad adecuada se procesan utilizando la técnica más apropiada (robust, power_robust, boxcox_robust) pozo por pozo. Esto permite un manejo personalizado de la distribución de cada característica.

Este enfoque basado en varianza asegura que cada característica sea tratada de acuerdo a sus propiedades estadísticas, optimizando la información que puede proporcionar al modelo.

La siguiente tabla resume las reglas de decisión implementadas en nuestro código:

| Condición | Acción Aplicada | Implementación |
|-----------|----------------|----------------|
| Varianza global < 1e-3 | 'drop' | La característica se excluye del conjunto de datos |
| Baja varianza dentro de pozos pero variando entre pozos | 'global_static' | Escalado una vez con global_scaler |
| Contiene valores negativos con asimetría | 'power_robust' | Pipeline Yeo-Johnson + RobustScaler |
| Valores positivos con alta asimetría | 'boxcox_robust' | Pipeline Box-Cox + RobustScaler |
| Distribución similar a la normal | 'robust' | Solo RobustScaler |
| Datos categóricos | 'categorical' | Se aplica OrdinalEncoder |

Estos tipos de transformadores corresponden directamente a la implementación en la función `determine_global_transformer_types()` de nuestro código base y la lógica de transformación en `SimpleColumnTransformer`. Esta alineación entre la documentación y el código asegura que la estrategia descrita refleje con precisión la implementación real.

### Escalado Global vs Local
Nuestro pipeline de normalización implementa un enfoque de escalado dual:

- **Escalado Global**: Aplicado a columnas como Well_ID, Latitude, Longitude, y columnas identificadas como 'global_static'. El global_scaler se entrena una vez usando datos de todos los pozos, asegurando un tratamiento consistente de estas características en todo el conjunto de datos.

- **Escalado Local**: Aplicado pozo por pozo a las columnas restantes basado en la técnica seleccionada. Esto preserva patrones específicos del pozo mientras estandariza escalas dentro de cada pozo.

Esta separación nos permite equilibrar entre mantener el contexto global (importante para características geográficas y estáticas) y respetar características específicas del pozo (críticas para mediciones petrofísicas).

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

### Exclusión de Identificadores de Pozo como Características
Incluir identificadores de pozo (como nombres de pozo o IDs) como características en nuestros modelos podría introducir sesgo y dificultar la generalización. Nuestra decisión es:

- **Excluir Identificadores de Pozo**: No incluimos identificadores de pozo como características en nuestros modelos.

- **Fundamento**: Los pozos nuevos, que el modelo no ha visto antes, no tendrían identificadores correspondientes, haciendo que esta característica sea inútil para la predicción. Al excluir los identificadores de pozo, aseguramos que el modelo se centre en las mediciones petrofísicas mismas.

- **Beneficio Operativo**: Esta decisión se alinea con la necesidad práctica de modelos que puedan aplicarse a nuevos pozos sin requerir ajustes específicos del pozo o información.

### Controles de Calidad
Nuestro pipeline de normalización incluye varios mecanismos de control de calidad:

- **Validación de NaN e Inf**: Las columnas con valores NaN o Inf después de la transformación son identificadas y abordadas. El pipeline asegura que todos los datos normalizados estén limpios y utilizables.

- **Documentación de Baja Varianza**: Las columnas con baja varianza son documentadas, y su manejo (exclusión o transformación) es rastreado para transparencia.

- **Alineación de Columnas**: El pipeline verifica que todos los pozos tengan las mismas columnas finales antes de la concatenación, asegurando consistencia en el conjunto de datos normalizado.

- **Manejo de Datos Faltantes**: El sistema maneja adecuadamente curvas faltantes, asegurando que el pipeline no se rompa cuando ciertas mediciones estén ausentes en algunos pozos.

Estos controles de calidad mejoran la robustez de nuestro proceso de normalización, haciéndolo más confiable en entornos de producción.

### Integración de Decisiones en un Pipeline de Procesamiento
Para asegurar consistencia y eficiencia, integramos nuestros pasos de normalización y decisiones en un pipeline de procesamiento. Este enfoque:

- **Asegura Consistencia**: Cada muestra de datos pasa por los mismos pasos de preprocesamiento en el mismo orden, manteniendo uniformidad en todo el conjunto de datos.

- **Mejora la Reproducibilidad**: Los pipelines ayudan a mantener la reproducibilidad, que es esencial para validar modelos y comparar resultados entre diferentes ejecuciones o conjuntos de datos.

- **Mejora la Eficiencia Operativa**: En entornos operativos, los pipelines simplifican el flujo de trabajo, facilitando el procesamiento de grandes conjuntos de datos típicos en la ingeniería petrolera.

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
La normalización es un paso vital en la preparación de datos petrofísicos para análisis y aprendizaje automático. Al adoptar una estrategia de normalización basada en columnas con reglas de decisión basadas en varianza y manejo apropiado de características especiales como Formation, mejoramos el rendimiento y la capacidad de generalización de nuestros modelos.

Nuestro enfoque está influenciado por las realidades prácticas de las operaciones petroleras, reconociendo la variabilidad inherente en los datos recolectados de diferentes pozos bajo condiciones variables. Al centrarnos en los datos mismos y aplicar métodos de escalado adecuados, desarrollamos modelos que son robustos y aplicables a través de una variedad de escenarios operativos.

A través de esta estrategia integrada, aseguramos que nuestros datos estén listos para un análisis efectivo, apoyando en última instancia una mejor toma de decisiones en la exploración y producción petrolera.

## Puntos Clave
- **Normalización Basada en Columnas**: Cada característica se evalúa globalmente pero se normaliza localmente, equilibrando consistencia con características específicas del pozo.

- **Decisiones Basadas en Varianza**: Las características se procesan de manera diferente según sus propiedades de varianza, asegurando la extracción óptima de información.

- **Escalado Global vs Local**: Diferentes enfoques de escalado para identificadores de pozo/características estáticas versus mediciones petrofísicas optimizan el equilibrio entre contexto global y patrones locales.

- **Codificación de Formaciones**: El manejo especial asegura una representación consistente de formaciones geológicas mientras gestiona adecuadamente valores desconocidos.

- **Controles de Calidad**: Mecanismos de validación robustos aseguran datos normalizados limpios y consistentes en todos los pozos.

- **Las Realidades Operativas Importan**: Considerar el impacto de las herramientas de registro, diversidad geológica y condiciones operativas en su estrategia de preprocesamiento de datos.

- **La Consistencia es Esencial**: Integrar la normalización en un pipeline de procesamiento asegura la aplicación consistente de pasos de preprocesamiento, apoyando la reproducibilidad y eficiencia.

Al entender e implementar estos principios y técnicas matemáticas, podemos aprovechar al máximo nuestros datos petrofísicos, llevando a modelos más precisos y mejores conocimientos en nuestros esfuerzos de ingeniería petrolera. 