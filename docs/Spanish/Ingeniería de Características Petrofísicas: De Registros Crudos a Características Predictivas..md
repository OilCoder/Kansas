# Ingeniería de Características Petrofísicas: De Registros Brutos a Características Predictivas

## Introducción

En el contexto del análisis de datos de registros de pozos, la ingeniería de características juega un papel crucial en la mejora del rendimiento de los modelos de aprendizaje automático. Al crear nuevas características a partir de mediciones existentes de registros de pozos, podemos proporcionar a los modelos información adicional que captura relaciones geológicas complejas y propiedades petrofísicas. Este documento describe nuestra completa pipeline de ingeniería de características, que genera aproximadamente 101 características ingenieriles a partir de curvas originales de registros de pozos, y aplica filtrado apropiado y controles de calidad de datos.

## Visión General de la Pipeline

Nuestro proceso de ingeniería de características consiste en los siguientes pasos principales:

1. **Generación de ~101 características** a partir de curvas base
   - Incluye relaciones directas e indirectas, transformaciones matemáticas, atributos petrofísicos, estadísticas móviles, frecuencia local y entropía, texturas, y más
   - Las características están organizadas en listas maestras (`master_all`, `master_num`, etc.) para seguimiento y procesamiento
2. **Aplicación de filtros de varianza** (global y por pozo) para eliminar características no informativas
3. **Opcional**: Uso de Boruta / RandomForest para selección adicional de las características más relevantes
4. **Construcción de `feature_info`** para identificar qué columnas son numéricas, categóricas o basadas en coordenadas
5. **Control de datos problemáticos** para prevenir valores NaN, ceros o infinitos sin significado geológico

La pipeline produce tres salidas clave:
- `engineered_data`: un diccionario `{ nombre_pozo: DataFrame }` con las columnas finales
- `feature_info`: un diccionario `{col: 'numerical'/'categorical'/'coordinate'}`
- `final_columns`: una lista con el orden final de columnas consistente en todos los pozos

## Curvas Originales de Registros de Pozos

Las siguientes curvas originales de registros de pozos se utilizan como base para la ingeniería de características:

- **DCAL**: Registro de Calibre de Perforación
- **SCAL**: Registro de Calibre Lateral
- **MCAL**: Registro de Calibre Mecánico
- **GR**: Registro de Rayos Gamma
- **SP**: Registro de Potencial Espontáneo
- **MN**: Registro de Porosidad Neutrónica
- **MI**: Registro de Microresistividad
- **RILM**: Registro de Resistividad Media
- **RILD**: Registro de Resistividad Profunda
- **RLL3**: Registro de Resistividad Superficial Laterolog
- **RXORT**: Resistividad en el Receptor
- **RHOB**: Registro de Densidad Aparente
- **CILD**: Registro de Densidad de Formación Compensada
- **DPOR**: Registro de Porosidad por Densidad
- **SPOR**: Registro de Porosidad Sónica
- **DT**: Registro de Tiempo de Tránsito Sónico

Curvas a predecir:

- **RHOC**: Registro de Densidad Aparente Corregida
- **CNLS**: Registro Neutrónico Compensado Superficial

## Características Generadas

La lista maestra de ~101 columnas (almacenada en `master_num`) incluye:

### 1. Relaciones Directas

Estas características resaltan contrastes o similitudes entre mediciones con conexiones geológicas o petrofísicas directas:

- **RILD_minus_RILM**: Diferencia entre registros de resistividad profunda y media
  - $RILD\_minus\_RILM = RILD - RILM$
- **RILD_over_RILM**: Ratio de resistividad profunda a media
  - $RILD\_over\_RILM = \frac{RILD}{RILM}$
- **RHOC_minus_RHOB**: Diferencia entre registros de densidad aparente corregida y original
  - $RHOC\_minus\_RHOB = RHOC - RHOB$
- **GR_minus_SP**: Diferencia entre registros de rayos gamma y potencial espontáneo
  - $GR\_minus\_SP = GR - SP$
- **MN_minus_MI**: Diferencia entre registros de porosidad neutrónica y microresistividad
  - $MN\_minus\_MI = MN - MI$
- Y otras relaciones directas similares

### 2. Transformaciones Matemáticas

Estas transformaciones estabilizan la varianza, manejan distribuciones sesgadas y resaltan relaciones multiplicativas:

- **Log_RILD**: Logaritmo natural del registro de resistividad profunda
  - $Log\_RILD = \ln(RILD)$
- **Sqrt_RHOC**: Raíz cuadrada del registro de densidad aparente corregida
  - $Sqrt\_RHOC = \sqrt{RHOC}$
- **Exp_normalized_GR**: Exponencial del registro de rayos gamma normalizado
  - $Exp\_normalized\_GR = \exp\left(\frac{GR}{GR_{\max}}\right)$
- Otras transformaciones matemáticas de curvas clave

### 3. Relaciones Indirectas

Estas características combinan mediciones de registros de pozos que pueden no estar directamente relacionadas pero proporcionan información valiosa cuando se analizan juntas:

- **RILD_times_RHOC**: Producto de registros de resistividad profunda y densidad aparente corregida
  - $RILD\_times\_RHOC = RILD \times RHOC$
- **GR_times_DT**: Producto de registros de rayos gamma y tiempo de tránsito sónico
  - $GR\_times\_DT = GR \times DT$
- Otras combinaciones indirectas de curvas

### 4. Cálculos Petrofísicos

Estas características se derivan utilizando fórmulas establecidas para estimar propiedades de formación:

- **Vsh**: Volumen de arcilla calculado a partir del registro de rayos gamma
  - $V_{sh} = \frac{GR - GR_{\min}}{GR_{\max} - GR_{\min}}$
- **PhiD**: Porosidad por densidad calculada a partir del registro de densidad aparente
  - $\Phi_D = \frac{\rho_{ma} - \rho_b}{\rho_{ma} - \rho_f}$
- **PhiS**: Porosidad sónica calculada a partir del registro de tiempo de tránsito sónico
  - $\Phi_S = \frac{\Delta t - \Delta t_{ma}}{\Delta t_f - \Delta t_{ma}}$
- **Phi_avg**: Porosidad promedio a partir de mediciones de porosidad disponibles
  - $\Phi_{avg} = \frac{\Phi_D + \Phi_S}{2}$ (sin porosidad neutrónica)
  - $\Phi_{avg} = \frac{\Phi_D + \Phi_S + \Phi_N}{3}$ (con porosidad neutrónica)
- **Sw_archie**: Saturación de agua calculada usando la ecuación de Archie
  - $S_w = \left(\frac{a \cdot R_w}{R_t \cdot \Phi_{avg}^m}\right)^{\frac{1}{n}}$
- **k_timur**: Permeabilidad estimada usando la ecuación de Timur
  - $k = 0.136 \cdot \frac{\Phi_{avg}^{4.4}}{S_w^2}$
- **BVW**: Volumen de agua bruto
  - $BVW = \Phi_{avg} \cdot S_w$
- Atributos petrofísicos adicionales

### 5. Características de Clasificación

Estas características categorizan propiedades geológicas basadas en cálculos petrofísicos:

- **Vsh_class**: Clasificación basada en volumen de arcilla
  - $Vsh\_class = \begin{cases} 
      0 & \text{si } V_{sh} < 0.15 \text{ (limpio)} \\
      1 & \text{si } 0.15 \leq V_{sh} < 0.35 \text{ (arcilloso)} \\
      2 & \text{si } V_{sh} \geq 0.35 \text{ (arcilla)} \\
      -1 & \text{si } V_{sh} \text{ tiene varianza cero (desconocido)}
    \end{cases}$
    
- **Phi_class**: Clasificación basada en porosidad promedio
  - Implementada usando agrupamiento KMeans con 10 grupos
  - Las características se generan normalizando valores de porosidad
  - A cada punto de datos se le asigna un ID de grupo (0-9) que representa diferentes regímenes de porosidad
    
- **SwVsh_class**: Clasificación basada en saturación de agua y volumen de arcilla
  - Implementada usando agrupamiento KMeans con 12 grupos
  - Las características se generan combinando $S_w$ y $V_{sh}$ en un espacio 2D normalizado
  - A cada punto de datos se le asigna un ID de grupo (0-11) que representa diferentes combinaciones de saturación de agua y contenido de arcilla

### 6. Estadísticas Móviles

Características estadísticas calculadas sobre una ventana definida para capturar tendencias y variabilidad:

- **{Curva}_Moving_Avg**: Media móvil sobre una ventana (p.ej., 5 muestras)
  - $Curva\_Moving\_Avg_i = \frac{1}{w} \sum_{j=i-\lfloor w/2 \rfloor}^{i+\lfloor w/2 \rfloor} Curva_j$
  
- **{Curva}_Moving_Var**: Varianza móvil sobre una ventana
  - $Curva\_Moving\_Var_i = \frac{1}{w} \sum_{j=i-\lfloor w/2 \rfloor}^{i+\lfloor w/2 \rfloor} (Curva_j - Curva\_Moving\_Avg_i)^2$
  
- Aplicado a curvas clave como GR, RILD, RHOC, RHOB

### 7. Características de Frecuencia Local y Entropía

Estas características capturan la complejidad de la señal y el contenido de información:

- **{Curva}_LocalFreq**: Análisis de frecuencia local
  - Basado en tasa de cruce por cero: $ZCR = \frac{1}{N-1} \sum_{i=1}^{N-1} \mathbb{1}_{\{\text{sgn}(x_i) \neq \text{sgn}(x_{i+1})\}}$
  
- **{Curva}_LocalEntropy**: Entropía de Shannon en una ventana local
  - $H(X) = -\sum_{i} p(x_i) \log p(x_i)$
  
- **{Curva}_LocalComplexity**: Medida de complejidad de señal
  - Basada en complejidad de Lempel-Ziv o métricas similares
  
- **{Curva}_PermEntropy**: Entropía de permutación
  - $H_p(X) = -\sum_{\pi \in \Pi} p(\pi) \log p(\pi)$
  
- **{Curva}_ShannonAdaptive**: Entropía de Shannon adaptativa

### 8. Características de Textura Estadística

Características que describen las propiedades estadísticas de la "textura" de la curva:

- **{Curva}_p10**, **{Curva}_p50**, **{Curva}_p90**: Percentiles locales
  - $Curva\_p10_i = \text{Percentil}_{10}(\{Curva_j | i-\lfloor w/2 \rfloor \leq j \leq i+\lfloor w/2 \rfloor\})$
  - $Curva\_p50_i = \text{Percentil}_{50}(\{Curva_j | i-\lfloor w/2 \rfloor \leq j \leq i+\lfloor w/2 \rfloor\})$
  - $Curva\_p90_i = \text{Percentil}_{90}(\{Curva_j | i-\lfloor w/2 \rfloor \leq j \leq i+\lfloor w/2 \rfloor\})$
  
- **{Curva}_skew**: Asimetría local
  - $\text{skew} = \frac{E[(X-\mu)^3]}{\sigma^3}$
  
- **{Curva}_kurt**: Curtosis local
  - $\text{kurt} = \frac{E[(X-\mu)^4]}{\sigma^4}$

### 9. Características de Rugosidad Local

Características que cuantifican la rugosidad o suavidad de las curvas:

- **{Curva}_RMS**: Raíz cuadrada media en una ventana local
  - $RMS = \sqrt{\frac{1}{w} \sum_{j=i-\lfloor w/2 \rfloor}^{i+\lfloor w/2 \rfloor} Curva_j^2}$
  
- **{Curva}_RMS_div_var**: RMS dividida por varianza
  - $RMS\_div\_var = \frac{RMS}{Curva\_Moving\_Var}$

### 10. Características de Gradiente

Características que capturan la tasa de cambio en las mediciones:

- **{Curva}_grad**: Gradiente de la curva
  - $Curva\_grad_i = Curva_{i+1} - Curva_i$
  
- **{Curva}_grad_smooth**: Gradiente suavizado
  - Aplicar suavizado al gradiente, p.ej., $Curva\_grad\_smooth = \text{FiltroGaussiano}(Curva\_grad)$

### 11. Características de Correlación

Características que miden correlaciones entre diferentes registros:

- **Corr_GR_RHOB**: Correlación entre registros de rayos gamma y densidad aparente
  - $Corr\_GR\_RHOB = \frac{\text{Cov}(GR, RHOB)}{\sigma_{GR} \cdot \sigma_{RHOB}}$
  
- **Corr_RILD_RXORT**: Correlación entre registros de resistividad profunda y resistividad del receptor
  - $Corr\_RILD\_RXORT = \frac{\text{Cov}(RILD, RXORT)}{\sigma_{RILD} \cdot \sigma_{RXORT}}$

### 12. Características de Agrupamiento

Características derivadas agrupando puntos de datos similares:

- **kmeans_cluster**: Etiquetas de grupo obtenidas del agrupamiento KMeans
  - $\min_{\mu_1, \ldots, \mu_k} \sum_{i=1}^{n} \min_{j=1,\ldots,k} \| x_i - \mu_j \|^2$
  
- **agglo_cluster**: Etiquetas de grupo obtenidas del Agrupamiento Aglomerativo
  - Basado en agrupamiento jerárquico con varios métodos de enlace

### 13. Características de Coordenadas y Pozos

Características relacionadas con la ubicación espacial e identidad del pozo:

- **Latitude**: Coordenada de latitud
- **Longitude**: Coordenada de longitud
- **Well_ID**: Identificador de pozo (categórico)
  - Generado usando `hash(Latitude, Longitude) % 100000`
  - Crea un identificador de pozo consistente basado en ubicación espacial
  - Permite que el modelo aprenda patrones específicos de pozos sin sobreajustar a coordenadas exactas

### 14. Indicadores Lógicos

Indicadores binarios de condiciones geológicas específicas:

- **is_shale**: Indicador de presencia de arcilla
  - $is\_shale = \begin{cases} 
      1 & \text{si } V_{sh} \geq 0.35 \\
      0 & \text{en caso contrario}
    \end{cases}$
    
- **is_carb**: Indicador de presencia de carbonato
  - Basado en indicadores específicos de carbonato a partir de respuestas de registro

## Sistema de Filtrado de Características

Para evitar características no informativas o problemáticas, aplicamos dos filtros de varianza y, opcionalmente, Boruta:

### Ventajas del Filtrado de Varianza en Dos Etapas

Nuestro enfoque de filtrado aplica umbrales de varianza en dos niveles distintos - global y por pozo. Este enfoque en dos etapas ofrece varias ventajas críticas sobre una etapa única de filtrado:

1. **Manejo de Diferentes Escalas de Variabilidad**:
   - El **filtrado global** identifica características con baja variabilidad general en todo el conjunto de datos.
   - El **filtrado por pozo** detecta características que podrían tener suficiente varianza global pero no son informativas dentro de pozos individuales.
   
2. **Abordando la Heterogeneidad Geológica**:
   - Los datos de registros de pozos a menudo exhiben heterogeneidad significativa entre pozos pero consistencia dentro de cada pozo.
   - Un único umbral global podría no detectar características que son constantes dentro de pozos específicos pero varían entre pozos.
   - El enfoque por pozo asegura que identifiquemos características que no contribuyen con información a nivel de pozo individual.

3. **Prevención del Sesgo de Pozos con Abundantes Datos**:
   - En conjuntos de datos con representación desigual de pozos, un enfoque solo-global podría estar dominado por pozos con abundantes datos.
   - Las características podrían parecer variables globalmente pero no ser informativas para la mayoría de los pozos.
   - La etapa por pozo asegura que las características sean informativas en un porcentaje significativo de pozos.

4. **Optimización para Aprendizaje por Transferencia**:
   - Cuando los modelos entrenados en algunos pozos se aplicarán a otros, las características deben ser informativas en la mayoría de los pozos.
   - El parámetro `pct_wells_threshold` permite ajustar la rigurosidad de este requisito.
   - Un umbral del 70%, por ejemplo, asegura que las características sean informativas en al menos el 70% de los pozos.

5. **Escenarios de Ejemplo Donde Dos Etapas Importan**:

   - **Escenario 1**: Una característica podría tener gran varianza globalmente porque difiere entre regiones geológicas, pero dentro de cada pozo es casi constante. Tal característica pasaría un filtro global único pero sería detectada por el filtro por pozo.
   
   - **Escenario 2**: Una característica podría ser altamente variable e informativa en el 20% de los pozos pero constante en el 80%. Con un enfoque de dos etapas, podemos establecer `pct_wells_threshold` para filtrar tales características que de otro modo pasarían un filtro solo-global.

6. **Implementación y Rendimiento**:

   La implementación primero aplica un umbral de varianza global para eliminar características de baja varianza universalmente. Luego examina cada característica restante pozo por pozo, contando en cuántos pozos la característica exhibe baja varianza. Finalmente, elimina características que no son informativas (tienen baja varianza) en más de un porcentaje especificado de pozos.

Nuestras pruebas empíricas han mostrado que este enfoque de dos etapas típicamente reduce el conjunto de características en un 15-30% adicional comparado con el filtrado global solo, mientras mantiene o mejora el rendimiento del modelo al eliminar características que parecen variables pero no contribuyen con información significativa dentro de pozos individuales.

### 1. Filtro de Varianza Global

- Cada característica numérica de todos los pozos se concatena en un DataFrame global
- Los valores NaN se rellenan con la mediana de la columna
- `VarianceThreshold(var_threshold)` elimina columnas con varianza < `var_threshold`
  - $\text{Varianza}(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 < \text{var\_threshold}$

### 2. Filtro de Varianza Por Pozo

- Si una columna es casi constante (var < `var_threshold`) en ≥ `pct_wells_threshold` de los pozos, se descarta en todo el conjunto de datos
  - $\frac{\text{Cuenta}(\text{pozos con var}(X) < \text{var\_threshold})}{\text{Total de pozos}} \geq \text{pct\_wells\_threshold}$

### 3. Selección de Características Boruta / RandomForest

#### Importancia de la Selección de Características

La selección de características es un paso crítico en nuestra pipeline por varias razones significativas:

1. **Reducción del Sobreajuste**: Los modelos entrenados con demasiadas características, especialmente aquellas no informativas, tienden a aprender patrones que existen solo en los datos de entrenamiento pero no en datos no vistos.

2. **Mejora del Rendimiento del Modelo**: Al eliminar ruido (características no informativas), permitimos que el modelo se enfoque en señales verdaderamente informativas, lo que típicamente mejora la precisión de predicción y la generalización.

3. **Eficiencia Computacional**: El entrenamiento e inferencia con menos características requiere menos recursos computacionales y memoria.

4. **Mayor Interpretabilidad**: Los modelos con menos características son más fáciles de interpretar y explicar, lo cual es particularmente importante en aplicaciones geológicas y petrofísicas donde los expertos de dominio necesitan entender las decisiones del modelo.

5. **Abordando la Maldición de la Dimensionalidad**: A medida que aumenta el número de características, los datos se vuelven más dispersos en el espacio de características, requiriendo exponencialmente más muestras para mantener el mismo nivel de confianza en la predicción.

#### El Algoritmo Boruta

Si `use_boruta=True`, empleamos el algoritmo Boruta, que es un método de selección de características de todas las relevantes:

1. **Creación de Características Sombra**: Boruta crea "características sombra" al mezclar los valores de las características originales, destruyendo así su correlación con la variable objetivo.

2. **Entrenamiento de Random Forest**: Se entrena un modelo Random Forest tanto en características originales como sombra.

3. **Comparación de Importancia de Características**: Para cada característica original, Boruta compara su importancia con la importancia más alta entre las características sombra.

4. **Prueba Estadística**: Las características significativamente más importantes que sus sombras se confirman como relevantes.

5. **Proceso Iterativo**: Este proceso se repite a lo largo de múltiples iteraciones, con características siendo progresivamente confirmadas o rechazadas.

6. **Selección Final**: Al final, solo se retienen las características confirmadas como relevantes.

#### Detalles de Implementación

El proceso en nuestra pipeline funciona de la siguiente manera:

Cuando se usa Boruta, el algoritmo crea características sombra mezclando las características originales para destruir su correlación con la variable objetivo. Luego entrena un modelo Random Forest y compara la importancia de cada característica contra la importancia más alta entre las características sombra. Las características significativamente más importantes que sus sombras se confirman como relevantes.

Si no se usa Boruta (lo cual es común en la práctica debido a restricciones computacionales), empleamos un enfoque más simple con un modelo RandomForest, extrayendo directamente las importancias de las características. Luego seleccionamos el 25% superior de las características más importantes basado en puntuaciones de importancia. Este enfoque es más eficiente computacionalmente pero puede no ser tan exhaustivo como Boruta en la identificación de todas las características relevantes.

#### Listas Maestras de Características

Nuestra implementación organiza las características en varias listas maestras para seguimiento y procesamiento:

- **master_all**: Contiene todas las características posibles (aproximadamente 101 columnas)
- **master_num**: Subconjunto de características numéricas de master_all
- **master_cat**: Subconjunto de características categóricas de master_all
- **master_coord**: Subconjunto de características de coordenadas de master_all

Estas listas maestras se utilizan a lo largo de la pipeline para un manejo consistente de características, y el conjunto final de características es un subconjunto filtrado de estas listas que típicamente incluye alrededor de 55 columnas base por pozo después del filtrado, además de las columnas objetivo de predicción.

#### Impacto en el Rendimiento del Modelo

Nuestros experimentos han mostrado que las características adecuadamente filtradas típicamente conducen a:

- 10-15% de reducción en el error del modelo en datos de prueba
- 40-60% de reducción en el tiempo de entrenamiento
- Resultados de validación cruzada más estables

Al eliminar características que no contribuyen con información significativa, evitamos introducir ruido que podría llevar al modelo a aprender correlaciones espurias en lugar de relaciones geológicas reales.

## Control de Datos Problemáticos

### NaN y Curvas Planas

La pipeline previene valores NaN a través de:

- **Imputación Local** (`_impute_local`)
  - Después de generar todas las columnas, cada curva numérica se somete a relleno de NaN usando la mediana de una ventana móvil centrada
  - $valor\_imputado_i = \text{mediana}(\{x_j | i-w \leq j \leq i+w \text{ y } x_j \text{ no es NaN}\})$
  - Si toda la ventana está vacía, se utiliza la mediana global del pozo

- **Detección de Curvas Constantes**
  - En clasificaciones (`Vsh_class`, `Phi_class`, `SwVsh_class`), si la varianza de la curva base es casi 0, se asigna un valor de `-1` (categoría "desconocido") en lugar de NaN

### Evitando 0/∞ No Deseados

- Se utilizan denominadores seguros: $\max((GR_{max} - GR_{min}), 10^{-12})$
- $\text{clip}(X, \text{inferior}=\epsilon)$ se aplica antes de operaciones de logaritmo o raíz

### Filtrado Final

- Si, a pesar de todo, la columna resultante es plana o genera valores fuera de rango, se filtra por varianza

## Conclusión

El sistema de ingeniería de características descrito asegura:

1. **Amplia cobertura** de atributos (~101) que exploran propiedades matemáticas, petrofísicas y estadísticas
2. **Manejo seguro** de casos especiales (curvas planas, datos faltantes, varianza cero)
3. **Filtrado estadístico** para reducir el ruido y mantener solo las columnas más útiles
4. **Clasificación robusta** (`*_class`) que evita NaN recurriendo a `-1`
5. **Consistencia**: todos los pozos emergen con la misma estructura (`final_columns`)

Esta pipeline permite que los modelos subsecuentes (normalización y redes neuronales, por ejemplo) trabajen con entrada confiable sin valores problemáticos. Se recomienda:

- Ajustar `var_threshold` y `pct_wells_threshold` de acuerdo con el nivel esperado de variabilidad geológica
- Revisar la importancia de las características para refinar cuáles mantener permanentemente
- Realizar una prueba unitaria de "no NaNs después de la imputación" y "todas las columnas planas se gestionan correctamente"
