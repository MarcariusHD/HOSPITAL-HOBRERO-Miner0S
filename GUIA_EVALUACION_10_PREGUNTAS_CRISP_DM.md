# 📋 Evaluación Grupal de Minería de Datos: Respuestas Oficiales (Fases 1 a 4 CRISP-DM)
### Proyecto: Análisis y Predicción de Demanda por Triaje en Urgencias Hospitalarias
**Institución:** Hospital Obrero N.º 1 de La Paz — Caja Nacional de Salud (CNS)  
**Equipo:** Marco Antonio Calderón Rojas, Nazareth Constantina Padilla Cáseres, Dilan José Paco Padilla, Daner Kathlen Lenny Escalante Sirpa, Hans Héctor Choque Quenta.

---

## BLOQUE 1: ENTENDIMIENTO DEL NEGOCIO

### Pregunta 1: ¿Qué problema concreto aborda su proyecto, qué decisión o acción busca apoyar y cuál es su criterio de éxito de negocio? Indiquen a quién afecta y cómo se conecta el análisis con esa necesidad.
* **Problema concreto:** En el Servicio de Urgencias del Hospital Obrero N.º 1 existe una saturación crítica de la capacidad operativa debido a una alta concurrencia de pacientes con patologías de baja complejidad (Triaje 4 y 5) que podrían resolverse en primer nivel o consulta ambulatoria. Esta congestión genera demoras que ponen en peligro potencial a pacientes con emergencias médicas de riesgo vital (Triaje 1, 2 y 3).
* **Decisión o acción que busca apoyar:** Implementar un **módulo de derivación rápida (*Fast-Track*)** en la admisión para canalizar a pacientes clasificados en **Triaje 4 y 5** hacia consultorios ambulatorios diurnos, liberando los boxes de atención y a los médicos especialistas para los pacientes de **Triaje 1, 2 y 3**.
* **A quién afecta:**
  1. *Pacientes graves (Triajes 1 al 3):* Riesgo de deterioro clínico o muerte por demoras en la atención inicial.
  2. *Pacientes no urgentes (Triajes 4 y 5):* Tiempos de espera excesivos en salas comunes (frustración e insatisfacción).
  3. *Personal médico y de enfermería:* Sobrecarga laboral, fatiga y estrés asistencial.
* **Criterio de éxito de negocio:** Lograr derivar al menos al **60%** de los pacientes de triaje 4 y 5 al circuito ambulatorio rápido en horarios de mayor afluencia (08:00 a 18:00 horas), y reducir en un **25%** el tiempo promedio de espera global en el servicio de urgencias.

---

### Pregunta 2: ¿Cuál es la métrica principal del proyecto y cómo se calcula? Presenten su fórmula, las variables y unidades involucradas, el valor observado hasta ahora y una acción concreta para mejorarla. Aclaren si es una métrica de negocio o de desempeño del modelo.
* **Aclaración conceptual:** Es una **métrica de negocio** denominada **Tasa de Demanda de Baja Complejidad en Urgencias (% Triaje 4 + 5)**. Mide la magnitud real del problema operativo, a diferencia de las métricas algorítmicas (Accuracy, Recall) que miden la precisión del modelo.
* **Fórmula:**
$$\text{Porcentaje Triaje 4+5} = \left( \frac{\text{Atenciones Nivel 4} + \text{Atenciones Nivel 5}}{\text{Total de Atenciones Válidas}} \right) \times 100$$
* **Variables y unidades involucradas:**
  * *Atenciones Nivel 4 (Menor):* $530,555$ episodios de atención médica (en base institucional) / $5,956$ episodios (en muestra de modelado).
  * *Atenciones Nivel 5 (No urgente):* $54,278$ episodios (base institucional) / $2,993$ episodios (en muestra de modelado).
  * *Total de Atenciones Válidas:* $1,001,766$ episodios (base institucional) / $15,000$ episodios (en muestra de modelado).
  * *Unidad:* Porcentaje (%).
* **Valor observado medido hasta ahora:** **58.38%** en la base institucional completa ($584,833$ pacientes no urgentes) y **59.66%** en el conjunto de datos de modelado ($8,949$ pacientes no urgentes).
* **Acción concreta para mejorarla:** Habilitar dos consultorios de triaje avanzado y derivación rápida hacia la red de policlínicos periféricos de la CNS en el horario diurno de máxima concentración (10:00 a 18:00 horas).

---

## BLOQUE 2: ENTENDIMIENTO Y PREPARACIÓN DE LOS DATOS

### Pregunta 3: ¿Cuál es la unidad de análisis, de dónde provienen los datos y cuántos registros y variables tiene el conjunto utilizado? Indiquen qué representa una fila y, si hubo filtros, cuántos registros quedaron después de aplicarlos.
* **Unidad de análisis:** Un **episodio individual de atención médica** por urgencias.
* **Qué representa una fila:** La llegada, registro administrativo y clasificación clínica de triaje de un asegurado que solicita atención médica en el hospital.
* **Origen de los datos:** Base de datos del Sistema de Admisiones del Hospital Obrero N.º 1 (`urgencias-hospitalarias-atendidas-20252026.xlsx` / `dataset_hospital_obredo_triaje.csv`).
* **Registros y variables del conjunto:**
  * Base institucional depurada: $1,001,766$ filas y $11$ variables.
  * Conjunto depurado para modelado de Minería de Datos: $15,000$ filas y $5$ variables clave (`Nivel de triaje`, `Edad`, `Sexo`, `Hora`, `Día`).
* **Filtros aplicados y registros conservados:**
  * Validación de escala de triaje admisible ($1, 2, 3, 4, 5$). Se descartaron registros nulos o sin asignación de nivel.
  * Validación de rango biológico razonable en edad ($0 \le \text{Edad} \le 120$).
  * Registros conservados tras depuración: **15,000 registros válidos (100% íntegros)** en la muestra de trabajo reproducible.

---

### Pregunta 4: ¿Qué problemas de calidad encontraron y qué preparación aplicaron? Informen la cantidad y las columnas con valores faltantes; mencionen también duplicados o inconsistencias relevantes y expliquen qué hicieron y por qué.
1. **Heterogeneidad de formatos en la variable `Hora`:** Existían cadenas horarias de texto (`'14:35'`) y marcas temporales mixtas. Se normalizó extrayendo la hora entera en escala discreta de $0$ a $23$ horas (`Hora_num`) para capturar el ciclo circadiano sin ruido de minutos.
2. **Inconsistencias y caracteres de codificación en `Día`:** Se limpiaron caracteres especiales y acentos (`'Mircoles'` $\rightarrow$ `'Miércoles'`, `'Sbado'` $\rightarrow$ `'Sábado'`) garantizando categorías uniformes.
3. **Valores nulos:** Se verificó la ausencia de nulos en el dataset depurado ($0$ en triaje, edad, sexo, hora y día). En la base macro, los valores de edad fuera de rango se imputaron con la **mediana** para mitigar la distorsión de colas largas.
4. **Codificación One-Hot Encoding:** Se aplicó `pd.get_dummies(..., drop_first=True)` sobre `Sexo` y `Día` para evitar la **trampa de la variable ficticia (*Dummy Variable Trap*)** y la multicolinealidad exacta en la regresión logística.
5. **Binarización del Target:** Transformación de la escala ordinal 1-5 a binaria: $0$ (Urgente: Triajes 1-3) y $1$ (No urgente: Triajes 4-5).

---

## BLOQUE 3: EXPLORACIÓN DE LOS DATOS (EDA)

### Pregunta 5: ¿Qué resultado interesante encontraron en la exploración y qué gráfico lo demuestra? Muéstrenlo con título y etiquetas, describan el patrón observado y expliquen por qué importa para el objetivo del proyecto.
* **Resultado y Patrón observado:**
  1. *Predominio masivo de patología ambulatoria:* El **59.66%** de los episodios corresponden a baja urgencia (Triaje 4 con 39.71% y Triaje 5 con 19.95%), mientras que las emergencias con riesgo vital (Triaje 1) son apenas el **4.97%** ($745$ casos).
  2. *Patrón de congestión circadiana (Horarios vs Gravedad):* Durante la madrugada (`00:00 - 06:00`), la proporción relativa de urgencias reales (Triaje 1-3) es significativamente mayor. Sin embargo, a partir de las `06:00` y hasta las `18:00` horas, se produce una avalancha de atenciones leves (Triaje 4-5), superando el **62%** de todas las admisiones diurnas.
* **Gráfico que lo demuestra:** Gráfico de *Volumen Total de Atenciones por Franja Horaria y Nivel de Triaje* (Cuaderno 2 / Cuaderno Optimizado, Celda de Pregunta 5).
* **Por qué importa para el objetivo del proyecto:** Demuestra que la saturación de urgencias es un fenómeno **estrictamente diurno y ambulatorio**. Esto justifica que el consultorio *Fast-Track* opere en horario de 08:00 a 18:00 horas, optimizando el recurso humano sin incrementar costos en guardias nocturnas.

---

## BLOQUE 4: MODELADO PREDICTIVO Y EVALUACIÓN

### Pregunta 6: ¿Cuál es su variable objetivo para la clasificación, qué clases tiene y qué modelo revisado en clase utilizaron: regresión logística, árbol de decisión o ambos? Expliquen brevemente por qué ese modelo es pertinente para su problema.
* **Variable objetivo:** `Objetivo_Triaje_45` (Binaria).
  * **Clase 0 (Urgente / Emergencia Crítica):** Triaje 1, 2 y 3 ($6,051$ pacientes, $40.34\%$).
  * **Clase 1 (No Urgente / Baja Complejidad):** Triaje 4 y 5 ($8,949$ pacientes, $59.66\%$).
* **Modelos utilizados:** Se implementaron **ambos modelos revisados en clase**:
  1. **Regresión Logística (`LogisticRegression`):** Pertinente porque modela la probabilidad cuantitativa posterior $P(Y=1|X)$ mediante la función sigmoide, permitiendo modular umbrales de decisión y calcular Odds Ratios ($e^\beta$) de riesgo relativo.
  2. **Árbol de Decisión (`DecisionTreeClassifier`):** Pertinente porque genera reglas de decisión visuales transparentes (*IF-THEN*), captura interacciones no lineales entre edad y horarios, y permite diseñar diagramas de flujo utilizables directamente por el personal de enfermería en ventanilla.

---

### Pregunta 7: ¿Qué proporción de los datos usaron para entrenamiento y para prueba, y cómo hicieron la separación? Indiquen el método o semilla si corresponde y cómo evitaron que información de prueba influyera en el entrenamiento o la preparación.
* **Proporción de partición:** **80% Entrenamiento (`Train`) y 20% Prueba (`Test`)**.
  * Muestra Train: $12,000$ registros.
  * Muestra Test: $3,000$ registros.
* **Método y Semilla:** Se utilizó `train_test_split` de Scikit-Learn con semilla fija **`random_state=42`** para garantizar reproducibilidad.
* **Prevención de fuga de información (*Data Leakage*):**
  1. Se utilizó **`stratify=y`**, manteniendo idéntica proporción de la variable objetivo ($59.67\%$ Clase 1 y $40.33\%$ Clase 0) en ambas particiones.
  2. La división se realizó antes del ajuste del modelo.
  3. Los estimadores de hiperparámetros (`GridSearchCV` con 5-Fold Stratified CV) se ejecutaron exclusivamente sobre el conjunto de entrenamiento (`X_train`), dejando a `X_test` como un conjunto ciego no visto.

---

### Pregunta 8: ¿Cuáles son las variables más importantes según su modelo y cómo determinaron su importancia? Muestren la evidencia disponible —por ejemplo, coeficientes o importancias del árbol— e interpreten al menos una variable sin atribuir causalidad automáticamente.
* **Determinación de importancia:**
  * En **Regresión Logística:** Mediante la magnitud de los **Coeficientes Beta ($\beta$)** y los **Odds Ratios ($OR = e^\beta$)**.
  * En **Árbol de Decisión:** Mediante la **Reducción de Impureza de Gini (*Feature Importance*)**.
* **Evidencia en Árbol de Decisión:**
  * `Edad`: **62.67%** de importancia relativa.
  * `Hora_num`: **37.33%** de importancia relativa.
  * `Sexo` y `Día`: **0.00%** (impacto marginal en nodos principales).
* **Interpretación prudente de la variable `Edad` ($\beta = -0.0122$, $OR = 0.9879$ en logística):**
  * Al tener coeficiente negativo ($OR < 1.0$), indica una **correlación estadística inversa**: por cada año adicional de edad en el paciente, los momios de ser clasificado como paciente no urgente disminuyen en un $1.21\%$ ($1 - 0.9879$). Dicho a la inversa: a mayor edad, mayor probabilidad relativa de ingresar en Triaje 1 a 3 (urgencia crítica).
  * *Sin atribuir causalidad automática:* No afirmamos que la edad avanzada "cause" directamente una emergencia, sino que los adultos mayores en la CNS concentran mayor carga de comorbilidades crónicas descompensadas (hipertensión, diabetes, cardiopatías) que demandan atención prioritaria.

---

### Pregunta 9: ¿Qué desempeño obtuvo el modelo en el conjunto de prueba? Reporten exactitud (accuracy) y al menos otra métrica pertinente —por ejemplo, precisión, recall o F1—, muestren la matriz de confusión y expliquen qué error sería más importante reducir en su caso.
* **Métricas en Conjunto de Prueba ($N = 3,000$ casos no vistos):**
  * **Regresión Logística (Optimizada con class_weight='balanced'):**
    * Exactitud (*Accuracy*): **49.37%**
    * Precisión (*Precision*, Clase 1 - Leve): **59.04%**
    * Sensibilidad (*Recall*, Clase 1 - Leve): **49.44%**
    * $F_1\text{-Score}$: **53.82%** | $AUC\text{-}ROC$: **0.497**
  * **Árbol de Decisión (Podado con GridSearchCV, profundidad 3):**
    * Exactitud (*Accuracy*): **59.63%**
    * Precisión (*Precision*, Clase 1 - Leve): **59.80%**
    * Sensibilidad (*Recall*, Clase 1 - Leve): **98.88%**
    * $F_1\text{-Score}$: **74.52%** | $AUC\text{-}ROC$: **0.486**
* **Matriz de Confusión (Regresión Logística en Test):**
$$\begin{pmatrix} VN = 596 & FP = 614 \\ FN = 905 & VP = 885 \end{pmatrix}$$
* **Matriz de Confusión (Árbol de Decisión en Test):**
$$\begin{pmatrix} VN = 19 & FP = 1,191 \\ FN = 20 & VP = 1,770 \end{pmatrix}$$

#### ¿Qué error sería más importante reducir en este caso médico?
* **Falso Negativo de levedad (Clasificar como urgente a un paciente leve):** Error conservador. El paciente esperará en salas de agudos consumiendo recursos, pero **su vida e integridad física están protegidas**.
* **FALSO POSITIVO DE LEVEDAD (Clasificar como leve a un paciente crítico - ERROR CRÍTICO):** Es el error más peligroso. Significa enviar a la fila de espera ambulatoria a una persona con infarto, paro inminente o hemorragia interna. **Consecuencia: Deterioro clínico acelerado o fallecimiento del asegurado.**
> **Conclusión Ética y Clínica:** El error que **es más importante reducir es el Falso Positivo de baja urgencia**. El modelo no debe actuar de forma autónoma: debe requerir siempre la validación de signos vitales por parte de enfermería antes de derivar al paciente.

---

### Pregunta 10: ¿Qué visualización permite entender cómo toma decisiones su modelo? Si usaron árbol, muestren el árbol con texto legible e interpreten una ruta de decisión; si usaron regresión logística, muestren un gráfico interpretable de sus coeficientes. Expliquen brevemente qué aporta la visualización.
* **Visualización del Árbol de Decisión (`plot_tree`):**
  * Presenta un árbol podado de profundidad máxima 3 con 8 nodos terminales coloreados por pureza de clase.
  * **Interpretación de una ruta de decisión:**
    $$\text{Nodo Raíz: } [\text{Edad} \le 18.5] \longrightarrow \text{Verdadero (Pediátrico)} \longrightarrow [\text{Hora\_num} \le 14.5] \longrightarrow \text{Clase 1 (No Urgente)}$$
    *Ruta:* Si el paciente tiene 18 años o menos y consulta antes de las 14:30 horas, el árbol lo clasifica con alta probabilidad en Triaje 4-5 (patología pediátrica leve o ambulatoria). Si el paciente es adulto mayor ($>60$ años) y llega en horario nocturno, la rama desciende hacia nodos de mayor gravedad clínica.
* **Visualización de Regresión Logística (Gráfico de Coeficientes Beta y Odds Ratios):**
  * Gráfico de barras horizontales con colores: verde para factores inductores de atención ambulatoria leve (`Día_Martes`, `Día_Sábado`, `Edad`) y rojo para predictores asociados a cuadros agudos (`Sexo_Mujer`, `Día_Viernes`, `Hora_num`).
* **Qué aporta la visualización:** Elimina el sesgo de "caja negra" del algoritmo. Permite a los médicos auditores y directores de la CNS entender la lógica matemática del modelo, aumentando la confianza para su implementación práctica en admisión de urgencias.
