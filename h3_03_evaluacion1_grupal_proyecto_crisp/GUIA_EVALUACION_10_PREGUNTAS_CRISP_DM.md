# 📋 Evaluación Grupal de Minería de Datos: Respuestas Oficiales (Fases 1 a 4 CRISP-DM)
### Proyecto: Análisis y Predicción de Demanda por Triaje en Urgencias Hospitalarias
**Modelo de Clasificación:** Regresión Logística Binaria  
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
* **Aclaración conceptual obligatoria:** Es una **métrica de negocio** denominada **Tasa de Demanda de Baja Complejidad en Urgencias (% Triaje 4 + 5)**. Mide la magnitud real del problema operativo, a diferencia de las métricas algorítmicas (Accuracy, Recall) que miden la precisión del modelo.
* **Fórmula:**
$$\text{Porcentaje Triaje 4+5} = \left( \frac{\text{Atenciones Nivel 4} + \text{Atenciones Nivel 5}}{\text{Total de Atenciones Válidas}} \right) \times 100$$
* **Variables y unidades involucradas:**
  * *Atenciones Nivel 4 (Urgencia Menor):* $5,956$ episodios de atención médica ($39.71\%$).
  * *Atenciones Nivel 5 (No urgente):* $2,993$ episodios de atención médica ($19.95\%$).
  * *Total de Atenciones Válidas:* $15,000$ episodios analizados ($100\%$).
  * *Unidad:* Porcentaje (%).
* **Valor observado medido hasta ahora:** **59.66%** en el conjunto de datos de modelado ($8,949$ pacientes no urgentes sobre $15,000$). En la base histórica institucional macro representa el **58.38%** ($584,833$ pacientes sobre $1,001,766$).
* **Acción concreta para mejorarla:** Habilitar dos consultorios de triaje avanzado y derivación temprana hacia la red de policlínicos periféricos de la CNS en el horario diurno de máxima concentración (10:00 a 18:00 horas).

---

## BLOQUE 2: ENTENDIMIENTO Y PREPARACIÓN DE LOS DATOS

### Pregunta 3: ¿Cuál es la unidad de análisis, de dónde provienen los datos y cuántos registros y variables tiene el conjunto utilizado? Indiquen qué representa una fila y, si hubo filtros, cuántos registros quedaron después de aplicarlos.
* **Unidad de análisis:** Un **episodio individual de atención médica** por urgencias.
* **Qué representa una fila:** La llegada, registro administrativo y clasificación clínica de triaje de un asegurado que solicita atención médica en el hospital.
* **Origen de los datos:** Base de datos del Sistema de Admisiones del Hospital Obrero N.º 1 (`dataset_hospital_obredo_triaje.csv` / `urgencias-hospitalarias-atendidas-20252026.xlsx`).
* **Registros y variables del conjunto:**
  * *Conjunto depurado para modelado:* $15,000$ filas y $5$ variables clave (`Nivel de triaje`, `Edad`, `Sexo`, `Hora`, `Día`).
  * *Base institucional macro:* $1,001,766$ filas y $11$ variables.
* **Filtros aplicados y registros conservados:**
  * Validación de escala de triaje admisible ($1, 2, 3, 4, 5$). Se descartaron registros nulos o sin asignación de nivel.
  * Validación de rango biológico razonable en edad ($0 \le \text{Edad} \le 120$).
  * Registros conservados tras depuración: **15,000 registros válidos (100% íntegros)** en la muestra de trabajo reproducible.

---

### Pregunta 4: ¿Qué problemas de calidad encontraron y qué preparación aplicaron? Informen la cantidad y las columnas con valores faltantes; mencionen también duplicados o inconsistencias relevantes y expliquen qué hicieron y por qué.
1. **Heterogeneidad de formatos en la variable `Hora`:** Existían cadenas horarias de texto (`'14:35'`) y marcas temporales mixtas de Excel. Se normalizó extrayendo la hora entera en escala discreta de $0$ a $23$ horas (`Hora_num`) para capturar el ciclo circadiano sin el ruido de los minutos.
2. **Inconsistencias y caracteres de codificación en `Día`:** Se limpiaron caracteres especiales y acentos (`'Mircoles'` $\rightarrow$ `'Miércoles'`, `'Sbado'` $\rightarrow$ `'Sábado'`) garantizando categorías uniformes.
3. **Valores nulos:** Se verificó la ausencia de nulos en el dataset depurado ($0$ en triaje, edad, sexo, hora y día).
4. **Codificación One-Hot Encoding:** Se aplicó `pd.get_dummies(..., drop_first=True)` sobre `Sexo` y `Día` para evitar la **trampa de la variable ficticia (*Dummy Variable Trap*)** y la multicolinealidad exacta en la regresión logística.
5. **Binarización del Target:** Transformación de la escala ordinal 1-5 a binaria: $0$ (Urgente: Triajes 1-3) y $1$ (No urgente: Triajes 4-5).

---

## BLOQUE 3: EXPLORACIÓN DE LOS DATOS (EDA)

### Pregunta 5: ¿Qué resultado interesante encontraron en la exploración y qué gráfico lo demuestra? Muéstrenlo con título y etiquetas, describan el patrón observado y expliquen por qué importa para el objetivo del proyecto.
* **Resultado y Patrón observado:**
  1. *Predominio masivo de patología ambulatoria:* El **59.66%** de los episodios corresponden a baja urgencia (Triaje 4 con 39.71% y Triaje 5 con 19.95%), mientras que las emergencias críticas (Triaje 1) representan apenas el **4.97%** ($745$ pacientes).
  2. *Patrón de congestión circadiana (Horarios vs Gravedad):* Durante la madrugada (`00:00 - 06:00`), la proporción relativa de urgencias reales (Triaje 1-3) es significativamente mayor. Sin embargo, a partir de las `06:00` y hasta las `18:00` horas, se produce una concentración masiva de atenciones leves (Triaje 4-5), superando el **62%** de todas las admisiones diurnas.
* **Gráficos que lo demuestran:** Gráficos 1 y 2 de la Pregunta 5 en el cuaderno `h3_03_evaluacion1_grupal_proyecto_crisp.ipynb` (Distribución por Triajes y Barras apiladas por Franja Horaria).
* **Por qué importa para el objetivo del proyecto:** Demuestra que la saturación de urgencias es un fenómeno **estrictamente diurno y ambulatorio**. Esto justifica que el consultorio *Fast-Track* opere en horario de 08:00 a 18:00 horas, optimizando el recurso humano sin incrementar costos en guardias nocturnas.

---

## BLOQUE 4: MODELADO PREDICTIVO Y EVALUACIÓN

### Pregunta 6: ¿Cuál es su variable objetivo para la clasificación, qué clases tiene y qué modelo revisado en clase utilizaron: regresión logística, árbol de decisión o ambos? Expliquen brevemente por qué ese modelo es pertinente para su problema.
* **Variable objetivo:** `Objetivo_Triaje_45` (Binaria).
  * **Clase 0 (Urgente / Emergencia Crítica):** Triaje 1, 2 y 3 ($6,051$ pacientes, $40.34\%$).
  * **Clase 1 (No Urgente / Baja Complejidad):** Triaje 4 y 5 ($8,949$ pacientes, $59.66\%$).
* **Modelo utilizado:** Se utilizó **Regresión Logística (`LogisticRegression`)** revisada en clase.
* **Pertinencia del modelo:**
  - Estima directamente la probabilidad condicional posterior $P(Y=1|X)$ mediante la función sigmoide:
    $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_k X_k)}}$$
  - Aporta interpretabilidad matemática directa a través de los **Odds Ratios ($e^\beta$)**, cuantificando en qué medida cada variable (edad, hora, día) incrementa o reduce el riesgo de ser una emergencia médica crítica.
  - Permite calibrar el umbral de decisión (*threshold*) para priorizar la sensibilidad hacia los pacientes graves (seguridad del paciente).

---

### Pregunta 7: ¿Qué proporción de los datos usaron para entrenamiento y para prueba, y cómo hicieron la separación? Indiquen el método o semilla si corresponde y cómo evitaron que información de prueba influyera en el entrenamiento o la preparación.
* **Proporción de partición:** **80% Entrenamiento (`Train`) y 20% Prueba (`Test`)**.
  * Muestra Train: $12,000$ registros.
  * Muestra Test: $3,000$ registros.
* **Método y Semilla:** Se utilizó `train_test_split` de Scikit-Learn con semilla fija **`random_state=42`** para garantizar reproducibilidad exacta.
* **Prevención de fuga de información (*Data Leakage*):**
  1. Se utilizó **`stratify=y`**, manteniendo idéntica proporción de la variable objetivo ($59.67\%$ Clase 1 y $40.33\%$ Clase 0) en ambas particiones.
  2. La división se realizó antes del ajuste del modelo.
  3. Los estimadores de hiperparámetros (`GridSearchCV` con 5-Fold Stratified CV) se ejecutaron exclusivamente sobre el conjunto de entrenamiento (`X_train`), dejando a `X_test` como un conjunto ciego no visto.

---

### Pregunta 8: ¿Cuáles son las variables más importantes según su modelo y cómo determinaron su importancia? Muestren la evidencia disponible —por ejemplo, coeficientes o importancias del árbol— e interpreten al menos una variable sin atribuir causalidad automáticamente.
* **Cómo se determinó la importancia:** En la Regresión Logística, la importancia y la dirección del impacto se determinan analizando los **coeficientes $\beta$** y los **Odds Ratios ($e^\beta$)**:
  - Un coeficiente positivo ($\beta > 0, \text{OR} > 1$) incrementa los momios de que la llegada sea de baja complejidad (Triaje 4 o 5).
  - Un coeficiente negativo ($\beta < 0, \text{OR} < 1$) se asocia a mayor probabilidad de emergencia médica grave (Triaje 1 a 3).
* **Evidencia cuantitativa:** La tabla de coeficientes muestra que los factores con mayor asociación hacia baja complejidad son los días hábiles (`Día: Martes` $\beta = +0.0451$, `Día: Sábado` $\beta = +0.0309$).
* **Interpretación prudente de la variable `Edad` ($\beta = -0.0122$, $\text{Odds Ratio} = 0.9879$):**
  - Al ser un coeficiente negativo, indica una **correlación estadística inversa**: por cada año adicional de edad en el paciente, los momios de ser clasificado como paciente no urgente disminuyen en aproximadamente un $1.21\%$ ($1 - 0.9879$). Dicho de forma complementaria: los pacientes de mayor edad se asocian a una mayor probabilidad de ingresar con niveles de urgencia grave (Triajes 1 a 3).
  - *Interpretación prudente sin atribuir causalidad:* No debe afirmarse que la edad avanzada "cause" directamente la enfermedad, sino que en el perfil epidemiológico del Hospital Obrero N.º 1, los pacientes adultos mayores presentan mayor carga de comorbilidades crónicas descompensadas que requieren mayor prioridad de atención.

---

### Pregunta 9: ¿Qué desempeño obtuvo el modelo en el conjunto de prueba? Reporten exactitud (accuracy) y al menos otra métrica pertinente —por ejemplo, precisión, recall o F1—, muestren la matriz de confusión y expliquen qué error sería más importante reducir en su caso.
* **Métricas en Conjunto de Prueba ($N = 3,000$ casos no vistos):**
  * Exactitud (*Accuracy*): **49.37%**
  * Precisión (*Precision*, Clase 1 - Leve): **59.04%**
  * Sensibilidad (*Recall*, Clase 1 - Leve): **49.44%**
  * $F_1\text{-Score}$: **53.82%**
  * Área ROC ($AUC\text{-}ROC$): **0.497**
* **Matriz de Confusión en Test Set:**
$$\begin{pmatrix} VN = 596 & FP = 614 \\ FN = 905 & VP = 885 \end{pmatrix}$$
  * *Verdaderos Negativos (VN):* $596$ casos urgentes retenidos correctamente en boxes de reanimación.
  * *Verdaderos Positivos (VP):* $885$ casos leves derivados correctamente al circuito rápido.
  * *Falsos Negativos (FN):* $905$ casos leves predichos como urgentes (consumen recursos pero preservan la vida).
  * *Falsos Positivos (FP):* $614$ casos urgentes predichos como leves.

#### ¿Qué error sería más importante reducir en este caso médico?
En medicina y triaje hospitalario, el impacto de los errores es asimétrico:
* **Falso Negativo de levedad (Clasificar como urgente a un paciente leve):** Error conservador. El paciente esperará en salas de agudos consumiendo recursos, pero **su vida e integridad física están protegidas**.
* **FALSO POSITIVO DE LEVEDAD (Clasificar como leve a un paciente crítico - ERROR CRÍTICO):** Es el error más peligroso. Significa enviar a la fila de espera ambulatoria a una persona con infarto agudo de miocardio, accidente cerebrovascular o hemorragia interna. **Consecuencia: Deterioro clínico acelerado o fallecimiento del asegurado.**
> **Conclusión Ética y Clínica:** El error que **es más importante reducir es el Falso Positivo de baja urgencia**. El modelo no debe actuar de forma autónoma: debe requerir siempre la validación de signos vitales por parte de enfermería antes de derivar al paciente.

---

### Pregunta 10: ¿Qué visualización permite entender cómo toma decisiones su modelo? Si usaron árbol, muestren el árbol con texto legible e interpreten una ruta de decisión; si usaron regresión logística, muestren un gráfico interpretable de sus coeficientes. Expliquen brevemente qué aporta la visualización.
* **Visualización de Regresión Logística (Gráfico Interpretable de Coeficientes Beta y Odds Ratios):**
  - Gráfico de barras horizontales con código de colores: verde para factores inductores de atención ambulatoria leve (`Día_Martes`, `Día_Sábado`, `Día_Jueves`) y rojo para predictores asociados a mayor urgencia clínica (`Sexo_Mujer`, `Día_Viernes`, `Hora_num`).
* **Visualización de la Curva ROC y AUC:**
  - Muestra la capacidad diagnóstica en todo el espectro de umbrales frente al clasificador aleatorio ($AUC = 0.500$).
* **Qué aporta la visualización:** Elimina el sesgo de "caja negra" del algoritmo. Permite a los directores médicos y jefes de guardia de la CNS entender la dirección e intensidad con la que cada variable sociodemográfica empuja la probabilidad hacia la levedad o la gravedad, facilitando la toma de decisiones informada antes de su implementación práctica en admisión de urgencias.
