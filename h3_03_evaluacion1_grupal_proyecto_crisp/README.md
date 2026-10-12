# 📌 Entrega Oficial Hito 3: Evaluación Grupal del Proyecto (CRISP-DM Fases 1 a 4)
**Archivo de Entrega Principal:** `h3_03_evaluacion1_grupal_proyecto_crisp.ipynb`  
**Institución:** Hospital Obrero N.º 1 de La Paz — Caja Nacional de Salud (CNS)  
**Tema:** Predicción y Gestión de la Demanda por Triaje en Urgencias Hospitalarias mediante Regresión Logística

---

## 👥 Integrantes del Equipo
1. **Marco Antonio Calderón Rojas**
2. **Nazareth Constantina Padilla Cáseres**
3. **Dilan José Paco Padilla**
4. **Daner Kathlen Lenny Escalante Sirpa**
5. **Hans Héctor Choque Quenta**

---

## 🗂️ Contenido de esta Carpeta de Entrega

1. **[`h3_03_evaluacion1_grupal_proyecto_crisp.ipynb`](./h3_03_evaluacion1_grupal_proyecto_crisp.ipynb):**
   * Cuaderno Jupyter oficial totalmente ejecutado con salidas visibles, tablas y gráficos en alta resolución.
   * Responde de forma secuencial y fundamentada las **10 preguntas obligatorias** de la rúbrica (Fases 1 a 4 de CRISP-DM).
   * Contiene el modelo de **Regresión Logística**, con matriz de confusión, curva ROC, coeficientes Beta y Odds Ratios.
2. **[`GUIA_EVALUACION_10_PREGUNTAS_CRISP_DM.md`](./GUIA_EVALUACION_10_PREGUNTAS_CRISP_DM.md):**
   * Documento de síntesis y guía de defensa oral con las respuestas redactadas punto por punto, fórmulas matemáticas, valores medidos y distribución de preguntas para el equipo.
3. **[`dataset_hospital_obredo_triaje.csv`](./dataset_hospital_obredo_triaje.csv):**
   * Dataset íntegro depurado con 15,000 registros para garantizar la reproducibilidad autónoma del cuaderno en cualquier entorno.
4. **[`03_Modelo_Regresion_Logistica_Triaje.ipynb`](./03_Modelo_Regresion_Logistica_Triaje.ipynb):**
   * Cuaderno complementario centrado en el ajuste detallado y validación de hiperparámetros con Regresión Logística.

---

## 📊 Resumen Ejecutivo de Métricas y Hallazgos Medidos

| Métrica / Concepto | Valor Medido Observado | Justificación de Negocio / Clínica |
| :--- | :---: | :--- |
| **Métrica Principal de Negocio (% Triaje 4+5)** | **59.66%** | Demuestra que casi 6 de cada 10 pacientes acuden por dolencias ambulatorias leves. |
| **Volumen de Pacientes Analizados** | **15,000 episodios** | Base depurada de admisiones de urgencias sin valores nulos. |
| **Partición Train / Test** | **80% / 20% (Estratificada)** | $12,000$ casos para entrenamiento y $3,000$ casos para evaluación ciega (`seed = 42`). |
| **Exactitud Test (Regresión Logística)** | **49.37%** | Ponderación balanceada (`class_weight='balanced'`) para no descuidar emergencias reales. |
| **Precisión Test (Leves - Logística)** | **59.04%** | Confiabilidad al clasificar a un paciente como candidato a circuito rápido ambulatorio. |
| **Error Más Crítico a Evitar** | **Falso Positivo de Levedad** | Enviar a un paciente crítico a espera ambulatoria pone en riesgo su vida. |
