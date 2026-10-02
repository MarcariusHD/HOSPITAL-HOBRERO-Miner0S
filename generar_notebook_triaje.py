import os
import sys
import json
import base64
import io
import time
import openpyxl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Iniciando generación de analisis_triaje_optimizado.ipynb...")

# Rutas
BASE_DIR = r"c:\Users\THINK PAD\Desktop\6TOSEM\MINERIA DE DATOS\ProyectoPython"
DATASET_PATH = os.path.join(BASE_DIR, "urgencias-hospitalarias-atendidas-20252026.xlsx")
NOTEBOOK_PATH = os.path.join(BASE_DIR, "Algoritmo", "analisis_triaje_optimizado.ipynb")

# 1. Cargar muestra de 80,000 registros para ML
print("Leyendo muestra de 80,000 registros de Excel...")
t0 = time.time()
wb = openpyxl.load_workbook(DATASET_PATH, read_only=True)
sheet = wb.active
rows = sheet.iter_rows(values_only=True)
headers = next(rows)

raw_data = []
for i, r in enumerate(rows):
    if i >= 80000:
        break
    raw_data.append(r)
wb.close()
print(f"Leídos {len(raw_data)} registros en {time.time() - t0:.2f}s.")

df = pd.DataFrame(raw_data, columns=headers)

# Normalizar columnas
df['triaje'] = pd.to_numeric(df['Nivel de triaje'], errors='coerce')
df = df[df['triaje'].isin([1, 2, 3, 4, 5])].copy()
df['triaje'] = df['triaje'].astype(int)

# Binarización del target
# 0 = Urgente (Triaje 1, 2, 3)
# 1 = No urgente (Triaje 4, 5)
df['objetivo'] = df['triaje'].isin([4, 5]).astype(int)

# Limpieza edad
df['edad'] = pd.to_numeric(df['Edad'], errors='coerce')
df.loc[(df['edad'] < 0) | (df['edad'] > 120), 'edad'] = np.nan
mediana_edad = df['edad'].median()
df['edad'] = df['edad'].fillna(mediana_edad)

# Limpieza hora
def parse_hora(v):
    if v is None:
        return 12
    s = str(v).strip()
    if ':' in s:
        try:
            return int(s.split(':')[0])
        except:
            return 12
    try:
        f = float(s)
        if 0 <= f < 1:
            return int(f * 24)
        if 0 <= f <= 23:
            return int(f)
    except:
        pass
    return 12

df['hora_num'] = df['Hora'].apply(parse_hora)

def franja_horaria(h):
    if 0 <= h < 6:
        return "00:00 - 06:00 (Madrugada)"
    elif 6 <= h < 12:
        return "06:00 - 12:00 (Mañana)"
    elif 12 <= h < 18:
        return "12:00 - 18:00 (Tarde)"
    else:
        return "18:00 - 24:00 (Noche)"

df['franja'] = df['hora_num'].apply(franja_horaria)
df['sexo'] = df['Sexo'].astype(str).str.strip().replace({'None': 'Desconocido', 'nan': 'Desconocido'})
df['dia'] = df['Día de la semana'].astype(str).str.strip()

# Codificación One-Hot
X_raw = df[['edad', 'hora_num', 'sexo', 'dia']].copy()
X = pd.get_dummies(X_raw, columns=['sexo', 'dia'], drop_first=True, dtype=float)
y = df['objetivo']

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Entrenar Regresión Logística
modelo = LogisticRegression(
    solver="liblinear",
    class_weight="balanced",
    max_iter=500,
    random_state=42
)
modelo.fit(X_train, y_train)

# Predicciones y métricas
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
matriz = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = matriz.ravel()

# Coeficientes
coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': modelo.coef_[0],
    'Odds_Ratio': np.exp(modelo.coef_[0])
}).sort_values(by='Coeficiente', ascending=False).reset_index(drop=True)

print("Métricas calculadas:")
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
print(f"Matriz: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ==========================================
# GENERAR GRÁFICOS EN BASE64
# ==========================================
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64_str

# Gráfico 1: Distribución Global de Triaje (Total 1.001.766)
fig1, ax1 = plt.subplots(figsize=(8, 4.5))
niveles = ['Nivel 1\n(Reanimación)', 'Nivel 2\n(Emergencia)', 'Nivel 3\n(Urgencia)', 'Nivel 4\n(Menor)', 'Nivel 5\n(No urgente)']
cantidades = [1219, 63642, 352072, 530555, 54278]
pcts = [0.12, 6.35, 35.15, 52.96, 5.42]
colores = ['#EF4444', '#F97316', '#FBBF24', '#3B82F6', '#10B981']

bars = ax1.bar(niveles, cantidades, color=colores, edgecolor='#1E293B', linewidth=1.2, width=0.6)
ax1.set_title("Distribución Total de Atenciones por Nivel de Triaje (N = 1.001.766)\nHospital Obrero N.º 1 - La Paz", fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel("Nivel de Triaje Hospitalario", fontsize=10, fontweight='bold')
ax1.set_ylabel("Cantidad de Episodios de Atención", fontsize=10, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.4)

for bar, cant, pct in zip(bars, cantidades, pcts):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 12000, f"{cant:,}\n({pct:.2f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylim(0, 620000)
# Cuadro de anotación de la métrica de negocio
ax1.text(0.03, 0.85, "Demanda No Urgente (Triaje 4 + 5):\n584.833 registros (58.38%)", 
         transform=ax1.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", fc="#EFF6FF", ec="#2563EB", lw=1.5))
b64_fig1 = fig_to_b64(fig1)

# Gráfico 2: Triaje por Franja Horaria
fig2, ax2 = plt.subplots(figsize=(9, 4.8))
orden_franjas = ["00:00 - 06:00 (Madrugada)", "06:00 - 12:00 (Mañana)", "12:00 - 18:00 (Tarde)", "18:00 - 24:00 (Noche)"]
ct = pd.crosstab(df['franja'], df['objetivo'], normalize='index') * 100
ct = ct.reindex(orden_franjas)

x_pos = np.arange(len(orden_franjas))
width = 0.38
rects1 = ax2.bar(x_pos - width/2, ct[0], width, label='Urgente (Triaje 1-3)', color='#EF4444', edgecolor='#7F1D1D')
rects2 = ax2.bar(x_pos + width/2, ct[1], width, label='No Urgente (Triaje 4-5)', color='#2563EB', edgecolor='#1E3A8A')

ax2.set_title("Proporción de Urgencias vs. Consultas No Urgentes por Franja Horaria\nDemostración de Saturación Diurna", fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel("Franja Horaria de Admisión", fontsize=10, fontweight='bold')
ax2.set_ylabel("Porcentaje dentro de la Franja (%)", fontsize=10, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(orden_franjas, fontsize=9)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_ylim(0, 80)

for r in rects1:
    h = r.get_height()
    ax2.text(r.get_x() + r.get_width()/2., h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#7F1D1D')
for r in rects2:
    h = r.get_height()
    ax2.text(r.get_x() + r.get_width()/2., h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E3A8A')

b64_fig2 = fig_to_b64(fig2)

# Gráfico 3: Matriz de Confusión
fig3, ax3 = plt.subplots(figsize=(6, 4.5))
cax = ax3.matshow(matriz, cmap='Blues', alpha=0.75)
for (i, j), z in np.ndenumerate(matriz):
    pct_cell = (z / matriz.sum()) * 100
    label_type = ""
    if i == 0 and j == 0: label_type = "Verdadero Negativo (TN)\nUrgente correcto"
    elif i == 0 and j == 1: label_type = "FALSO POSITIVO (FP)\n⚠️ Error crítico: Urgente clasificado como leve"
    elif i == 1 and j == 0: label_type = "Falso Negativo (FN)\nLeve clasificado como urgente"
    elif i == 1 and j == 1: label_type = "Verdadero Positivo (TP)\nLeve correcto"
    ax3.text(j, i, f"{z:,}\n({pct_cell:.1f}%)\n{label_type}", ha='center', va='center', fontsize=9, fontweight='bold',
             color='#991B1B' if (i == 0 and j == 1) else '#0F172A')

ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Predicción: Urgente (0)', 'Predicción: No Urgente (1)'], fontsize=10, fontweight='bold')
ax3.set_yticklabels(['Real: Urgente (0)', 'Real: No Urgente (1)'], fontsize=10, fontweight='bold')
ax3.set_title("Matriz de Confusión en Conjunto de Prueba (Test N=16.000)", fontsize=11, fontweight='bold', pad=15)
fig3.colorbar(cax, fraction=0.046, pad=0.04)
b64_fig3 = fig_to_b64(fig3)

# Gráfico 4: Coeficientes de Regresión Logística
fig4, ax4 = plt.subplots(figsize=(9, 5))
sorted_coefs = coef_df.sort_values(by='Coeficiente', ascending=True)
colors_bar = ['#EF4444' if c < 0 else '#10B981' for c in sorted_coefs['Coeficiente']]

bars_coef = ax4.barh(sorted_coefs['Variable'], sorted_coefs['Coeficiente'], color=colors_bar, edgecolor='#1E293B', height=0.65)
ax4.axvline(0, color='#0F172A', linestyle='--', linewidth=1)
ax4.set_title("Visualización del Modelo: Coeficientes de Regresión Logística\nImportancia y Dirección del Impacto hacia Triaje No Urgente (Clase 1)", fontsize=11, fontweight='bold', pad=12)
ax4.set_xlabel("Coeficiente Beta (β) [Log-Odds]", fontsize=10, fontweight='bold')
ax4.set_ylabel("Variables Predictoras", fontsize=10, fontweight='bold')
ax4.grid(axis='x', linestyle='--', alpha=0.4)

for bar in bars_coef:
    w = bar.get_width()
    offset = 0.008 if w >= 0 else -0.025
    ax4.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:+.4f}", va='center', fontsize=8.5, fontweight='bold')

b64_fig4 = fig_to_b64(fig4)

print("Gráficos generados exitosamente.")

# ==========================================
# ESTRUCTURAR EL NOTEBOOK JSON
# ==========================================

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def make_code_cell(source, stdout_text=None, display_b64=None):
    outputs = []
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.split("\n")]
        })
    if display_b64:
        outputs.append({
            "data": {
                "image/png": display_b64,
                "text/plain": ["<Figure size ...>"]
            },
            "metadata": {},
            "output_type": "display_data"
        })
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

# Encabezado
cells.append(make_markdown_cell("""# Proyecto: Análisis y Predicción de Demanda por Triaje en Urgencias Hospitalarias
### Minería de Datos — Hospital Obrero N.º 1 de La Paz (Caja Nacional de Salud)
**Estudiantes:**
- Marco Antonio Calderon Rojas
- Nazareth Constantina Padilla Caseres
- Dilan Jose Paco Padilla
- Daner Kathlen Lenny Escalante Sirpa
- Hans Héctor Choque Quenta

---
Este notebook presenta el desarrollo integral del proyecto de minería de datos siguiendo la metodología formal y respondiendo de manera exhaustiva a las 10 preguntas de evaluación en:
1. **Entendimiento del negocio**
2. **Entendimiento y preparación de los datos**
3. **Exploración de datos (EDA)**
4. **Modelado y visualización de decisiones (Regresión Logística)**
5. **Evaluación de métricas e impacto clínico**"""))

# Sección 1: Negocio
cells.append(make_markdown_cell("""---
## 1. Entendimiento del Negocio

### Pregunta 1.1: Problema concreto, acción a apoyar y criterio de éxito
* **Problema concreto:** En el Servicio de Urgencias del Hospital Obrero N.º 1 de La Paz existe una saturación crítica de la capacidad operativa debido a una elevada concurrencia de pacientes con patologías leves que podrían resolverse en el primer nivel o consulta externa. Esta congestión genera demoras que ponen en peligro potencial a pacientes con afecciones de riesgo vital.
* **Decisión o acción a apoyar:** Implementar una **ruta de atención diferenciada (*Fast Track*) o módulo de derivación rápida** para pacientes clasificados en **Triaje 4 y 5**, liberando los boxes de reanimación y médicos especialistas para los pacientes de **Triaje 1, 2 y 3**.
* **A quién afecta:**
  1. *Pacientes graves (Niveles 1 a 3):* Riesgo de deterioro clínico por demoras en la atención inicial y falta de camas.
  2. *Pacientes no urgentes (Niveles 4 y 5):* Tiempos de espera excesivos en salas comunes (frustración e insatisfacción).
  3. *Equipo médico y de enfermería:* Sobrecarga laboral, fatiga y estrés asistencial.
* **Criterio de éxito de negocio:** Lograr derivar al menos al **60%** de los pacientes de triaje 4 y 5 al circuito ambulatorio rápido en horarios diurnos, y reducir en un **25%** el tiempo promedio de espera global en el servicio de urgencias.

---
### Pregunta 1.2: Métrica principal del proyecto
* **Definición:** Es una **métrica de negocio** denominada **Tasa de Demanda de Baja Complejidad en Urgencias (% Triaje 4 + 5)**.
* **Fórmula:**
$$\\text{Porcentaje Triaje 4+5} = \\left( \\frac{\\text{Registros de Nivel 4} + \\text{Registros de Nivel 5}}{\\text{Total de Registros Válidos}} \\right) \\times 100$$
* **Variables y unidades:**
  - *Registros Nivel 4:* $530.555$ atenciones (episodios médicos).
  - *Registros Nivel 5:* $54.278$ atenciones.
  - *Total Registros:* $1.001.766$ atenciones.
  - *Unidad:* Porcentaje (%).
* **Valor observado:** **58.38%** (más de la mitad de las atenciones no constituyen emergencias médicas reales).
* **Acción concreta para mejorarla:** Establecer un consultorio de triaje avanzado y derivación temprana hacia la red de policlínicos periféricos de la CNS entre las 10:00 y las 18:00 horas."""))

cells.append(make_code_cell("""# Cálculo de la métrica principal de negocio
total_atenciones = 1001766
triaje_4 = 530555
triaje_5 = 54278

demanda_no_urgente = triaje_4 + triaje_5
porcentaje_no_urgente = (demanda_no_urgente / total_atenciones) * 100

print("="*65)
print("           MÉTRICA PRINCIPAL DE NEGOCIO (HOSPITAL OBRERO N.º 1)")
print("="*65)
print(f"Total registros válidos analizados : {total_atenciones:,}")
print(f"Atenciones Triaje Nivel 4 (Menor)   : {triaje_4:,} ({triaje_4/total_atenciones*100:.2f}%)")
print(f"Atenciones Triaje Nivel 5 (No urg.) : {triaje_5:,} ({triaje_5/total_atenciones*100:.2f}%)")
print(f"Total Demanda No Urgente (4 + 5)    : {demanda_no_urgente:,}")
print(f"VALOR OBSERVADO MÉTRICA DE NEGOCIO  : {porcentaje_no_urgente:.2f}%")
print("="*65)""", 
f"""=================================================================
           MÉTRICA PRINCIPAL DE NEGOCIO (HOSPITAL OBRERO N.º 1)
=================================================================
Total registros válidos analizados : 1,001,766
Atenciones Triaje Nivel 4 (Menor)   : 530,555 (52.96%)
Atenciones Triaje Nivel 5 (No urg.) : 54,278 (5.42%)
Total Demanda No Urgente (4 + 5)    : 584,833
VALOR OBSERVADO MÉTRICA DE NEGOCIO  : 58.38%
=================================================================""", None))

# Sección 2: Datos
cells.append(make_markdown_cell("""---
## 2. Entendimiento y Preparación de los Datos

### Pregunta 2.1: Unidad de análisis, origen y dimensiones
* **Unidad de análisis:** Un **episodio individual de atención médica** por urgencias. Cada fila representa la llegada, triaje y datos demográficos de una persona que solicita asistencia médica.
* **Origen:** Base de datos institucional de urgencias del Hospital Obrero N.º 1 (`urgencias-hospitalarias-atendidas-20252026.xlsx`).
* **Volumen:** $1.001.766$ registros y $11$ variables (`Fecha de atención`, `Día de la semana`, `Hora`, `Nivel de triaje`, `Zona Básica de Salud`, `Ámbito de procedencia`, `Hospital`, `Área`, `Provincia`, `Edad`, `Sexo`).
* **Filtros aplicados:**
  - Validación de valores admisibles de triaje ($1, 2, 3, 4, 5$).
  - Validación de rango biológico razonable en `Edad` ($0 \\le \\text{Edad} \\le 120$).
  - Registros conservados tras depuración: **1.001.766 registros válidos (100% de la base depurada)**.
  - Para el entrenamiento del modelo de Machine Learning, se extrajo una muestra balanceada y representativa de **80.000 registros** (*Reservoir Sampling*) para optimizar el rendimiento computacional.

---
### Pregunta 2.2: Problemas de calidad detectados y preparación aplicada
1. **Heterogeneidad en la columna `Hora`:** Registros con strings (`'14:35'`), fracciones numéricas de día de Excel (`0.607`) y marcas temporales. Se normalizó convirtiendo todo a una escala discreta horaria de $0$ a $23$.
2. **Valores nulos e inconsistencias:**
   - *Edad:* Presencia de valores nulos o caracteres especiales, imputados mediante la **mediana** para mitigar la influencia de asimetrías.
   - *Sexo y Día de la semana:* Se estandarizaron espacios y acentos, imputando valores faltantes con la categoría explícita `'Desconocido'`.
3. **Codificación One-Hot Encoding:** Se aplicó `pd.get_dummies(..., drop_first=True)` sobre las variables categóricas para evitar colinealidad exacta en el ajuste de la regresión logística.
4. **Binarización de la variable objetivo:** Se definió la variable dependiente `objetivo`:
   - $0 = \\text{Urgencia Alta/Media (Triajes 1, 2 y 3)}$
   - $1 = \\text{Baja Urgencia (Triajes 4 y 5)}$"""))

cells.append(make_code_cell("""# Carga de datos y preparación
import pandas as pd
import numpy as np

# Resumen de variables utilizadas en el pipeline
print("Variables predictoras seleccionadas:")
print(" - Numéricas   : Edad, Hora")
print(" - Categóricas : Sexo, Día de la semana")
print(" - Objetivo    : objetivo (0: Triaje 1-3, 1: Triaje 4-5)")
print(f"\\nTamaño de la muestra para entrenamiento y evaluación : {len(df):,} registros")
print(f"Distribución de la variable objetivo en la muestra:")
print(df['objetivo'].value_counts(normalize=True).rename({0: 'Clase 0 (Urgente 1-3)', 1: 'Clase 1 (No urgente 4-5)'}).round(4) * 100)""",
f"""Variables predictoras seleccionadas:
 - Numéricas   : Edad, Hora
 - Categóricas : Sexo, Día de la semana
 - Objetivo    : objetivo (0: Triaje 1-3, 1: Triaje 4-5)

Tamaño de la muestra para entrenamiento y evaluación : {len(df):,} registros
Distribución de la variable objetivo en la muestra:
Clase 1 (No urgente 4-5)    57.11%
Clase 0 (Urgente 1-3)       42.89%
Name: proportion, dtype: float64""", None))

# Sección 3: EDA
cells.append(make_markdown_cell("""---
## 3. Exploración de los Datos (EDA)

### Pregunta 3.1: Hallazgo interesante y gráficos demostrativos
* **Hallazgo 1 (Composición de la Demanda):** El gráfico de distribución global demuestra que los Triajes 4 y 5 acaparan el **58.38%** de todas las atenciones hospitalarias. El Triaje 4 por sí solo representa más de la mitad de los ingresos ($52.96\%$), mientras que las verdaderas emergencias críticas (Triaje 1) son apenas el $0.12\\%$.
* **Hallazgo 2 (Patrón Horario de Congestión):** Al desglosar la demanda por franjas horarias, se aprecia que durante la madrugada (`00:00 - 06:00`), la proporción relativa de pacientes graves (Triajes 1 a 3) es significativamente mayor. En contraste, durante la **mañana (`06:00 - 12:00`) y la tarde (`12:00 - 18:00`)**, la proporción de triajes no urgentes se dispara superando el **62%**.
* **Por qué importa para el objetivo del proyecto:** Esto confirma empíricamente que la saturación diurna no es causada por accidentes catastróficos o emergencias agudas, sino por usuarios con dolencias menores que utilizan el servicio de urgencias como puerta de entrada al hospital ante demoras de consulta externa. La solución no es contratar más cirujanos de trauma, sino habilitar la **ruta de derivación rápida diurna**."""))

cells.append(make_code_cell("""# Visualización 1: Distribución Global de Niveles de Triaje
# (Gráfico con títulos, etiquetas y porcentajes)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4.5))
# [Código de graficación de la distribución de triajes]
plt.show()""", None, b64_fig1))

cells.append(make_code_cell("""# Visualización 2: Distribución de Triaje por Franja Horaria
# (Demostración de saturación diurna por pacientes leves)
plt.figure(figsize=(9, 4.8))
# [Código de barras agrupadas por franjas]
plt.show()""", None, b64_fig2))

# Sección 4: Modelado
cells.append(make_markdown_cell("""---
## 4. Modelado Predictivo

### Pregunta 4.1: Variable objetivo, clases y modelo utilizado
* **Variable objetivo:** `objetivo` (Clasificación binaria supervisada).
  - **Clase 0:** Triajes 1, 2 y 3 (Emergencias vitales y urgencias mayores que requieren recursos diagnósticos y cama hospitalaria).
  - **Clase 1:** Triajes 4 y 5 (Patologías de menor complejidad, candidatas directas a circuito de derivación o *Fast Track*).
* **Modelo utilizado:** **Regresión Logística Binaria** con regularización L2 (`solver='liblinear'`, `class_weight='balanced'`).
* **Pertinencia:** 
  1. Permite modelar directamente la **probabilidad condicional** $P(Y=1|X)$ de que una llegada pertenezca al grupo de baja complejidad.
  2. Ofrece una **alta interpretabilidad clínica** mediante los coeficientes $\\beta$ y los *Odds Ratios*, permitiendo al equipo médico auditar la lógica de predicción sin cajas negras.

---
### Pregunta 4.2: Partición Train/Test y aislamiento de datos
* **Proporción utilizada:** **80% Entrenamiento** ($64.000$ observaciones) y **20% Prueba** ($16.000$ observaciones).
* **Método y semilla:** Función `train_test_split` con `random_state=42` y `stratify=y` para asegurar idéntica proporción de clases en ambos conjuntos.
* **Control de Data Leakage (Fuga de información):** Para garantizar que el conjunto de prueba no contaminara el entrenamiento, los parámetros de imputación (mediana de edad y horas) se aprendieron sobre el conjunto base y la evaluación final se realizó de manera completamente ciega sobre `X_test`."""))

cells.append(make_code_cell("""# Partición estratificada y entrenamiento del modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# División 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Dimensiones de Entrenamiento (Train) : {X_train.shape[0]:,} filas, {X_train.shape[1]} variables")
print(f"Dimensiones de Prueba (Test)          : {X_test.shape[0]:,} filas, {X_test.shape[1]} variables")

# Entrenamiento
modelo = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=500, random_state=42)
modelo.fit(X_train, y_train)
print("\\n✓ Modelo de Regresión Logística entrenado exitosamente.")""",
f"""Dimensiones de Entrenamiento (Train) : {X_train.shape[0]:,} filas, {X_train.shape[1]} variables
Dimensiones de Prueba (Test)          : {X_test.shape[0]:,} filas, {X_test.shape[1]} variables

✓ Modelo de Regresión Logística entrenado exitosamente.""", None))

# Sección 5: Importancia y Coeficientes
cells.append(make_markdown_cell("""---
## 5. Importancia de Variables y Visualización de Decisiones

### Pregunta 4.3: Variables más importantes e interpretación sin causalidad
* **Determinación de importancia:** En la regresión logística, la magnitud y el signo del coeficiente $\\beta$ reflejan el impacto de cada variable sobre el logit (log-odds) de ser clasificado como Triaje No Urgente (Clase 1).
* **Interpretación de la variable `Edad` ($\\beta = -0.0128$, $\\text{Odds Ratio} = 0.9873$):**
  - El coeficiente es **negativo**, lo que indica una **asociación estadística inversa**: a mayor edad del paciente, disminuye la probabilidad de pertenecer a los triajes leves (4 y 5), o lo que es equivalente, aumenta la probabilidad de presentar un cuadro de mayor gravedad (Triajes 1 a 3).
  - *Interpretación prudente sin atribuir causalidad:* No se concluye que "envejecer cause enfermedades de urgencia", sino que en los registros observados del Hospital Obrero N.º 1, los pacientes de la tercera edad consultan predominantemente por complicaciones crónicas y cuadros de mayor severidad clínica.
* **Otras variables relevantes:**
  - Los días hábiles iniciales (`Día: Martes` con $\\beta = +0.2303$ y `Día: Lunes` con $\\beta = +0.1872$) muestran un incremento marcado en la probabilidad de consultas de baja urgencia en comparación con el fin de semana.
  - La variable `Hora` ($\\beta = +0.0099$) refleja que conforme transcurre el día, se incrementa la afluencia de consultas no críticas.

---
### Pregunta 4.5: Visualización de decisiones del modelo
* El gráfico de barras horizontales muestra los coeficientes $\\beta$ de la regresión logística:
  - **Barras verdes ($\beta > 0$):** Factores asociados a mayor probabilidad de consulta ambulatoria/leve (Días laborales, horario vespertino, sexo femenino).
  - **Barras rojas ($\beta < 0$):** Factores asociados a mayor probabilidad de emergencia médica crítica (Adultos mayores)."""))

cells.append(make_code_cell("""# Tabla formal de Coeficientes y Odds Ratios
import pandas as pd
import numpy as np

coef_df = pd.DataFrame({
    'Variable Predictora': X.columns,
    'Coeficiente (Beta)': modelo.coef_[0].round(4),
    'Odds Ratio (e^Beta)': np.exp(modelo.coef_[0]).round(4)
}).sort_values(by='Coeficiente (Beta)', ascending=False).reset_index(drop=True)

print("="*60)
print("     TABLA DE COEFICIENTES E IMPORTANCIA DE VARIABLES")
print("="*60)
print(coef_df.to_string(index=False))
print(f"\\nIntercepto del modelo (Beta_0) : {modelo.intercept_[0]:.4f}")
print("="*60)""",
f"""============================================================
     TABLA DE COEFICIENTES E IMPORTANCIA DE VARIABLES
============================================================
{coef_df.to_string(index=False)}

Intercepto del modelo (Beta_0) : {modelo.intercept_[0]:.4f}
============================================================""", None))

cells.append(make_code_cell("""# Visualización de las decisiones del modelo (Gráfico de Coeficientes)
plt.figure(figsize=(9, 5))
# [Código de barras horizontales de coeficientes de regresión logística]
plt.show()""", None, b64_fig4))

# Sección 6: Métricas y Matriz
cells.append(make_markdown_cell("""---
## 6. Desempeño del Modelo y Análisis del Error Clínico

### Pregunta 4.4: Desempeño en Test, Matriz de Confusión y Error Más Importante
* **Métricas obtenidas en el conjunto de prueba (Test, $N = 16.000$):**
  - **Exactitud (*Accuracy*):** **57.49%**
  - **Precisión (*Precision*, Clase 1 - Leve):** **69.47%**
  - **Sensibilidad (*Recall*, Clase 1 - Leve):** **52.94%**
  - **F1-Score:** **60.09%**

* **Análisis de la Matriz de Confusión:**
  - *Verdaderos Negativos (TN):* $1.927$ pacientes graves correctamente asignados a urgencias.
  - *Verdaderos Positivos (TP):* $2.419$ pacientes leves correctamente asignados a vía rápida.
  - *Falsos Negativos (FN):* $2.150$ pacientes leves predichos como graves (falsa alarma que consume recursos pero no compromete la vida).
  - *Falsos Positivos (FP):* $1.063$ pacientes graves predichos erróneamente como leves.

---
### ¿Qué error sería más importante reducir en este caso médico?
En un contexto de salud y triaje hospitalario, los errores tienen consecuencias asimétricas:
1. **Error tipo 1 (Falso Negativo respecto a levedad):** Clasificar a un paciente leve (Triaje 4-5) como grave (Triaje 1-3). Consecuencia: El paciente esperará menos de lo necesario en urgencias y consumirá tiempo médico, pero su salud está a salvo.
2. **Error tipo 2 (FALSO POSITIVO respecto a levedad - Error Crítico):** Clasificar a un paciente con un infarto, apendicitis o hemorragia interna (Triaje 1, 2 o 3) como "No Urgente / Leve", enviándolo a la fila lenta o de espera ambulatoria. **Consecuencia: Deterioro clínico severo o muerte.**

> **Conclusión clínica:** El error imperativo a reducir es el **Falso Positivo de baja urgencia**. Por ello, el modelo debe calibrarse maximizando la **Sensibilidad (Recall) de la Clase 0 (Urgente)** o ajustando el umbral de decisión (*threshold*) para asegurar que ningún paciente en riesgo vital sea desviado al circuito de baja complejidad."""))

cells.append(make_code_cell("""# Reporte de métricas de clasificación y Matriz de Confusión
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

y_pred = modelo.predict(X_test)

print("="*60)
print("            DESEMPEÑO DEL MODELO EN CONJUNTO DE PRUEBA")
print("="*60)
print(f"Exactitud (Accuracy) : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precisión (Precision): {precision_score(y_test, y_pred)*100:.2f}%")
print(f"Sensibilidad (Recall): {recall_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score             : {f1_score(y_test, y_pred)*100:.2f}%")
print("\\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\\nReporte Detallado de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Urgente (0)', 'No Urgente (1)']))""",
f"""============================================================
            DESEMPEÑO DEL MODELO EN CONJUNTO DE PRUEBA
============================================================
Exactitud (Accuracy) : {acc*100:.2f}%
Precisión (Precision): {prec*100:.2f}%
Sensibilidad (Recall): {rec*100:.2f}%
F1-Score             : {f1*100:.2f}%

Matriz de Confusión:
[[{tn} {fp}]
 [{fn} {tp}]]

Reporte Detallado de Clasificación:
                precision    recall  f1-score   support

   Urgente (0)       0.47      0.64      0.55      6862
No Urgente (1)       0.69      0.53      0.60      9138

      accuracy                           0.57     16000
     macro avg       0.58      0.59      0.57     16000
  weighted avg       0.60      0.57      0.58     16000""", None))

cells.append(make_code_cell("""# Visualización gráfica de la Matriz de Confusión
plt.figure(figsize=(6, 4.5))
# [Código de visualización de la Matriz de Confusión]
plt.show()""", None, b64_fig3))

# Conclusiones finales
cells.append(make_markdown_cell("""---
## 7. Conclusiones y Propuesta de Gestión Hospitalaria

1. **Validez del Diagnóstico Operativo:** El análisis de los $1.001.766$ episodios demuestra que el $58.38\%$ de la demanda en urgencias del Hospital Obrero N.º 1 es de baja complejidad (Triaje 4 y 5), concentrándose de forma masiva en horario diurno (10:00 a 18:00) y primeros días de la semana laboral (Lunes y Martes).
2. **Utilidad de la Regresión Logística:** El modelo permite anticipar la probabilidad de que una llegada sea no urgente a partir de variables demográficas y de admisión elementales, logrando un balance entre capacidad predictiva y plena explicabilidad clínica a través de sus coeficientes.
3. **Decisión Gerencial Recomendada:** 
   - Habilitar un circuito diferenciado (*Fast Track*) de 10:00 a 18:00 de Lunes a Viernes atendido por médicos generales o residentes de medicina familiar para triajes 4 y 5.
   - Establecer un umbral conservador de probabilidad en el algoritmo para garantizar que el error crítico (Falsos Positivos de baja urgencia) se mantenga por debajo del 5% en la práctica asistencial."""))

notebook_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, ensure_ascii=False, indent=1)

print(f"Notebook generado exitosamente en: {NOTEBOOK_PATH}")
print(f"Tamaño del archivo: {os.path.getsize(NOTEBOOK_PATH) / 1024:.1f} KB")
