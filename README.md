# Project2 — Shark Attacks Investigation

Análisis exploratorio y limpieza del dataset de ataques de tiburón (GSAF) para estudiar patrones por tipo de ataque, año, geografía y actividad. El trabajo se ha coordinado por secciones (Marlene, Marta, Alejandro) y se desarrolla en el notebook `Project2_SharkAttacksInvestigation.ipynb`.

## Datos y cuaderno de trabajo
- Fuente: Google Drive (CSV GSAF; separador `;`). También se incluye `GSAF5.csv` en el repo.
- Notebook principal: `Project2_SharkAttacksInvestigation.ipynb`.

## Preprocesado común
- Carga con `pandas.read_csv(..., sep=';', low_memory=False)`.
- Eliminación de columnas completamente vacías: `dropna(axis=1, how='all')`.
- Normalización de texto y nombres de columnas con `standardize_text(df)` del módulo `Functions.py`:
    - Minúsculas y `strip()` en columnas de tipo texto.
    - Nombres de columnas normalizados (sin espacios, consistentes).

---

## Marlene — TYPE + YEAR

Limpieza (filtro temporal y de tipo de ataque):
- TYPE: se filtra a ataques de tipo `unprovoked` para centrar el análisis en incidentes no provocados.
- YEAR: se eliminan `NaN` en `year` y se normaliza el tipo numérico:
    - Reemplazo de coma por punto cuando aplica (`, -> .`).
    - Conversión a numérico y filtro de rango: `2000 ≤ year ≤ 2025`.
    - Conversión a entero/cadena cuando es útil para visualización.

Análisis principales:
- Distribución anual de ataques y comparación por fatalidad (`fatal_y/n`).
- Agrupaciones `groupby(['year', 'country', 'fatal_y/n'])` para contar casos y visualizar:
    - Barras por año con facetas por país.
    - Serie temporal comparando ataques fatales vs no fatales.

Resultados esperados:
- Tendencias anuales claras y comparables entre países.
- Reducción de ruido al acotar años recientes y tipo de ataque homogéneo.

---

## Marta — COUNTRY + STATE

Limpieza (foco geográfico y homogeneización de estados):
- COUNTRY: selección de países que concentran ~90% de los ataques:
    `usa, australia, south africa, bahamas, brazil, new zealand, new caledonia, egypt, reunion, french polynesia, mexico, reunion island`.
- Unificación de `reunion island → reunion`.
- `STATE`: se eliminan filas con `state` nulo y se normalizan estados por país mediante diccionarios específicos (minúsculas, `strip()`, corrección de typos y abreviaturas). Se cubren mapeos para: USA, Australia, South Africa, Bahamas, Brazil, New Zealand, New Caledonia, Egypt, Reunion, French Polynesia y Mexico.
- Caso especial Reunion: se agrupa todo lo que empieza por `saint ` bajo `saint areas` tras normalizar y quitar guiones.

Análisis principales:
- Conteos por `country` y por `state` dentro de cada país.
- Visualizaciones:
    - Heatmap de dispersión de ataques por año y país.
    - Mapa geográfico interactivo por país/estado (`scatter_geo`).

Resultados esperados:
- Geografía estandarizada que permite comparaciones por país/estado sin duplicidades por errores tipográficos.

---

## Alejandro — ACTIVITY + SEX

Limpieza (categorías de actividad y calidad de `sex`):
- `activity`: normalización a categorías canónicas agregando variantes tipográficas y sinónimos. Categorías trabajadas:
    - diving, swimming, study, paddle, fishing, surfing, kayaking
    (cada una con su lista de equivalencias y reemplazos).
- Relleno de vacíos:
    - `activity == "" → NaN → 'unknown activity'`.
    - `sex == "" → NaN → 'unknown sex'` y corrección de `lli → 'unknown sex'`.

Análisis principales:
- Distribución de ataques por actividad (top categorías) y cruces con `sex`.
- Heatmap de ataques por `country × activity` para entender patrones de práctica.

Resultados esperados:
- Reducción del número de categorías dispersas por errores de escritura.
- Mejor lectura de patrones por actividad y diferencias por sexo.

---

## Claudia — Species + Fatal (Logistic Regression)

Este análisis estima el riesgo relativo (odds ratio) de ataque fatal por especie usando regresión logística.

1) Variable dependiente
- Se filtra `fatal_y/n` a valores válidos (`'y'`, `'n'`).
- Se crea `fatal_binary` (1=fatal, 0=no fatal):
    ```python
    shark_df = shark_df.copy()
    shark_df = shark_df[shark_df['fatal_y/n'].isin(['y','n'])]
    shark_df['fatal_binary'] = shark_df['fatal_y/n'].map({'y':1, 'n':0})
    ```

2) Exclusión de especie desconocida
- Se elimina `species_clean == 'unknown'`:
    ```python
    shark_df_clean = shark_df[shark_df['species_clean'] != 'unknown'].copy()
    ```

3) Agrupar especies raras
- Especies con < 8 observaciones se agrupan en `Other` para estabilidad:
    ```python
    species_counts = shark_df_clean['species_clean'].value_counts()
    rare_species = species_counts[species_counts < 8].index
    shark_df_clean['species_grouped'] = shark_df_clean['species_clean'].replace(rare_species, 'Other')
    ```

4) Modelo logit
- Fórmula: `fatal_binary ~ C(species_grouped)` con `statsmodels`:
    ```python
    model = logit("fatal_binary ~ C(species_grouped)", data=shark_df_clean).fit()
    print(model.summary())
    ```

5) Odds Ratios (OR) e intervalos de confianza (IC 95%)
    ```python
    params = model.params
    conf = model.conf_int()
    odds_ratios = pd.DataFrame({
            "Species": params.index,
            "OR": params.apply(lambda x: round(np.exp(x),2)),
            "CI_lower": conf[0].apply(lambda x: round(np.exp(x),2)),
            "CI_upper": conf[1].apply(lambda x: round(np.exp(x),2))
    }).sort_values("OR", ascending=False).reset_index(drop=True)
    ```

6) Interpretación
- OR > 1 → mayor probabilidad de fatalidad que la categoría de referencia (`Other`).
- OR < 1 → menor probabilidad relativa.
- Los IC muestran la incertidumbre estadística.

Notas y limitaciones
- Se agrupan especies raras y se excluyen desconocidas.
- El modelo no ajusta por otros factores (lugar, actividad, tamaño, etc.).
- Cuando no hay especie, se usaron categorías de tamaño (small/medium/large) en ciertas observaciones; si se tratan como especies, pueden introducir sesgos en la distribución.

---

## Cómo reproducir
1) Abrir el notebook `Project2_SharkAttacksInvestigation.ipynb` y ejecutar las celdas en orden.
2) Asegurar el import de funciones de limpieza:
     ```python
     import importlib, Functions
     importlib.reload(Functions)
     from Functions import standardize_text
     ```
3) Generar las visualizaciones para cada sección (YEAR/COUNTRY/STATE/ACTIVITY/SEX) y, opcionalmente, el modelo logístico por especie.


