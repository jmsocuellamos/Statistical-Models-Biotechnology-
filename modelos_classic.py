### Funciones para bloque de Modelos clásicos
##############################################

# Paquetes
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Funciones de selección forward y backward para modelos lineales de statsmodels
# Reutilizada de https://cienciadedatos.net/documentos/py10b-regresion-lineal-multiple-python,
# web mantenida por Joaquín Amat [CC BY-NC-SA]
# ==============================================================================

def forward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    criterio: str='aic',
    add_constant: bool=True,
    verbose: bool=True
)-> list:
    """
    Realiza un procedimiento de selección de variables hacia adelante (forward)
    utilizando como criterio de bondad la métrica especificada. El procedimiento
    se detiene cuando no es posible mejorar más el modelo añadiendo variables.

    Parameters
    ----------
    X: pd.DataFrame
        Matriz de predictores
    y: pd.Series
        Variable respuesta
    metrica: str, default='aic'
        Métrica utilizada para seleccionar las variables. Debe ser una de las
        siguientes opciones: 'aic', 'bic', 'rsquared_adj'.
    add_constant: bool, default=True
        Si `True` añade una columna de 1s a la matriz de predictores con el
        con el nombre de intercept.
    verbose: bool, default=True
        Si `True` muestra por pantalla los resultados de cada iteración.

    Returns
    -------
    seleccion: list
        Lista con las variables seleccionadas.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    if add_constant:
        X = sm.add_constant(X, prepend=True).rename(columns={'const':'intercept'})

    restantes = X.columns.to_list()
    seleccion = []
    if criterio == 'rsquared_adj':
        mejor_metrica = -np.inf
        ultima_metrica = -np.inf
    else:
        mejor_metrica = np.inf
        ultima_metrica = np.inf

    while restantes:
        metricas = []
        for candidata in restantes:
            seleccion_temp = seleccion + [candidata]
            modelo  = sm.OLS(endog=y, exog=X[seleccion_temp])
            modelo_res = modelo.fit()
            metrica = getattr(modelo_res, criterio)
            metricas.append(metrica)
        if criterio == 'rsquared_adj':
            mejor_metrica = max(metricas)
            if mejor_metrica > ultima_metrica:
                mejor_variable = restantes[np.argmax(metricas)]
            else:
                break
        else:
            mejor_metrica = min(metricas)
            if mejor_metrica < ultima_metrica:
                mejor_variable = restantes[np.argmin(metricas)]
            else:
                break

        seleccion.append(mejor_variable)
        restantes.remove(mejor_variable)
        ultima_metrica = mejor_metrica

        if verbose:
            print(f'variables: {seleccion} | {criterio}: {mejor_metrica:.3f}')

    return sorted(seleccion)


def backward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    criterio: str='aic',
    add_constant: bool=True,
    verbose: bool=True
)-> list:
    """
    Realiza un procedimiento de selección de variables hacia atrás (backward)
    utilizando como criterio de bondad la métrica especificada. El procedimiento
    se detiene cuando no es posible mejorar más el modelo eliminando variables.

    Parameters
    ----------
    X: pd.DataFrame
        Matriz de predictores
    y: pd.Series
        Variable respuesta
    metrica: str, default='aic'
        Métrica utilizada para seleccionar las variables. Debe ser una de las
        siguientes opciones: 'aic', 'bic', 'rsquared_adj'.
    add_constant: bool, default=True
        Si `True` añade una columna de 1s a la matriz de predictores con el
        con el nombre de intercept.
    verbose: bool, default=True
        Si `True` muestra por pantalla los resultados de cada iteración.

    Returns
    -------
    seleccion: list
        Lista con las variables seleccionadas.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    if add_constant:
        X = sm.add_constant(X, prepend=True).rename(columns={'const':'intercept'})

    # Se inicia con todas las variables como predictores
    seleccion = X.columns.to_list()
    modelo  = sm.OLS(endog=y, exog=X[seleccion])
    modelo_res = modelo.fit()
    ultima_metrica = getattr(modelo_res, criterio)
    mejor_metrica = ultima_metrica
    if verbose:
            print(f'variables: {seleccion} | {criterio}: {mejor_metrica:.3f}')

    while seleccion:
        metricas = []
        for candidata in seleccion:
            seleccion_temp = seleccion.copy()
            seleccion_temp.remove(candidata)
            modelo  = sm.OLS(endog=y, exog=X[seleccion_temp])
            modelo_res = modelo.fit()
            metrica = getattr(modelo_res, criterio)
            metricas.append(metrica)
        if criterio == 'rsquared_adj':
            mejor_metrica = max(metricas)
            if mejor_metrica > ultima_metrica:
                peor_variable = seleccion[np.argmax(metricas)]
            else:
                break
        else:
            mejor_metrica = min(metricas)
            if mejor_metrica < ultima_metrica:
                peor_variable = seleccion[np.argmin(metricas)]
            else:
                break

        seleccion.remove(peor_variable)
        ultima_metrica = mejor_metrica

        if verbose:
            print(f'variables: {seleccion} | {criterio}: {mejor_metrica:.3f}')

    return sorted(seleccion)



# Función para comparar y seleccionar el mejor modelo de los proporcionados
# en 'mod', con la métrica AIC o BIC
def anova_selection(mod, metrica, datos):
    """
    Evalúa la métrica considerada para un conjunto de modelos.

    Parameters
    ----------
    mod: lista
    metrica: str
        Métrica utilizada para seleccionar el modelo. Debe ser una de las
        siguientes opciones: 'aic', 'bic'.
    datos: pdDataFrame
        Conjunto de datos sobre los que evaluar los modelos

    Returns
    -------
    modelo: list
        modelo con el mejor valor de la métrica.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    valores = []
    for i in range(len(mod)):
        ajuste = smf.ols(mod[i], data = datos).fit()
        valores.append(round(getattr(ajuste, metrica),4))
        pos = valores.index(min(valores))
    return mod[pos]

# Funciones para el análisis de influencia

# Función para el análisis de influencia mediante la distancia de Cook
# ==============================================================================

def influence_cooks(data: pd.DataFrame,infl_val: pd.Series)-> list:
    """
    Realiza el análisis de influencia mediante la distancia de Cook. Representa gráficamente
    el análisis de influencia y si hay alguna observación influyente nos proporciona la
    información de entrada.

    Parameters
    ----------
    data: pd.DataFrame
        Objeto con los datos del modelo
    infl_val: pd.Series
        Objeto que contiene los valores de influencia de un modelo

    Returns
    -------
    seleccion: list
        Lista con observaciones influyentes según la distancia de Cook.
    """

    # Valores de distancia para cada observación
    cooks_dist = infl_val.cooks_distance[0]
    mean_cooks_list = [1 for i in data.index]
    # Gráfico de influencia
    plt.figure(figsize = (10, 4))
    plt.scatter(data.index, cooks_dist)
    plt.plot(data.index, mean_cooks_list, color="red")
    plt.xlabel('Observación')
    plt.ylabel('Distancia Cook')
    plt.title('Puntos influyentes')
    plt.show()
    # Puntos influyentes
    influencial_points = data.index[cooks_dist > 1]
    data.iloc[influencial_points, :]

    return sorted(influencial_points)


def influence_dfbetas(modelo, infl_val):
    """
    Realiza el análisis de influencia mediante el estadístico DFBETAS. Representa gráficamente
    el análisis de influencia y si hay alguna observación influyente nos porporciona la
    información de entrada.

    Parameters
    ----------
    modelo: modelo de statsmodels
        Objeto con el modelo ajustado
    infl_val: pd.Series
        Objeto que contiene los valores de influencia de un modelo

    Returns
    -------
    influencial_points:
        índice de los puntos influyentes apra cada coeficiente.
    """

    # Valores de distancia para cada observación
    dfbetas = infl_val.dfbetas
    coef = modelo.model.exog_names
    ncoef = len(coef)
    nsample = modelo.nobs
    data = pd.DataFrame(modelo.model.exog, columns = coef)

    # valor de influencia
    dfbetas_list = [2/np.sqrt(nsample) for i in data.index]
    # Diccionario de resultados
    influencial_points = dict()
    # Puntos influyentes
    for i in range(0, ncoef):
      influencial_points['dfbeta_'+coef[i]] = data.index[dfbetas[:,i] > dfbetas_list]

    return influencial_points

def influence_dffits(modelo, infl_val):
    """
    Realiza el análisis de influencia mediante el estadístico DFFITS. Representa gráficamente
    el análisis de influencia y si hay alguna observación influyente nos porporciona la
    información de entrada.

    Parameters
    ----------
    modelo: modelo de statsmodels
        Objeto con el modelo ajustado
    infl_val: pd.Series
        Objeto que contiene los valores de influencia de un modelo

    Returns
    -------
    influencial_points:
        índice de los puntos influyentes apra cada coeficiente.
    """

    # Valores de distancia para cada observación
    dffits = infl_val.dffits[0]
    coef = modelo.model.exog_names
    ncoef = len(coef)
    nsample = modelo.nobs
    data = pd.DataFrame(modelo.model.exog, columns = coef)

    # valor de influencia
    dffits_list = [2/np.sqrt(ncoef/nsample) for i in data.index]
    # Puntos influyentes
    influencial_points = data.index[dffits > dffits_list]

    return influencial_points

def influence_covratio(modelo,infl_val):
  # Extraer el leverage (h_ii)
  leverage = infl_val.hat_matrix_diag

  # Extraer residuos estudentizados (externally studentized residuals)
  studentized_residuals = infl_val.resid_studentized_external

  # Calcular COVRATIO
  n = modelo.nobs  # Número de observaciones
  p = len(modelo.params)  # Número de parámetros (incluyendo intercepto)

  covratio = (1 / (1 - leverage)) * ((n - p - 1 + studentized_residuals**2) / (n - p))**p

  # Imprimir resultados
  print("COVRATIO para cada observación:")
  print(covratio)

  # Identificar observaciones influyentes
  threshold = 3 * p / n
  influential = np.abs(covratio - 1) > threshold
  print("\nObservaciones influyentes (|COVRATIO - 1| > 3p/n):")
  print(np.where(influential)[0])

  return np.where(influential)[0]


def diagnostico_modelo_regresion(modelo, alpha=0.05, figsize=(15, 10)):
    """
    Realiza un diagnóstico completo de un modelo de regresión lineal (statsmodels).
    
    Parámetros:
    -----------
    modelo : result object de statsmodels (ej. resultado de sm.OLS(...).fit())
    alpha : float, nivel de significancia para los tests (default 0.05)
    figsize : tuple, tamaño de la figura para los gráficos.
    
    Retorna:
    --------
    df_resultados : pd.DataFrame con los estadísticos de los tests y su interpretación.
    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.stattools import durbin_watson
    import statsmodels.stats.api as sms

    # 1. Extracción de datos del modelo
    fitted_vals = modelo.fittedvalues
    residuos = modelo.resid
    # Estandarización de residuos (residuo / raiz(MSE))
    dt_resid = np.sqrt(modelo.mse_resid)
    residuos_std = residuos / dt_resid
    
    # Obtenemos la variable dependiente (Target) real
    # En statsmodels OLS, model.endog es el target (Y)
    y_real = modelo.model.endog
    
    # DataFrame auxiliar para gráficos
    df_diag = pd.DataFrame({
        'Real': y_real,
        'Prediccion': fitted_vals,
        'Residuos': residuos,
        'Residuos_Std': residuos_std
    })

    # ==============================================================================
    # 2. GENERACIÓN DE GRÁFICOS
    # ==============================================================================
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    fig.suptitle('Diagnóstico de Residuos del Modelo', fontsize=16)

    # Gráfico 1: Predicción vs Real (Linealidad)
    sns.regplot(x=df_diag['Prediccion'], y=df_diag['Real'],
                scatter=True, ci=None, line_kws={'color': 'red'}, ax=axes[0,0])
    axes[0,0].set_title('Linealidad: Predicción vs Real')
    axes[0,0].set_xlabel('Valores Predichos')
    axes[0,0].set_ylabel('Valores Reales')

    # Gráfico 2: Predicción vs Residuos (Homocedasticidad)
    sns.regplot(x=df_diag['Prediccion'], y=df_diag['Residuos_Std'],
                scatter=True, ci=None, line_kws={'color': 'red'}, ax=axes[0,1])
    axes[0,1].axhline(y=0, color='black', linestyle='--')
    axes[0,1].set_title('Homocedasticidad: Predichos vs Residuos Std')
    axes[0,1].set_xlabel('Valores Predichos')
    axes[0,1].set_ylabel('Residuos Estandarizados')

    # Gráfico 3: Histograma de Residuos (Normalidad)
    sns.histplot(df_diag['Residuos_Std'], kde=True, stat="density", ax=axes[0,2])
    axes[0,2].set_title('Normalidad: Histograma de Residuos')

    # Gráfico 4: Q-Q Plot (Normalidad)
    sm.qqplot(residuos_std, line='45', fit=True, ax=axes[1,0])
    axes[1,0].set_title('Normalidad: Q-Q Plot')

    # Gráfico 5: Autocorrelación (ACF)
    sm.graphics.tsa.plot_acf(residuos, lags=min(20, len(residuos)//2 - 1), ax=axes[1,1])
    axes[1,1].set_title('Independencia: Autocorrelación')

    # Borramos el 6to gráfico vacío para limpieza visual
    fig.delaxes(axes[1,2])
    
    plt.tight_layout()
    plt.show()

    # ==============================================================================
    # 3. TESTS ESTADÍSTICOS
    # ==============================================================================
    resultados = []

    # A) Test de Shapiro-Wilk (Normalidad)
    # H0: Los residuos siguen una distribución normal
    stat_sw, p_val_sw = stats.shapiro(residuos)
    cumple_sw = p_val_sw > alpha
    resultados.append({
        'Test': 'Shapiro-Wilk (Normalidad)',
        'Hipótesis Nula (H0)': 'Residuos Normales',
        'Estadístico': stat_sw,
        'P-Valor': p_val_sw,
        'Cumple Hipótesis': 'Sí' if cumple_sw else 'No'
    })

    # B) Test de Breusch-Pagan (Homocedasticidad)
    # H0: La varianza de los errores es constante (Homocedasticidad)
    # Necesitamos la matriz de diseño (exog)
    exog = modelo.model.exog
    try:
        test_bp = sms.het_breuschpagan(residuos, exog)
        # El output es: (lm, lm_pvalue, fvalue, f_pvalue). Usamos lm_pvalue [1]
        p_val_bp = test_bp[1]
        cumple_bp = p_val_bp > alpha
        resultados.append({
            'Test': 'Breusch-Pagan (Homocedasticidad)',
            'Hipótesis Nula (H0)': 'Varianza Constante',
            'Estadístico': test_bp[0],
            'P-Valor': p_val_bp,
            'Cumple Hipótesis': 'Sí' if cumple_bp else 'No'
        })
    except Exception as e:
        # En casos simples univariantes a veces da error de dimensiones si no se maneja bien exog
        resultados.append({'Test': 'Breusch-Pagan', 'Resultado': 'Error en cálculo'})
        cumple_bp = False

    # C) Test de Durbin-Watson (Independencia / Autocorrelación)
    # H0: No hay autocorrelación de primer orden
    # Rango: 0 a 4. 2 es no correlación.
    # Regla general: 1.5 < DW < 2.5 se considera aceptable (independencia).
    stat_dw = durbin_watson(residuos)
    # No devuelve p-valor directo, usamos regla heurística
    cumple_dw = 1.5 <= stat_dw <= 2.5
    resultados.append({
        'Test': 'Durbin-Watson (Independencia)',
        'Hipótesis Nula (H0)': 'No Autocorrelación (stat ~ 2)',
        'Estadístico': stat_dw,
        'P-Valor': np.nan, # No aplica p-valor estándar
        'Cumple Hipótesis': 'Sí' if cumple_dw else 'No'
    })

    df_resultados = pd.DataFrame(resultados)
    
    # ==============================================================================
    # 4. COMENTARIOS Y CONCLUSIONES
    # ==============================================================================
    print("\n" + "="*50)
    print("CONCLUSIONES DEL DIAGNÓSTICO")
    print("="*50)
    
    # Comentario Normalidad
    if cumple_sw:
        print(f"[OK] Normalidad: Los residuos parecen seguir una distribución normal (p={p_val_sw:.4f} > {alpha}).")
    else:
        print(f"[X] Normalidad: Se rechaza la hipótesis de normalidad (p={p_val_sw:.4f} < {alpha}).")
    
    # Comentario Homocedasticidad
    if cumple_bp:
        print(f"[OK] Homocedasticidad: La varianza de los errores es constante (p={p_val_bp:.4f} > {alpha}).")
    else:
        print(f"[X] Homocedasticidad: Existen indicios de heterocedasticidad (varianza no constante).")
        
    # Comentario Independencia
    if cumple_dw:
        print(f"[OK] Independencia: No parece haber autocorrelación severa (DW={stat_dw:.2f} está entre 1.5 y 2.5).")
    else:
        if stat_dw < 1.5:
            tipo = "positiva"
        else:
            tipo = "negativa"
        print(f"[X] Independencia: Posible autocorrelación {tipo} de los residuos (DW={stat_dw:.2f}).")
        
    print("="*50 + "\n")

    return df_resultados

def analisis_multicolinealidad(modelo, plot_corr=True, figsize=(10, 8)):
    """
    Realiza un diagnóstico de multicolinealidad basado en VIF y Número de Condición.
    
    Parámetros:
    -----------
    modelo : result object de statsmodels 
        Modelo de regresión lineal ya ajustado.
    plot_corr : bool
        Si True, genera un mapa de calor de correlaciones entre predictores.
    figsize : tuple
        Tamaño de la figura para el gráfico.
        
    Retorna:
    --------
    df_vif : pd.DataFrame
        Tabla con los valores VIF por variable y su interpretación.
    dict_global : dict
        Diccionario con el Número de Condición y su diagnóstico.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # ==============================================================================
    # 1. EXTRACCIÓN DE DATOS
    # ==============================================================================
    # Matriz de diseño (Variables Explicativas)
    X = modelo.model.exog
    nombres_variables = modelo.model.exog_names
    
    # R^2 del modelo global (para el criterio comparativo)
    r2_modelo = modelo.rsquared
    
    # Umbral dinámico: 1 / (1 - R^2_modelo)
    # Si R^2 es 1, evitamos división por cero
    if r2_modelo < 1:
        vif_umbral_modelo = 1 / (1 - r2_modelo)
    else:
        vif_umbral_modelo = np.inf
    
    # ==============================================================================
    # 2. CÁLCULO DE VIF (Nivel Variable)
    # ==============================================================================
    datos_vif = []
    
    for i in range(X.shape[1]):
        # Calculamos VIF para la variable i
        vif_i = variance_inflation_factor(X, i)
        var_name = nombres_variables[i]
        
        # INTERPRETACIÓN SEGÚN TUS REGLAS
        # -------------------------------
        problemas = []
        
        # Criterio 1: VIF > 10
        es_alto = vif_i > 10
        if es_alto:
            problemas.append("VIF > 10")
            
        # Criterio 2: VIF > 1/(1-R^2_modelo)
        # Implica que la relación entre X's es mayor que entre X y Y
        es_mayor_modelo = vif_i > vif_umbral_modelo
        if es_mayor_modelo:
            problemas.append("VIF > 1/(1-R²)")
            
        # Conclusión
        if len(problemas) > 0:
            conclusion = f"Multicolinealidad ({' y '.join(problemas)})"
        else:
            conclusion = "Sin problemas evidentes"
            
        datos_vif.append({
            'Variable': var_name,
            'VIF': round(vif_i, 4),
            'Umbral Modelo (1/(1-R²))': round(vif_umbral_modelo, 4),
            '¿VIF > 10?': 'SÍ' if es_alto else 'No',
            '¿VIF > Umbral Modelo?': 'SÍ' if es_mayor_modelo else 'No',
            'Diagnóstico': conclusion
        })
    
    df_vif = pd.DataFrame(datos_vif)
    
    # ==============================================================================
    # 3. CÁLCULO DE NÚMERO DE CONDICIÓN (Nivel Global)
    # ==============================================================================
    # kappa = lambda_max / lambda_min
    cond_number = np.linalg.cond(X)
    
    # INTERPRETACIÓN SEGÚN TUS REGLAS
    # -------------------------------
    if cond_number < 100:
        diag_cond = "No hay problemas de multicolinealidad"
        severidad = "Ninguna"
    elif 100 <= cond_number < 1000:
        diag_cond = "Multicolinealidad MODERADA"
        severidad = "Moderada"
    else: # cond_number >= 1000
        diag_cond = "Multicolinealidad SEVERA"
        severidad = "Severa"

    # ==============================================================================
    # 4. VISUALIZACIÓN (Matriz de Correlación)
    # ==============================================================================
    if plot_corr:
        # Convertimos exog a DataFrame para facilitar el ploteo (excluyendo constante si es 1)
        df_X = pd.DataFrame(X, columns=nombres_variables)
        
        # Si existe la constante (intercepto), a veces se quita del heatmap por no aportar correlación
        if 'const' in df_X.columns and df_X['const'].nunique() == 1:
             df_X = df_X.drop(columns=['const'])
        
        plt.figure(figsize=figsize)
        # Mapa de calor de correlaciones absolutas (para ver intensidad)
        corr_matrix = df_X.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title('Matriz de Correlación entre Variables Explicativas')
        
        plt.show()

    # ==============================================================================
    # 5. SALIDA DE RESULTADOS
    # ==============================================================================
    print("-" * 60)
    print("DIAGNÓSTICO GLOBAL (NÚMERO DE CONDICIÓN)")
    print("-" * 60)
    print(f"Número de Condición (κ): {cond_number:,.2f}")
    print(f"Diagnóstico: {diag_cond}")
    print("-" * 60 + "\n")
    
    print("DIAGNÓSTICO POR VARIABLE (VIF)")
    
    return df_vif


def analisis_influencia(modelo, figsize=(15, 10)):
    """
    Realiza un análisis de influencia completo indicando el índice original de la observación.
    
    Parámetros:
    -----------
    modelo : statsmodels result
        Modelo de regresión ajustado.
    figsize : tuple
        Tamaño del lienzo de gráficos.
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame filtrado mostrando solo las observaciones influyentes, 
        su Índice Original y la razón de su influencia.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.outliers_influence import OLSInfluence
    
    # 1. Extracción de parámetros básicos
    n = modelo.nobs             
    p = len(modelo.params)      
    infl = modelo.get_influence()
    
    # --- NOVEDAD: Recuperamos el índice original del DataFrame de entrenamiento ---
    # Esto permite identificar la fila real (ej. ID cliente, Fecha, etc.)
    idx_original = modelo.model.data.row_labels
    
    # 2. Definición de Umbrales
    th_cook = 1.0
    th_dfbetas = 2.0 / np.sqrt(n)
    th_dffits = 2.0 * np.sqrt(p / n)
    th_cov_upper = 1.0 + (3.0 * p / n)
    th_cov_lower = 1.0 - (3.0 * p / n)
    
    # 3. Obtención de métricas
    cooks_d, _ = infl.cooks_distance
    dfbetas = infl.dfbetas
    dfbetas_names = modelo.model.exog_names
    max_dfbetas_val = np.max(np.abs(dfbetas), axis=1)
    dffits, _ = infl.dffits
    covratio = infl.cov_ratio
    
    # 4. Construcción del DataFrame de Resultados
    # Incluimos 'Index_Original' como primera columna
    df_res = pd.DataFrame({
        'Index_Original': idx_original, 
        'Cook_D': cooks_d,
        'DFFITS': dffits,
        'COVRATIO': covratio,
        'Max_Abs_DFBETA': max_dfbetas_val
    })
    
    # 5. Lógica de Detección de Influencia
    razones = []
    es_influyente = []
    
    for i in df_res.index:
        motivos = []
        
        # Criterio Cook
        if df_res.loc[i, 'Cook_D'] > th_cook:
            motivos.append(f"Cook > 1")
            
        # Criterio DFFITS
        if np.abs(df_res.loc[i, 'DFFITS']) > th_dffits:
            motivos.append(f"|DFFITS| > {th_dffits:.2f}")
            
        # Criterio COVRATIO
        val_cov = df_res.loc[i, 'COVRATIO']
        if (val_cov > th_cov_upper) or (val_cov < th_cov_lower):
            motivos.append(f"COVRATIO fuera rango ({th_cov_lower:.2f}, {th_cov_upper:.2f})")
            
        # Criterio DFBETAS
        if df_res.loc[i, 'Max_Abs_DFBETA'] > th_dfbetas:
            # Buscamos qué variables específicas fallan
            row_dfbetas = dfbetas[i, :]
            vars_afectadas = [dfbetas_names[j] for j, val in enumerate(row_dfbetas) if abs(val) > th_dfbetas]
            motivos.append(f"DFBETAS > {th_dfbetas:.2f} en: {vars_afectadas}")
            
        # Consolidación
        if len(motivos) > 0:
            es_influyente.append(True)
            razones.append(" | ".join(motivos))
        else:
            es_influyente.append(False)
            razones.append("Normal")
            
    df_res['Es_Influyente'] = es_influyente
    df_res['Diagnóstico'] = razones
    
    # Filtramos y reordenamos columnas
    df_relevantes = df_res[df_res['Es_Influyente']].copy()
    df_relevantes = df_relevantes[['Index_Original', 'Diagnóstico', 'Cook_D', 'DFFITS', 'COVRATIO', 'Max_Abs_DFBETA']]
    
    # ==============================================================================
    # 6. GENERACIÓN DE GRÁFICOS (LIENZO)
    # ==============================================================================
    # Nota: En los gráficos usamos el índice posicional (0, 1, 2...) en el eje X para
    # mantener la legibilidad, pero el usuario puede buscar el punto en la tabla.
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Análisis de Influencia (n={n}, p={p})', fontsize=16)
    
    indices_posicionales = range(len(df_res)) # 0 a n-1

    # GRÁFICO 1: Distancia de Cook
    axes[0, 0].vlines(indices_posicionales, 0, cooks_d, color='gray', alpha=0.7)
    axes[0, 0].scatter(indices_posicionales, cooks_d, alpha=0.7)
    axes[0, 0].axhline(y=th_cook, color='r', linestyle='--', label='Umbral = 1')
    axes[0, 0].set_title("Distancia de Cook")
    axes[0, 0].set_ylabel("Cook's D")

    # GRÁFICO 2: DFFITS
    axes[0, 1].scatter(indices_posicionales, dffits, alpha=0.7)
    axes[0, 1].axhline(y=th_dffits, color='r', linestyle='--', label=f'+/- {th_dffits:.2f}')
    axes[0, 1].axhline(y=-th_dffits, color='r', linestyle='--')
    axes[0, 1].set_title("DFFITS")
    axes[0, 1].set_ylabel("DFFITS")

    # GRÁFICO 3: COVRATIO
    axes[1, 0].scatter(indices_posicionales, covratio, alpha=0.7)
    axes[1, 0].axhline(y=1, color='k', alpha=0.3)
    axes[1, 0].axhline(y=th_cov_upper, color='r', linestyle='--', label=f'Lim {th_cov_upper:.2f}')
    axes[1, 0].axhline(y=th_cov_lower, color='r', linestyle='--', label=f'Lim {th_cov_lower:.2f}')
    axes[1, 0].set_title("COVRATIO")
    axes[1, 0].set_ylabel("CovRatio")

    # GRÁFICO 4: DFBETAS
    axes[1, 1].scatter(indices_posicionales, max_dfbetas_val, alpha=0.7, color='green')
    axes[1, 1].axhline(y=th_dfbetas, color='r', linestyle='--', label=f'Umbral {th_dfbetas:.2f}')
    axes[1, 1].set_title("Max |DFBETAS|")
    axes[1, 1].set_ylabel("Max DFBETA")

    plt.tight_layout()
    plt.show()
    
    return df_relevantes

def diagnostico_anova(modelo, figsize=(14, 10)):
    """
    Realiza el diagnóstico completo de un modelo ANOVA (simple o factorial).
    
    Detecta automáticamente múltiples factores e interacciones y agrupa los
    residuos por la combinación de niveles presentes en el diseño.
    
    Parámetros:
    -----------
    modelo : statsmodels result
        Modelo ajustado (ej. sm.OLS.from_formula(...).fit()).
    figsize : tuple
        Tamaño de la figura.
        
    Retorna:
    --------
    pd.DataFrame
        Tabla con los resultados de los tests de hipótesis (Levene y Shapiro-Wilk)
        desglosados por combinación de factores.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.outliers_influence import OLSInfluence
    import re


    # ==============================================================================
    # 1. AUTO-DETECCIÓN DE VARIABLES Y CONSTRUCCIÓN DE GRUPOS
    # ==============================================================================
    try:
        # Recuperamos el DataFrame original usado en el ajuste
        df = modelo.model.data.frame.copy()
        col_response = modelo.model.endog_names
        
        # Obtenemos los términos del modelo (ej: ['Intercept', 'C(f1)', 'C(f2)', 'C(f1):C(f2)'])
        terminos = modelo.model.data.design_info.term_names
        
        # Filtramos 'Intercept' y extraemos los nombres limpios de las columnas
        vars_predictoras = set()
        for term in terminos:
            if term == 'Intercept':
                continue
            
            # Separamos interacciones (ej: "A:B" -> ["A", "B"])
            partes = term.split(':')
            
            for p in partes:
                # Limpieza: Si la fórmula usó 'C(variable)', extraemos solo 'variable'
                match = re.search(r"C\((.*?)\)", p)
                if match:
                    vars_predictoras.add(match.group(1))
                else:
                    vars_predictoras.add(p)
        
        cols_factores = list(vars_predictoras)
        
        if not cols_factores:
            raise ValueError("No se encontraron factores en el modelo.")
            
        # --- CREACIÓN DE LA VARIABLE DE AGRUPACIÓN COMBINADA ---
        # Si hay factores ['Spray', 'Temp'], crea grupos tipo "A - High", "A - Low"...
        # Esto captura la celda del diseño experimental.
        if len(cols_factores) > 1:
            df['Grupo_Comb'] = df[cols_factores].astype(str).agg(' & '.join, axis=1)
            nombre_factor_eje = f"Combinación: {' & '.join(cols_factores)}"
        else:
            df['Grupo_Comb'] = df[cols_factores[0]].astype(str)
            nombre_factor_eje = cols_factores[0]
            
        print(f"Diagnóstico automático para: {col_response}")
        print(f"Factores detectados: {cols_factores}")
        print(f"Agrupación por: {nombre_factor_eje}")
        print("-" * 60)

    except AttributeError:
        raise ValueError("El modelo no contiene metadatos suficientes. Use la API de fórmulas.")

    # ==============================================================================
    # 2. CÁLCULO DE RESIDUOS
    # ==============================================================================
    prediccion = modelo.fittedvalues
    dt_resid = np.sqrt(modelo.mse_resid)
    residuos_std = modelo.resid / dt_resid
    
    indices_modelo = modelo.model.data.row_labels
    df_diag = df.loc[indices_modelo].copy()
    
    df_diag['Prediccion'] = prediccion
    df_diag['Residuos_Std'] = residuos_std
    df_diag['Residuos'] = modelo.resid 
    
    # Lista de grupos únicos (combinaciones)
    grupos_unicos = np.sort(df_diag['Grupo_Comb'].unique())
    
    # ==============================================================================
    # 3. GENERACIÓN DE GRÁFICOS (LIENZO)
    # ==============================================================================
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Diagnóstico de Residuos\nModelo: {col_response} ~ {nombre_factor_eje}', fontsize=14)
    
    # A) Homogeneidad de Varianzas (Boxplot por Combinación de Niveles)
    sns.boxplot(
        data=df_diag, 
        x='Grupo_Comb',      # Usamos la variable combinada
        y='Residuos_Std', 
        hue='Grupo_Comb',    # Asignamos hue para evitar warnings
        legend=False,       
        palette="viridis",   # Viridis ayuda a distinguir si hay muchos grupos
        ax=axes[0,0]
    )
    axes[0,0].axhline(y=0, color='red', linestyle='--')
    axes[0,0].set_title('Homogeneidad: Residuos por Grupo')
    axes[0,0].set_ylabel('Residuos Estandarizados')
    axes[0,0].set_xlabel('Niveles de Factores')
    # Rotamos etiquetas si hay muchos grupos para que se lean bien
    axes[0,0].tick_params(axis='x', rotation=45) 
    
    # B) Normalidad Global (QQ Plot)
    sm.qqplot(df_diag['Residuos_Std'], line='45', fit=True, ax=axes[0,1])
    axes[0,1].set_title('Normalidad Global: Q-Q Plot')
    
    # C) Distancia de Cook
    infl = OLSInfluence(modelo)
    (c, p) = infl.cooks_distance
    axes[1,0].stem(np.arange(len(c)), c, markerfmt=",")
    axes[1,0].set_title('Puntos Influyentes: Distancia de Cook')
    axes[1,0].set_ylabel('Distancia Cook')
    axes[1,0].set_xlabel('Índice de Observación')
    axes[1,0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Umbral (1.0)')
    axes[1,0].legend()

    # D) Histograma Global
    sns.histplot(df_diag['Residuos_Std'], kde=True, stat="density", ax=axes[1,1])
    axes[1,1].set_title('Distribución de Residuos Global')
    
    plt.tight_layout()
    plt.show()
    
    # ==============================================================================
    # 4. TESTS ESTADÍSTICOS
    # ==============================================================================
    resultados_tests = []
    
    # Test de Levene (Homocedasticidad) entre todos los grupos/combinaciones
    lista_residuos_grupos = [df_diag[df_diag['Grupo_Comb'] == g]['Residuos'] for g in grupos_unicos]
    
    # Levene requiere al menos dos grupos
    if len(lista_residuos_grupos) > 1:
        stat_levene, p_levene = stats.levene(*lista_residuos_grupos, center='median')
        concl_levene = 'Varianzas Iguales (H0)' if p_levene > 0.05 else 'Varianzas Diferentes (H1)'
    else:
        stat_levene, p_levene = (np.nan, np.nan)
        concl_levene = "Solo hay 1 grupo (No aplicable)"

    resultados_tests.append({
        'Test': 'Levene (Homocedasticidad)',
        'Ámbito': 'Global (Entre combinaciones)',
        'P-Valor': round(p_levene, 4) if not np.isnan(p_levene) else "-",
        'Conclusión': concl_levene
    })
    
    # Test de Shapiro-Wilk (Normalidad DENTRO de cada combinación)
    for g in grupos_unicos:
        datos_grupo = df_diag[df_diag['Grupo_Comb'] == g]['Residuos']
        
        # Shapiro requiere al menos 3 datos
        if len(datos_grupo) >= 3:
            stat_sw, p_sw = stats.shapiro(datos_grupo)
            concl = 'Normal (H0)' if p_sw > 0.05 else 'No Normal (H1)'
        else:
            p_sw = np.nan
            concl = f"N={len(datos_grupo)} (Insuficiente)"

        resultados_tests.append({
            'Test': 'Shapiro-Wilk (Normalidad)',
            'Ámbito': f'Grupo: {g}',
            'P-Valor': round(p_sw, 4) if not np.isnan(p_sw) else "-",
            'Conclusión': concl
        })
        
    df_resultados = pd.DataFrame(resultados_tests)
    
    return df_resultados

def prediccion_anova(modelo, func=None, figsize=(12, 5)):
    """
    Realiza predicciones y gráficos de intervalos de confianza para modelos ANOVA.
    Permite aplicar una función de transformación a las predicciones (útil para 
    invertir transformaciones realizadas durante el ajuste: log, sqrt, etc.).

    Parámetros:
    -----------
    modelo : statsmodels result
        El modelo ajustado.
    func : str o callable, opcional (Default: None)
        Función para transformar la predicción (ej. para volver a la escala original).
        - Strings admitidos: 
            'exp' (exponencial, inversa del log), 
            'log' (logaritmo), 
            'sqrt' (raíz cuadrada), 
            'sq' (cuadrado, inversa de la raíz), 
            'inverse' (1/x).
        - Callable: Una función personalizada (ej. lambda x: x**3).
    figsize : tuple
        Tamaño de la figura.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con las predicciones en escala del modelo y escala transformada.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools
    import re

    # 1. DICCIONARIO DE FUNCIONES COMUNES
    # -----------------------------------
    mapa_funciones = {
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'sq': np.square,
        'square': np.square,
        'inverse': lambda x: 1/x,
        'inv': lambda x: 1/x
    }

    # 2. INTROSPECCIÓN DEL MODELO
    # ---------------------------
    terminos = modelo.model.data.design_info.term_names
    vars_predictoras = set()
    
    for term in terminos:
        if term == 'Intercept': continue
        partes = term.split(':')
        for p in partes:
            match = re.search(r"C\((.*?)\)", p)
            if match:
                vars_predictoras.add(match.group(1))
            else:
                vars_predictoras.add(p)
    
    factores = list(vars_predictoras)
    df_orig = modelo.model.data.frame
    
    if len(factores) == 0:
        raise ValueError("No se detectaron factores en el modelo.")

    # 3. GENERACIÓN DE LA REJILLA DE PREDICCIÓN
    # -----------------------------------------
    niveles = [df_orig[f].unique() for f in factores]
    combinaciones = list(itertools.product(*niveles))
    df_pred = pd.DataFrame(combinaciones, columns=factores)
    
    # 4. PREDICCIÓN (Escala del modelo)
    # ---------------------------------
    pred_obj = modelo.get_prediction(df_pred)
    resumen = pred_obj.summary_frame(alpha=0.05)
    
    cols_interes = ['mean', 'mean_ci_lower', 'mean_ci_upper']
    df_resultado = pd.concat([df_pred, resumen[cols_interes]], axis=1)

    # 5. APLICACIÓN DE LA FUNCIÓN (func)
    # ----------------------------------
    funcion_aplicar = None
    nombre_trans = "Transformada"

    if func is not None:
        if isinstance(func, str):
            if func.lower() in mapa_funciones:
                funcion_aplicar = mapa_funciones[func.lower()]
                nombre_trans = func.capitalize()
            else:
                raise ValueError(f"La función '{func}' no está predefinida. Use: {list(mapa_funciones.keys())}")
        elif callable(func):
            funcion_aplicar = func
            nombre_trans = "Custom"
        else:
            raise TypeError("El parámetro 'func' debe ser un string o una función.")

        # Aplicar transformación
        df_resultado['mean_trans'] = funcion_aplicar(df_resultado['mean'])
        df_resultado['ci_lower_trans'] = funcion_aplicar(df_resultado['mean_ci_lower'])
        df_resultado['ci_upper_trans'] = funcion_aplicar(df_resultado['mean_ci_upper'])

    # 6. GENERACIÓN DE GRÁFICOS
    # -------------------------
    ncols = 2 if funcion_aplicar else 1
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
    if ncols == 1: ax = [ax]

    # --- Helper para plotear ---
    def plot_intervals(axis, data, x_col, y_col, y_min, y_max, hue_col, title):
        if hue_col:
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, s=100, ax=axis)
            niveles_hue = data[hue_col].unique()
            palette = sns.color_palette(n_colors=len(niveles_hue))
            
            for i, nivel in enumerate(niveles_hue):
                subset = data[data[hue_col] == nivel]
                color = palette[i]
                axis.vlines(x=subset[x_col], ymin=subset[y_min], ymax=subset[y_max], color=color)
                axis.plot(subset[x_col], subset[y_min], '_', color=color, markersize=10, markeredgewidth=2)
                axis.plot(subset[x_col], subset[y_max], '_', color=color, markersize=10, markeredgewidth=2)
        else:
            axis.plot(data[x_col], data[y_col], 'p', markersize=8, color='steelblue')
            axis.vlines(x=data[x_col], ymin=data[y_min], ymax=data[y_max], color='steelblue')
            axis.plot(data[x_col], data[y_min], '_', color='steelblue', markersize=10, markeredgewidth=2)
            axis.plot(data[x_col], data[y_max], '_', color='steelblue', markersize=10, markeredgewidth=2)
            axis.set_xlabel(x_col)

        axis.set_title(title)
        axis.set_ylabel("Respuesta")
        axis.grid(True, alpha=0.3)

    x_var = factores[0]
    hue_var = factores[1] if len(factores) > 1 else None

    # Plot 1: Escala Modelo
    plot_intervals(ax[0], df_resultado, x_var, 'mean', 'mean_ci_lower', 'mean_ci_upper', hue_var, "Predicción (Escala Modelo)")

    # Plot 2: Escala Transformada (si existe)
    if funcion_aplicar:
        plot_intervals(ax[1], df_resultado, x_var, 'mean_trans', 'ci_lower_trans', 'ci_upper_trans', hue_var, f"Predicción ({nombre_trans})")

    plt.tight_layout()
    plt.show()

    return df_resultado


def ancova_selection(mod, metrica, datos):

    """
    Evalúa la métrica considerada para un conjunto de modelos.

    Parameters
    ----------
    mod: lista
    metrica: str
        Métrica utilizada para seleccionar el modelo. Debe ser una de las
        siguientes opciones: 'aic', 'bic'.
    datos: pdDataFrame
        Conjunto de datos sobre los que evaluar los modelos

    Returns
    -------
    modelo: list
        modelo con el mejor valor de la métrica.
    """

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    valores = []

    if metrica == 'aic':
      for i in range(len(mod)):
        ajuste = smf.ols(mod[i], data = datos).fit()
        valores.append(round(getattr(ajuste, metrica),4))
        pos = valores.index(min(valores))
    if metrica == 'bic':
      for i in range(len(mod)):
        ajuste = smf.ols(mod[i], data = datos).fit()
        valores.append(round(getattr(ajuste, metrica),4))
        pos = valores.index(min(valores))

    return mod[pos]


def estandarizar(df, lista_variables):
  """
  Estandariza las variables en el dataframe df, detalladas en la lista lista_variables.

  Parámetros:
    - df: base de datos
    - lista_variables: lista con los nombres de las variables numéricas a estandarizar
  Devuelve el mismo dataframe con una columna adicional por cada una de las contenidas en
  la lista, identificadas con el sufijo '_est'.
  """
  # Recorremos la lista con las variables numéricas
  for i in lista_variables:
    # para cada variable "i", le añadimos el sufijo _est.
    df[f"{i}_est"] = (df[i] - df[i].mean())/df[i].std()

def diagnostico_ancova(modelo, figsize=(16, 6)):
    """
    Realiza el diagnóstico de un modelo ANCOVA con gráficos detallados por grupo.
    
    Parámetros:
    -----------
    modelo : statsmodels result
        Modelo ajustado.
    figsize : tuple (ancho, alto)
        Tamaño del lienzo. 
        - Se aplica directamente a la Figura General.
        - Para la Figura Detallada, se usa el 'ancho', pero la 'altura' se 
          ajusta dinámicamente según la cantidad de grupos para evitar gráficos aplastados.
        
    Retorna:
    --------
    pd.DataFrame
        Tabla con los resultados de los tests de hipótesis (Levene y K-S).
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.outliers_influence import OLSInfluence
    import re

    # ==============================================================================
    # 1. AUTO-DETECCIÓN: SEPARAR FACTORES vs COVARIABLES
    # ==============================================================================
    try:
        df = modelo.model.data.frame.copy()
        col_response = modelo.model.endog_names
        terminos = modelo.model.data.design_info.term_names
        
        vars_todas = set()
        vars_forzadas_factor = set()

        for term in terminos:
            if term == 'Intercept': continue
            partes = term.split(':')
            for p in partes:
                match = re.search(r"C\((.*?)\)", p)
                if match:
                    nombre_limpio = match.group(1)
                    vars_todas.add(nombre_limpio)
                    vars_forzadas_factor.add(nombre_limpio)
                else:
                    vars_todas.add(p)
        
        cols_factores = []
        cols_covariables = []

        print(f"Diagnóstico ANCOVA Detallado para: {col_response}")
        print("-" * 60)
        
        for var in vars_todas:
            if var not in df.columns: continue 
            es_numerica = pd.api.types.is_numeric_dtype(df[var])
            es_factor_explicito = var in vars_forzadas_factor
            
            if (not es_numerica) or es_factor_explicito:
                cols_factores.append(var)
            else:
                cols_covariables.append(var)

        cols_factores.sort()
        cols_covariables.sort()

        print(f"Factores (Grupos): {cols_factores}")
        print(f"Covariables:       {cols_covariables}")

        if not cols_factores:
            df['Grupo_Comb'] = "Global"
            nombre_factor_eje = "Global"
        else:
            if len(cols_factores) > 1:
                df['Grupo_Comb'] = df[cols_factores].astype(str).agg(' & '.join, axis=1)
                nombre_factor_eje = f"Comb: {' & '.join(cols_factores)}"
            else:
                df['Grupo_Comb'] = df[cols_factores[0]].astype(str)
                nombre_factor_eje = cols_factores[0]

        print(f"Agrupación visual: {nombre_factor_eje}")
        print("-" * 60)

    except AttributeError:
        raise ValueError("El modelo no contiene metadatos suficientes.")

    # ==============================================================================
    # 2. CÁLCULO DE RESIDUOS
    # ==============================================================================
    prediccion = modelo.fittedvalues
    dt_resid = np.sqrt(modelo.mse_resid)
    residuos_std = modelo.resid / dt_resid
    
    indices_modelo = modelo.model.data.row_labels
    df_diag = df.loc[indices_modelo].copy()
    
    df_diag['Prediccion'] = prediccion
    df_diag['Residuos_Std'] = residuos_std
    df_diag['Residuos'] = modelo.resid 
    
    grupos_unicos = np.sort(df_diag['Grupo_Comb'].unique())
    n_grupos = len(grupos_unicos)

    # ==============================================================================
    # 3. GENERACIÓN DE GRÁFICOS
    # ==============================================================================
    
    # --- FIGURA 1: DIAGNÓSTICO GLOBAL (Usa figsize tal cual) ---
    fig1, ax1 = plt.subplots(1, 2, figsize=figsize)
    fig1.suptitle(f'Diagnóstico General: {col_response}', fontsize=16)
    
    # A) Homogeneidad (Boxplots)
    sns.boxplot(
        data=df_diag, x='Grupo_Comb', y='Residuos_Std', hue='Grupo_Comb', 
        legend=False, palette="viridis", order=grupos_unicos, ax=ax1[0]
    )
    ax1[0].axhline(y=0, color='red', linestyle='--')
    ax1[0].set_title(f'Homogeneidad de Varianzas ({nombre_factor_eje})')
    ax1[0].set_ylabel('Residuos Std')
    ax1[0].tick_params(axis='x', rotation=45)
    
    # B) Distancia de Cook
    infl = OLSInfluence(modelo)
    (c, p) = infl.cooks_distance
    ax1[1].stem(np.arange(len(c)), c, markerfmt=",")
    ax1[1].set_title('Puntos Influyentes: Distancia de Cook')
    ax1[1].set_ylabel('Distancia Cook')
    ax1[1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

    # --- FIGURA 2: DETALLE POR GRUPO (Dinámica) ---
    # Usamos el ANCHO de figsize, pero calculamos la ALTO necesaria
    ancho_fig = figsize[0]
    alto_fila = 4  # Altura fija por fila para legibilidad
    alto_total = alto_fila * n_grupos
    
    fig2, ax2 = plt.subplots(nrows=n_grupos, ncols=2, figsize=(ancho_fig, alto_total), squeeze=False)
    fig2.suptitle(f'Diagnóstico Detallado por Grupo: {nombre_factor_eje}', fontsize=16, y=1.005)

    for i, grupo in enumerate(grupos_unicos):
        subset = df_diag[df_diag['Grupo_Comb'] == grupo]
        
        # Columna 0: Q-Q Plot
        sm.qqplot(subset['Residuos_Std'], line='45', fit=True, ax=ax2[i, 0])
        ax2[i, 0].set_title(f'Normalidad (Q-Q): {grupo}', fontsize=11, fontweight='bold')
        ax2[i, 0].set_ylabel('Cuartiles Muestrales')
        
        # Columna 1: Residuos vs Predicción
        sns.scatterplot(x=subset['Prediccion'], y=subset['Residuos_Std'], ax=ax2[i, 1], color='steelblue', s=60, alpha=0.8)
        ax2[i, 1].axhline(y=0, color='red', linestyle='--')
        ax2[i, 1].set_title(f'Residuos vs Predicción: {grupo}', fontsize=11, fontweight='bold')
        ax2[i, 1].set_ylabel('Residuos Std')
        ax2[i, 1].set_xlabel('Valor Predicho')
        ax2[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ==============================================================================
    # 4. TESTS ESTADÍSTICOS (K-S y Levene)
    # ==============================================================================
    resultados_tests = []
    
    # Levene Global
    lista_residuos_grupos = [df_diag[df_diag['Grupo_Comb'] == g]['Residuos'] for g in grupos_unicos]
    
    if len(lista_residuos_grupos) > 1:
        stat_levene, p_levene = stats.levene(*lista_residuos_grupos, center='median')
        concl_levene = 'Varianzas Iguales (H0)' if p_levene > 0.05 else 'Varianzas Diferentes (H1)'
    else:
        stat_levene, p_levene = (np.nan, np.nan)
        concl_levene = "N/A (1 solo grupo)"

    resultados_tests.append({
        'Test': 'Levene (Homocedasticidad)',
        'Ámbito': f'Global: {nombre_factor_eje}',
        'P-Valor': round(p_levene, 4) if not np.isnan(p_levene) else "-",
        'Conclusión': concl_levene
    })
    
    # Kolmogorov-Smirnov por Grupo
    for g in grupos_unicos:
        datos_grupo = df_diag[df_diag['Grupo_Comb'] == g]['Residuos']
        
        if len(datos_grupo) >= 3:
            media_g = np.mean(datos_grupo)
            std_g = np.std(datos_grupo, ddof=1)
            # K-S contra normal ajustada
            stat_ks, p_ks = stats.kstest(datos_grupo, 'norm', args=(media_g, std_g))
            concl = 'Normal (H0)' if p_ks > 0.05 else 'No Normal (H1)'
        else:
            p_ks = np.nan
            concl = f"N={len(datos_grupo)} (Insuficiente)"

        resultados_tests.append({
            'Test': 'Kolmogorov-Smirnov',
            'Ámbito': f'Grupo: {g}',
            'P-Valor': round(p_ks, 4) if not np.isnan(p_ks) else "-",
            'Conclusión': concl
        })
        
    df_resultados = pd.DataFrame(resultados_tests)
    
    return df_resultados

def gof_test(fit):
    '''
    Evalúa el test Chi-cuadrado de la deviance (Bondad de ajuste).
    
    Hipótesis:
    H0: El modelo ajusta bien a los datos.
    H1: El modelo no ajusta bien a los datos.
    
    Parámetros:
    -----------
    fit : statsmodels result
        Modelo GLM ajustado (binomial/poisson).
        
    Retorna:
    --------
    float
        El p-valor del test.
    '''
    import scipy.stats as stats

    # 1. Extraemos la Deviance y los Grados de Libertad de los residuos
    deviance = fit.deviance
    df_resid = fit.df_resid
    
    # 2. Cálculo del P-Valor
    # Calculamos la probabilidad de encontrar una deviance mayor a la observada.
    pvalor = stats.chi2.sf(deviance, df_resid)
    
    # 3. Presentación de resultados
    print("-" * 40)
    print("Test de Bondad de Ajuste (Deviance)")
    print("-" * 40)
    print(f"Deviance del Modelo: {deviance:.4f}")
    print(f"Grados de Libertad:  {df_resid}")
    print(f"P-Valor:             {pvalor:.4f}")
    print("-" * 40)
    
    if pvalor > 0.05:
        print("✅ CONCLUSIÓN: El modelo AJUSTA BIEN (No rechazamos H0).")
        print("   No hay diferencia significativa entre el modelo ajustado y los datos.")
    else:
        print("⚠️ CONCLUSIÓN: El modelo NO AJUSTA bien (Rechazamos H0).")
        print("   Existe una discrepancia significativa con los datos observados.")
        
    return pvalor

def selection_glm(formulas, datos, metrica='aic', familia=sm.families.Binomial()):
    """
    Evalúa un conjunto de fórmulas para modelos GLM y selecciona el mejor según AIC o BIC.
    
    Parámetros:
    -----------
    formulas : list of str
        Lista de strings con las fórmulas a evaluar (ej: ['y ~ x1', 'y ~ x1 + x2']).
    datos : pd.DataFrame
        Conjunto de datos.
    metrica : str
        'aic' o 'bic'.
    familia : statsmodels.families
        Familia del GLM (por defecto Binomial, útil para regresión logística).
        
    Retorna:
    --------
    str
        La fórmula del mejor modelo.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    resultados = []
    nombre_metrica = metrica.upper()
    
    print(f"--- Selección de Modelos GLM ({nombre_metrica}) ---")
    print(f"Total de modelos a evaluar: {len(formulas)}")
    
    for i, formula in enumerate(formulas):
        try:
            # Ajustamos el modelo
            modelo = smf.glm(formula=formula, data=datos, family=familia).fit()
            
            # Obtenemos el valor de la métrica
            # Nota: En statsmodels GLM, el BIC basado en log-likelihood suele llamarse 'bic_llf'
            if metrica.lower() == 'bic':
                valor = modelo.bic_llf
            else:
                valor = modelo.aic
            
            resultados.append({
                'ID': i,
                'Formula': formula,
                nombre_metrica: valor,
                'Gl_Resid': modelo.df_resid
            })
            
        except Exception as e:
            print(f"⚠️ Error al ajustar modelo {formula}: {e}")

    # Convertimos a DataFrame para ordenar y visualizar
    df_res = pd.DataFrame(resultados)
    
    if df_res.empty:
        return None

    # Ordenamos de menor a mayor (menor AIC/BIC es mejor)
    df_res = df_res.sort_values(by=nombre_metrica, ascending=True).reset_index(drop=True)
    
    # Calculamos el Delta (diferencia respecto al mejor)
    mejor_valor = df_res.iloc[0][nombre_metrica]
    df_res['Delta'] = df_res[nombre_metrica] - mejor_valor
    
    # Mostramos la tabla
    print("-" * 80)
    # Formato de impresión limpio
    print(df_res[['ID', 'Formula', nombre_metrica, 'Delta']].to_string(index=False))
    print("-" * 80)
    
    mejor_formula = df_res.iloc[0]['Formula']
    print(f"✅ Mejor Modelo: {mejor_formula}")
    print(f"   {nombre_metrica}: {mejor_valor:.4f}")
    
    return mejor_formula
    
