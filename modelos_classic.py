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
