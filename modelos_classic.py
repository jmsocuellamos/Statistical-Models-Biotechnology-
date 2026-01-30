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
