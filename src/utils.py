import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from IPython.display import display
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import *

#Funciones IA


def analisis_univariado_categorico(df, exclude_cols=None, target=None):
    if exclude_cols is None:
        exclude_cols = []

    # columnas categóricas a analizar
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    categorical_cols = [c for c in cat_cols if c not in exclude_cols]
    n_vars = len(categorical_cols)
    if n_vars == 0:
        print("No hay columnas categóricas que analizar.")
        return

    # figura y ejes
    fig, axes = plt.subplots(n_vars, 2, figsize=(14, 5 * n_vars))
    fig.subplots_adjust(top=0.93, hspace=0.6, wspace=0.4, right=0.88)
    fig.suptitle("Análisis Univariado de Variables Categóricas", fontsize=15)

    if n_vars == 1:
        axes = [axes]

    for i, col in enumerate(categorical_cols):
        ax0, ax1 = axes[i]

        # Countplot
        order = df[col].value_counts().index
        if target:
            sns.countplot(x=col, data=df, hue=target, ax=ax0,
                          order=order, dodge=False)
        else:
            sns.countplot(x=col, data=df, ax=ax0,
                          order=order, dodge=False)
        ax0.set_title(f"Frecuencia - {col}" + (f" por {target}" if target else ""))
        ax0.tick_params(axis='x', rotation=45)

        # Pie chart
        counts = df[col].value_counts()
        counts.plot.pie(ax=ax1,
                        labels=None,
                        startangle=140,
                        colors=plt.cm.Paired.colors,
                        textprops={'fontsize': 9})
        ax1.set_title(f"Distribución - {col}")
        ax1.set_ylabel("")

        # Legend on its own axis, to the right of ax1
        labels = [f"{c} ({v/sum(counts)*100:.1f}%)" for c, v in counts.items()]
        ncol = 1 if len(labels) < 10 else 2
        ax1.legend(
            labels=labels,
            title=col,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize='small',
            title_fontsize='small',
            ncol=ncol,
            frameon=False
        )

    plt.show()


def analisis_bivariado_categorico(
    df,
    target,
    positive_label=">50K",
    exclude_cols=None,
    figsize_base=(14, 5)
):
    """
    Análisis bivariable (categórica vs target binario):
      • Frecuencia absoluta por categoría y target
      • Proporción de la clase positiva por categoría
      • Test chi-cuadrado de independencia anotado en el gráfico de proporciones
      
      Símbolo | p‑valor | Interpretación
        *** | p < 0.001 | Evidencia muy fuerte de asociación
        ** | p < 0.01 | Evidencia fuerte
        * | p < 0.05 | Evidencia moderada
        ns | p ≥ 0.05 | No hay evidencia significativa
    """
    if exclude_cols is None:
        exclude_cols = []

    # columnas categóricas a analizar
    cat_cols = (
        df.select_dtypes(include=["object", "category", "bool"])
          .columns.difference([target] + exclude_cols)
          .tolist()
    )
    if not cat_cols:
        print("No hay columnas categóricas que analizar.")
        return

    n = len(cat_cols)
    fig, axes = plt.subplots(n, 2,
                             figsize=(figsize_base[0],
                                      figsize_base[1] * n))
    fig.suptitle("Análisis Bivariable (categórica vs income)",
                 fontsize=15, y=0.995)
    axes = axes if n > 1 else [axes]

    for i, col in enumerate(cat_cols):
        ax_freq, ax_prop = axes[i]

        # ------------ 1) Frecuencia absoluta ------------
        sns.countplot(data=df, x=col, hue=target,
                      order=df[col].value_counts().index,
                      ax=ax_freq)
        ax_freq.set_title(f"{col} – Frecuencia absoluta")
        ax_freq.set_xlabel("")
        ax_freq.set_ylabel("Recuento")
        ax_freq.tick_params(axis='x', rotation=45)

        # ------------ 2) Proporción de la clase positiva ------------
        prop = (
            df.groupby(col)[target]
              .apply(lambda s: (s == positive_label).mean())
              .loc[ df[col].value_counts().index ]
        )
        sns.barplot(x=prop.index, y=prop.values,
                    color="orange", ax=ax_prop)
        ax_prop.set_title(f"{col} – % {positive_label}")
        ax_prop.set_ylabel("Proporción")
        ax_prop.set_xlabel("")
        ax_prop.set_ylim(0, 1)
        ax_prop.tick_params(axis='x', rotation=45)
        # Anotar porcentaje encima de cada barra
        for p in ax_prop.patches:
            ax_prop.annotate(f"{p.get_height():.1%}",
                             (p.get_x() + p.get_width() / 2,
                              p.get_height()),
                             ha='center', va='bottom', fontsize=8)

        # ------------ 3) Test chi-cuadrado ------------
        contingency = pd.crosstab(df[col], df[target])
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        ax_prop.text(
            0.95, 0.9,
            f"p = {p_val:.3f}  {sig}",
            transform=ax_prop.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8)
        )

    plt.tight_layout()
    plt.show()


def analisis_univariado_numerico(
    df,
    exclude_numeric_cols=None,
    bins=40,
    apply_log_cols=None,
    orig_color="C0",
    log_color="C1"
):
    """
    Análisis univariado de variables numéricas:
      - Histograma con KDE + boxplot en escala original
      - Opcionalmente, Histograma con KDE + boxplot en escala log1p justo debajo,
        usando distinto color para distinguir ambas.
    
    Parámetros:
    - df: DataFrame con datos.
    - exclude_numeric_cols: lista de columnas a excluir.
    - bins: número de bins para los histogramas.
    - apply_log_cols: lista de columnas para las que dibujar también la versión log1p.
    - orig_color: color para la versión original.
    - log_color: color para la versión logarítmica.
    """
    if exclude_numeric_cols is None:
        exclude_numeric_cols = []
    if apply_log_cols is None:
        apply_log_cols = []

    # Generar tareas: (col, is_log)
    tasks = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in exclude_numeric_cols:
            continue
        tasks.append((col, False))
        if col in apply_log_cols:
            tasks.append((col, True))

    if not tasks:
        print("No hay columnas numéricas para analizar.")
        return

    # Crear figura
    n = len(tasks)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    fig.suptitle("Análisis Univariado de Variables Numéricas", fontsize=16, y=1.02)

    if n == 1:
        axes = [axes]

    for i, (col, is_log) in enumerate(tasks):
        ax_hist, ax_box = axes[i]
        data = df[col].dropna()

        # Determinar color y etiquetas
        if is_log:
            data = np.log1p(data)
            color = log_color
            title_suffix = " (log1p)"
            xlabel = f"log1p({col})"
        else:
            color = orig_color
            title_suffix = ""
            xlabel = col

        # Histograma + KDE
        if col == 'hours-per-week' and not is_log:
            sns.histplot(data, binwidth=1, kde=True, color=color, ax=ax_hist)
        else:
            sns.histplot(data, bins=bins, kde=True, color=color, ax=ax_hist)

        ax_hist.set_title(f"Histograma (KDE){title_suffix} - {col}")
        ax_hist.set_xlabel(xlabel)
        ax_hist.set_ylabel("Frecuencia")

        # Boxplot horizontal
        sns.boxplot(x=data, color=color, ax=ax_box, orient='h')
        ax_box.set_title(f"Boxplot{title_suffix} - {col}")
        ax_box.set_xlabel(xlabel)

    plt.tight_layout()
    plt.show()


def analizar_multicolinealidad(df, target=None, corr_method='pearson', top_n=10, corr_limit=0.8):
    """
    1) Muestra la correlación ABSOLUTA de cada variable numérica con el target (usando Pearson y Spearman),
       ordenadas por fuerza, en una tabla (si el target es numérico y existe).

    2) Muestra, en una única tabla, las parejas de variables de entrada (excluyendo el target)
       que tienen alta correlación entre sí (valor absoluto >= corr_limit), es decir, aquellas que pueden estar
       explicando lo mismo. Se limitan a mostrar las top_n parejas.

    Parámetros:
       df (pd.DataFrame): DataFrame de entrada.
       target (str, opcional): Columna objetivo (numérica) para evaluar correlaciones.
       corr_method (str): Método de correlación para evaluar redundancia ('pearson', 'spearman', 'kendall').
       top_n (int): Número máximo de parejas a mostrar.
       corr_limit (float): Umbral mínimo de correlación para considerar alta la relación entre variables.
    """
    # Validación básica del DataFrame y columnas numéricas
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df debe ser un DataFrame válido y no vacío.")
        
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        raise ValueError("El DataFrame debe contener al menos dos columnas numéricas.")
    
    # 1) Correlación con el target (valor absoluto)
    if target and target in num_cols:
        pearson_corr = df[num_cols].corr(method='pearson')[target].abs()
        spearman_corr = df[num_cols].corr(method='spearman')[target].abs()

        corr_target = pd.DataFrame({'Pearson': pearson_corr, 'Spearman': spearman_corr})
        # Eliminar la fila del target
        corr_target.drop(index=target, inplace=True)
        corr_target = corr_target.sort_values('Pearson', ascending=False)
        
        print(f"\nCorrelación ABSOLUTA de variables con el target '{target}':")
        display(corr_target.head(top_n))
    else:
        print("No se analizará el target (no es numérico o no está definido).")
    
    # 2) Multicolinealidad: análisis de redundancia entre variables de entrada (excluyendo el target)
    input_vars = [col for col in num_cols if col != target] if target in num_cols else num_cols.copy()
    
    corr_matrix = df[input_vars].corr(method=corr_method).abs()
    # Extraer la parte superior triangular sin duplicados ni diagonal
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_pairs = corr_matrix.where(mask).stack().sort_values(ascending=False)
    
    high_corr = corr_pairs[corr_pairs >= corr_limit]
    
    print(f"\nParejas de variables de entrada con correlación >= {corr_limit} (posible redundancia):")
    if not high_corr.empty:
        high_corr_df = high_corr.reset_index()
        high_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlación']
        display(high_corr_df.head(top_n))
    else:
        print("No hay parejas de variables que cumplan ese umbral.")


def analizar_asociaciones_categoricas(df, cat_limit=0.3, top_n=10, exclude_cols=None):
    """
    Analiza la fuerza de asociación (Cramér’s V) entre variables categóricas de un DataFrame,
    con opción de excluir ciertas columnas.

    Guía para los gráficos
    

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene al menos dos columnas categóricas.
        cat_limit (float): Umbral para mostrar asociaciones 'fuertes'. Por defecto 0.3.
        top_n (int): Número máximo de parejas de variables a mostrar.
        exclude_cols (list): Lista de columnas que deseas excluir del análisis.

    Salida (print en consola):
        - Parejas de columnas categóricas con Cramér’s V >= cat_limit
          (ordenadas de mayor a menor asociación).
        - Si no se cumplen los criterios, un aviso correspondiente.

    Ejemplo de uso:
        analizar_asociaciones_categoricas(mi_df, cat_limit=0.4, top_n=5, exclude_cols=['ID', 'nombre'])
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df debe ser un DataFrame válido y no vacío.")

    # Creamos la lista de columnas a excluir si no se proporciona
    if exclude_cols is None:
        exclude_cols = []

    # Identificamos columnas categóricas, excluyendo las que especifiques
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in exclude_cols]

    if len(cat_cols) < 2:
        print("No hay suficientes columnas categóricas (excluyendo las indicadas) para analizar asociaciones.")
        return

    resultados = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            col1 = cat_cols[i]
            col2 = cat_cols[j]

            tabla_cruzada = pd.crosstab(df[col1], df[col2])
            chi2, p, dof, expected = chi2_contingency(tabla_cruzada)

            n = tabla_cruzada.sum().sum()
            min_dim = min(tabla_cruzada.shape) - 1
            if min_dim == 0:
                cv = np.nan
            else:
                cv = np.sqrt(chi2 / (n * min_dim))

            resultados.append((col1, col2, cv))
    
    df_cramers = pd.DataFrame(resultados, columns=['Variable 1', 'Variable 2', 'Cramér’s V'])
    df_cramers.sort_values(by='Cramér’s V', ascending=False, inplace=True)
    
    df_cramers_fuertes = df_cramers[df_cramers['Cramér’s V'] >= cat_limit]

    print(f"\nParejas de variables categóricas (excluyendo {exclude_cols}) con Cramér’s V >= {cat_limit}:")
    if not df_cramers_fuertes.empty:
        display(df_cramers_fuertes.head(top_n))
    else:
        print("No hay parejas de variables categóricas que cumplan ese umbral.")
    
    print("\n--- Fin del análisis de asociaciones categóricas ---")



def plot_num_t_test(
    df: pd.DataFrame,
    target: str = "income",
    positive_label: str = ">50K",
    exclude_cols: list | None = None,
    bins: int = 40,
    colors: tuple = ("#4C72B0", "#DD8452"),  # (negativo, positivo)
    kde: bool = True
):
    """
    Para cada variable numérica en df (salvo las excluidas):
      • Histograma por clase con alpha diferenciado
      • KDE opcional (sin aparecer en la leyenda)
      • t-test de Welch anotado con p‑valor (decimal) + significancia




      Guía para los gráficos:
        *** → p < 0.001 → Altamente significativo

        ** → p < 0.01 → Muy significativo

        * → p < 0.05 → Significativo

        ns → p ≥ 0.05 → No significativo (not significant)


    """
    if exclude_cols is None:
        exclude_cols = []

    # 1) Detectar columnas numéricas
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    if not num_cols:
        print("No hay variables numéricas para analizar.")
        return

    # 2) Configuración de clases y colores
    hue_order = [lbl for lbl in df[target].unique() if lbl != positive_label] + [positive_label]
    pal = dict(zip(hue_order, colors))

    # 3) Loop por cada variable
    for col in num_cols:
        plt.figure(figsize=(7, 4))
        ax = plt.gca()

        cls0, cls1 = hue_order  # negativa, positiva

        # Histograma clase negativa
        sns.histplot(
            df[df[target] == cls0][col],
            bins=bins, color=pal[cls0], alpha=0.75,
            label=cls0, kde=kde
        )

        # Histograma clase positiva
        sns.histplot(
            df[df[target] == cls1][col],
            bins=bins, color=pal[cls1], alpha=0.75,
            label=cls1, kde=False
        )

        # KDE clase positiva sin label (no sale en leyenda)
        if kde:
            sns.kdeplot(
                df[df[target] == cls1][col],
                color=pal[cls1], linewidth=2.0,
                label=None
            )

        # Leyenda limpia (solo dos entradas)
        ax.legend(title=target)

        # t‑test de Welch
        grp0 = df[df[target] == cls0][col].dropna()
        grp1 = df[df[target] == cls1][col].dropna()
        _, p_val = ttest_ind(grp0, grp1, equal_var=False)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        # P‑valor en decimal con 3 decimales
        ax.text(
            0.98, 0.95,
            f"p = {p_val:.3f}  {sig}",
            ha="right", va="top",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
        )

        ax.set_title(f"{col} – Distribución por clase")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.show()


def analizar_multicolinealidad(df, target=None, corr_method='pearson', top_n=10, corr_limit=0.8):
    """
    1) Muestra la correlación ABSOLUTA de cada variable numérica con el target (usando Pearson y Spearman),
       ordenadas por fuerza, en una tabla (si el target es numérico y existe).

    2) Muestra, en una única tabla, las parejas de variables de entrada (excluyendo el target)
       que tienen alta correlación entre sí (valor absoluto >= corr_limit), es decir, aquellas que pueden estar
       explicando lo mismo. Se limitan a mostrar las top_n parejas.

    Parámetros:
       df (pd.DataFrame): DataFrame de entrada.
       target (str, opcional): Columna objetivo (numérica) para evaluar correlaciones.
       corr_method (str): Método de correlación para evaluar redundancia ('pearson', 'spearman', 'kendall').
       top_n (int): Número máximo de parejas a mostrar.
       corr_limit (float): Umbral mínimo de correlación para considerar alta la relación entre variables.
    """
    # Validación básica del DataFrame y columnas numéricas
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df debe ser un DataFrame válido y no vacío.")
        
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        raise ValueError("El DataFrame debe contener al menos dos columnas numéricas.")
    
    # 1) Correlación con el target (valor absoluto)
    if target and target in num_cols:
        pearson_corr = df[num_cols].corr(method='pearson')[target].abs()
        spearman_corr = df[num_cols].corr(method='spearman')[target].abs()

        corr_target = pd.DataFrame({'Pearson': pearson_corr, 'Spearman': spearman_corr})
        # Eliminar la fila del target
        corr_target.drop(index=target, inplace=True)
        corr_target = corr_target.sort_values('Pearson', ascending=False)
        
        print(f"\nCorrelación ABSOLUTA de variables con el target '{target}':")
        display(corr_target.head(top_n))
    else:
        print("No se analizará el target (no es numérico o no está definido).")
    
    # 2) Multicolinealidad: análisis de redundancia entre variables de entrada (excluyendo el target)
    input_vars = [col for col in num_cols if col != target] if target in num_cols else num_cols.copy()
    
    corr_matrix = df[input_vars].corr(method=corr_method).abs()
    # Extraer la parte superior triangular sin duplicados ni diagonal
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_pairs = corr_matrix.where(mask).stack().sort_values(ascending=False)
    
    high_corr = corr_pairs[corr_pairs >= corr_limit]
    
    print(f"\nParejas de variables de entrada con correlación >= {corr_limit} (posible redundancia):")
    if not high_corr.empty:
        high_corr_df = high_corr.reset_index()
        high_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlación']
        display(high_corr_df.head(top_n))
    else:
        print("No hay parejas de variables que cumplan ese umbral.")



class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.limits_ = {}

    # --------------------------------------------------
    # Ajuste: guardamos los cuantiles límite por columna
    # --------------------------------------------------
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns
        for col in self.feature_names_in_:
            lo = X_df[col].quantile(self.lower)
            hi = X_df[col].quantile(self.upper)
            self.limits_[col] = (lo, hi)
        return self

    # -------------------------
    # Transformación winsorizada
    # -------------------------
    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        for col, (lo, hi) in self.limits_.items():
            X_df[col] = np.clip(X_df[col], lo, hi)
        # Devolvemos numpy para integrarlo en pipelines
        return X_df.values

    # -----------------------------------------------
    # NUEVO: nombres de salida para FeatureUnion/Pipe
    # -----------------------------------------------
    def get_feature_names_out(self, input_features=None):
        """Devuelve la lista de nombres de las columnas de salida."""
        # Si scikit‑learn pasa input_features, lo priorizamos;
        # en caso contrario usamos lo aprendido en fit
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(self.feature_names_in_, dtype=object)

    # Alias para compatibilidad con versiones <1.2
    get_feature_names = get_feature_names_out