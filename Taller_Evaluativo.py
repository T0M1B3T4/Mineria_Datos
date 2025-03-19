# Actividad 1 EDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1.1 
ruta = "/home/T0M1/Desktop/T0M1/Universidad_Semestre_7/Minería De Datos/DataSets_Minería/dataset_ninos_vulnerables.csv"
df = pd.read_csv(ruta)  

#1.2
#Con las siguientes líneas de código damos un "vistazo" o "sondeo" a lo que contiene el dataset
print(df.info())
print("\nEstadísticas Generales\n", df.describe())
print("\nPrimeros 5 registros:\n", df.head())
print("\nUltimos 5 registros:\n", df.tail())
print("Data Cleaning")
print(df.isnull().sum()) # --> Observar cuantas filas y columnas están sin datos y mostrar la la sumatoria final
print(f"Filas duplicadas: {df.duplicated().sum()}") # --> Mostrar las filas duplicadas
df = df.drop_duplicates() # --> luego eliminar si hay registros duplicados y, eventualmente, eliminarlos

estandarizacion_columnas = [
    "Genero", "Condiciones_Vivienda", "Trabajo_Padres", "Acceso_Subsidios",
    "Estado_Nutricional", "Afiliacion_Salud", "Vacunacion", "Enfermedades_Cronicas",
    "Nivel_Educativo", "Desercion_Escolar", "Acceso_Internet", "Infraestructura_Educativa",
    "Violencia_Intrafamiliar", "Trabajo_Infantil", "Riesgo_Reclutamiento", "Explotacion_Abuso",
    "Migrante_Desplazado", "Programas_Apoyo", "Actividades_Extracurriculares"
]

for col in estandarizacion_columnas:
    df[col] = df[col].str.strip().str.title() 

df_mayores_10 = df[df["Edad"] > 10]  # Filtrar niños mayores de 10 años
print("\nNiños Mayores a 10 años\n", df_mayores_10.head())

#Actividad 2.1
# Estadísticas Generales y Descriptivas:

print("\n--> 2.1.1 Medidas de tendencia central <--\n")
print("Media de ingresos:", df["Ingresos_Hogar"].mean())
print("Mediana de ingresos:", df["Ingresos_Hogar"].median())
print("Moda de ingresos:", df["Ingresos_Hogar"].mode()[0])

print("\n --> 2.1.2 Medidas de dispersión <-- \n")
print("Varianza de ingresos:", df["Ingresos_Hogar"].var())
print("Desviación estándar de ingresos:", df["Ingresos_Hogar"].std())
print("Rango de ingresos:", df["Ingresos_Hogar"].max() - df["Ingresos_Hogar"].min())

# Histograma De Los Ingresos punto 2.1.3
plt.figure(figsize=(8, 5))
sns.histplot(df["Ingresos_Hogar"], bins=30, kde=True)
plt.title("Distribución de Ingresos del Hogar")
plt.xlabel("Ingresos del Hogar")
plt.ylabel("Frecuencia")
plt.show()

# Diagrama De Caja punto 2.1.3
plt.figure(figsize=(8, 5))
sns.boxplot(x="Estado_Nutricional", y="Ingresos_Hogar", data=df)
plt.title("Distribución de Ingresos por Estado Nutricional")
plt.xlabel("Estado Nutricional")
plt.ylabel("Ingresos del Hogar")
plt.show()

# --> Estadística Inferencial 2.2.1 - 2.2.3 <--
# Grupos de ingresos según estado nutricional 2.2.1
normal = df[df["Estado_Nutricional"] == "Normal"]["Ingresos_Hogar"]
sobrepeso = df[df["Estado_Nutricional"] == "Sobrepeso"]["Ingresos_Hogar"]
desnutricion = df[df["Estado_Nutricional"] == "Desnutrición"]["Ingresos_Hogar"]

# Prueba ANOVA o Prueba de Hipotesis
anova_result = stats.f_oneway(normal, sobrepeso, desnutricion)
print("\n --> 2.2.1 Prueba de hipótesis (ANOVA) <--\n")
print(f"F={anova_result.statistic}, p={anova_result.pvalue}")

if anova_result.pvalue < 0.05:
    print("Rechazamos H₀: Hay diferencias significativas entre los ingresos y el estado nutricional.")
else:
    print("No podemos rechazar H₀: No hay suficiente evidencia para afirmar diferencias significativas.")

# Intervalo de confianza para ingresos en niños con nutrición normal 2.2.2
confianza = 0.95
media = normal.mean()
std_error = stats.sem(normal)  # Error estándar
intervalo = stats.t.interval(confianza, len(normal)-1, loc=media, scale=std_error)

print("\n2.2.2 Intervalos de confianza\n")
print(f"Intervalo de confianza del 95% para ingresos (Nutrición normal): {intervalo}")

# Regresión Lineal 2.2.3

X = df[["Edad"]]  # Variable independiente
y = df["Ingresos_Hogar"]  # Variable dependiente

# Modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Coeficientes
print("\n📊 2.2.3 Regresión lineal")
print(f"Coeficiente de regresión (pendiente): {modelo.coef_[0]}")
print(f"Intercepto: {modelo.intercept_}")

# Predicción de ingresos para niños de 10 años
edad_10 = np.array([[10]])
ingreso_predicho = modelo.predict(edad_10)
print(f"Predicción de ingreso para un niño de 10 años: {ingreso_predicho[0]}")

# Gráfico de regresión
plt.figure(figsize=(8, 5))
sns.regplot(x="Edad", y="Ingresos_Hogar", data=df, line_kws={"color": "red"})
plt.title("Regresión Lineal: Edad vs Ingresos del Hogar")
plt.xlabel("Edad")
plt.ylabel("Ingresos del Hogar")
plt.show()
#Modulo 3:
#PCA
columnas_numericas = ["Edad", "Ingresos_Hogar"] #Selección de las variables
X = df[columnas_numericas]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Estado_Nutricional"] = df["Estado_Nutricional"]

plt.figure(figsize=(8, 5))
sns.scatterplot(x="PC1", y="PC2", hue="Estado_Nutricional", data=df_pca, palette="viridis")
plt.title("PCA - Reducción de Dimensionalidad")
plt.show()
#LDA

X = df[["Edad", "Ingresos_Hogar"]]
y = df["Afiliacion_Salud"]  # Variable categórica

# Aplicar LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

# Convertir a DataFrame
df_lda = pd.DataFrame(X_lda, columns=["LDA1"])
df_lda["Afiliacion_Salud"] = y

# Gráfico LDA
plt.figure(figsize=(8, 5))
sns.boxplot(x="Afiliacion_Salud", y="LDA1", data=df_lda)
plt.title("LDA - Diferencias en acceso a salud")
plt.show()
