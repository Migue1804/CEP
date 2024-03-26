import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy.stats as stats

def main():
    st.image("CEP.jpg", width=720) 
    st.sidebar.header("Configuración del Gráfico de Control")
    variable_name = st.sidebar.text_input("Nombre de la Variable:")
    lower_spec_limit = st.sidebar.number_input("Límite Especificado Inferior:", step=0.01)
    upper_spec_limit = st.sidebar.number_input("Límite Especificado Superior:", step=0.01)

    # Ingreso de datos
    st.sidebar.subheader("Ingreso de Datos")
    num_data = st.sidebar.number_input("Número de Datos:", min_value=0, step=1, value=0)
    data_entries = []
    data_ranges = []
    for i in range(num_data):
        st.sidebar.subheader(f"Dato {i+1}")
        data_entry = st.sidebar.number_input(f"Ingrese el dato {i+1}:", step=0.01)
        data_entries.append(data_entry)

        # Calcular rango móvil solo si hay al menos dos datos ingresados
        if len(data_entries) >= 2:
            data_range = max(data_entries[-2:]) - min(data_entries[-2:])
            data_ranges.append(data_range)
        else:
            data_ranges.append(np.nan)  # Agregar NaN para mantener las longitudes iguales
    
    # Convertir la lista de datos en un DataFrame
    data_df = pd.DataFrame({"Datos": data_entries, "Rango": data_ranges})

    # Mostrar datos ingresados
    st.subheader("Datos Ingresados")
    if not data_df.empty:
        st.write(data_df)  # Mostramos la tabla con los datos

    # Gráficos de Control
    st.subheader("Gráficos de Control")
    if not data_df.empty:
        # Crear gráfico de Medias
        fig_means = go.Figure()
        fig_means.add_trace(go.Scatter(x=np.arange(1, len(data_entries)+1), y=data_entries, mode='lines+markers', name='Lecturas'))
        fig_means.add_hline(y=np.mean(data_entries), line_dash="dash", line_color="red", annotation_text="Media Global", annotation_position="bottom right")
        fig_means.update_layout(title="Gráfico de Medias",
                          xaxis_title="Lecturas",
                          yaxis_title="Valores")
        
        # Crear gráfico de Rangos
        fig_ranges = go.Figure()
        fig_ranges.add_trace(go.Scatter(x=np.arange(1, len(data_ranges)+1), y=data_ranges, mode='lines+markers', name='Rangos'))
        fig_ranges.add_hline(y=np.nanmean(data_ranges), line_dash="dash", line_color="blue", annotation_text="Media de Rangos", annotation_position="top right")
        fig_ranges.update_layout(title="Gráfico de Rangos",
                          xaxis_title="Lecturas",
                          yaxis_title="Valores")
        
        st.plotly_chart(fig_means)
        st.plotly_chart(fig_ranges)

    # Histograma y Curva Normal
    st.subheader("Histograma y Curva Normal")
    if not data_df.empty:
        fig_hist_normal = go.Figure()

        # Histograma
        fig_hist_normal.add_trace(go.Histogram(x=data_df["Datos"], name='Histograma'))

        # Curva Normal
        mu = np.mean(data_df["Datos"])
        sigma = np.std(data_df["Datos"])
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = stats.norm.pdf(x, mu, sigma)
        fig_hist_normal.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Curva Normal'))

        fig_hist_normal.update_layout(title="Histograma y Curva Normal",
                                      xaxis_title="Valores",
                                      yaxis_title="Frecuencia / Densidad de Probabilidad")
        
        st.plotly_chart(fig_hist_normal)

    # Índices de Capacidad de Procesos
    st.subheader("Índices de Capacidad de Procesos")
    if not data_df.empty:
        Cp = (upper_spec_limit - lower_spec_limit) / (6 * np.std(data_df["Datos"]))
        Cpk = min((np.mean(data_df["Datos"]) - lower_spec_limit) / (3 * np.std(data_df["Datos"])), (upper_spec_limit - np.mean(data_df["Datos"])) / (3 * np.std(data_df["Datos"])))

        st.write(f"Índice de Capacidad del Proceso (Cp): {Cp}")
        st.write(f"Índice de Capacidad del Proceso Ajustado (Cpk): {Cpk}")

    # DPMO y Nivel Sigma
    st.subheader("DPMO y Nivel Sigma")
    if not data_df.empty:
        DPMO_upper = stats.norm.sf((upper_spec_limit - np.mean(data_df["Datos"])) / np.std(data_df["Datos"])) * 1e6
        DPMO_lower = stats.norm.sf((np.mean(data_df["Datos"]) - lower_spec_limit) / np.std(data_df["Datos"])) * 1e6
        sigma = min((upper_spec_limit - np.mean(data_df["Datos"])) / np.std(data_df["Datos"]), (np.mean(data_df["Datos"]) - lower_spec_limit) / np.std(data_df["Datos"])) / 3

        st.write(f"DPMO Superior: {DPMO_upper}")
        st.write(f"DPMO Inferior: {DPMO_lower}")
        st.write(f"Nivel Sigma: {sigma}")

if __name__ == "__main__":
    main()
