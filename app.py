import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy.stats as stats

def main():
    st.image("CEP.jpg", width=720)
    with st.sidebar: 
        st.title("Instrucciones")
        st.write("¡Bienvenido a la aplicación de control de calidad!")
        st.markdown("Por favor sigue las instrucciones paso a paso para utilizar la herramienta correctamente.")
        
        # Pasos con diferentes formatos
        st.markdown("### Pasos:")
        st.markdown("- **Paso 1:** Ingresa el nombre de la variable.")
        st.markdown("- **Paso 2:** Establece los límites específicados inferior y superior.")
        st.markdown("- **Paso 3:** Ingresa los datos en la sección 'Ingreso de Datos'. Cada vez que aumentes el número de datos, se habilitará un campo para ingresar un nuevo dato.")
        st.markdown("- **Paso 4:** Observa los gráficos de control, histograma y curva normal.")
        st.markdown("- **Paso 5:** Analiza los índices de capacidad de proceso y el nivel Sigma.")
        st.markdown("- **Paso 6:** ¡Listo! Puedes visualizar los datos ingresados en la sección 'Datos Ingresados'.")     
        # Opciones adicionales para agregar iconos o imágenes
        st.markdown("### Información Adicional:")
        st.markdown("👉 **Para más información: [LinkedIn](https://www.linkedin.com/in/josemaguilar/)**")

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


    # Gráficos de Control
    #st.subheader("Gráficos de Control")
    if not data_df.empty:
        # Calcular estadísticas para el gráfico de control
        media_datos = np.mean(data_df["Datos"])
        rango_moving = data_df["Rango"].dropna()  # Eliminar NaN
        media_rango_moving = np.mean(rango_moving)
        limite_superior_i = media_datos + (2.66 * media_rango_moving)
        limite_inferior_i = media_datos - (2.66 * media_rango_moving)
        # Calcular límites de control estadístico para el rango móvil
        limite_superior_rango_moving = media_rango_moving + (2.66 * media_rango_moving)
        limite_inferior_rango_moving = media_rango_moving - (2.66 * media_rango_moving)

        # Crear el gráfico de control I-MR en Plotly
        fig_i_mr = go.Figure()

        # Calcular límites de especificación
        df_proceso = data_df.copy()  # Copiar los datos para el gráfico de control
        df_proceso['LSE'] = upper_spec_limit
        df_proceso['LIE'] = lower_spec_limit

        fig_i_mr.add_trace(go.Scatter(x=np.arange(1, len(data_entries) + 1), y=data_entries, mode='markers+lines', name='Datos'))
        fig_i_mr.add_trace(go.Scatter(x=np.arange(1, len(data_entries) + 1), y=[media_datos] * len(data_entries), mode='lines', name='Media', line=dict(color='blue', dash='dot')))
        fig_i_mr.add_trace(go.Scatter(x=np.arange(1, len(data_entries) + 1), y=[limite_superior_i] * len(data_entries), mode='lines', name='Límite Superior Control', line=dict(color='blue', dash='dash')))
        fig_i_mr.add_trace(go.Scatter(x=np.arange(1, len(data_entries) + 1), y=[limite_inferior_i] * len(data_entries), mode='lines', name='Límite Inferior Control', line=dict(color='blue', dash='dash')))
        fig_i_mr.add_trace(go.Scatter(x=np.arange(1, len(data_entries) + 1), y=[upper_spec_limit] * len(data_entries), mode='lines', name='Límite Superior Especificación', line=dict(color='red')))
        fig_i_mr.add_trace(go.Scatter(x=np.arange(1, len(data_entries) + 1), y=[lower_spec_limit] * len(data_entries), mode='lines', name='Límite Inferior Especificación', line=dict(color='red')))

        # Agregar rectángulos horizontales para las zonas de alerta
        zona_roja = go.layout.Shape(
            type="rect",
            x0=1,
            x1=len(data_entries),
            y0=lower_spec_limit,
            y1=lower_spec_limit + ((upper_spec_limit - lower_spec_limit) / 2) / 3,
            fillcolor="red",
            opacity=0.3,
            layer="below",
            line_width=0,
        )

        zona_amarilla = go.layout.Shape(
            type="rect",
            x0=1,
            x1=len(data_entries),
            y0=lower_spec_limit + ((upper_spec_limit - lower_spec_limit) / 2) / 3,
            y1=lower_spec_limit + ((upper_spec_limit - lower_spec_limit) / 2) / 3 * 2,
            fillcolor="yellow",
            opacity=0.3,
            layer="below",
            line_width=0,
        )

        zona_verde = go.layout.Shape(
            type="rect",
            x0=1,
            x1=len(data_entries),
            y0=lower_spec_limit + ((upper_spec_limit - lower_spec_limit) / 2) / 3 * 2,
            y1=lower_spec_limit + ((upper_spec_limit - lower_spec_limit) / 2) / 3 * 4,
            fillcolor="green",
            opacity=0.3,
            layer="below",
            line_width=0,
        )

        zona_roja2 = go.layout.Shape(
            type="rect",
            x0=1,
            x1=len(data_entries),
            y0=lower_spec_limit + ((upper_spec_limit - lower_spec_limit) / 2) / 3 * 5,
            y1=lower_spec_limit + ((upper_spec_limit - lower_spec_limit) / 2) / 3 * 6,
            fillcolor="red",
            opacity=0.3,
            layer="below",
            line_width=0,
        )

        zona_amarilla2 = go.layout.Shape(
            type="rect",
            x0=1,
            x1=len(data_entries),
            y0=lower_spec_limit + ((upper_spec_limit - lower_spec_limit) / 2) / 3 * 4,
            y1=lower_spec_limit + ((upper_spec_limit - lower_spec_limit) / 2) / 3 * 5,
            fillcolor="yellow",
            opacity=0.3,
            layer="below",
            line_width=0,
        )

        fig_i_mr.update_layout(
            title='Gráfico de Control I-MR',
            xaxis_title='Muestra',
            yaxis_title='Datos',
            shapes=[zona_roja, zona_amarilla, zona_verde, zona_amarilla2, zona_roja2],  # Agregar las zonas de alerta
            legend=dict(x=0, y=-0.2, orientation='h')  # Posición de la leyenda en la parte inferior y centrada horizontalmente
        )

        st.plotly_chart(fig_i_mr)

        # Crear gráfico de Rangos
        fig_ranges = go.Figure()
        fig_ranges.add_trace(go.Scatter(x=np.arange(1, len(data_ranges)+1), y=data_ranges, mode='lines+markers', name='Rangos'))
        fig_ranges.add_hline(y=np.nanmean(data_ranges), line_dash="dot", line_color="blue", annotation_text="Media de Rangos", annotation_position="top right")
        fig_ranges.add_trace(go.Scatter(x=np.arange(1, len(data_ranges) + 1), y=[limite_superior_rango_moving] * len(data_ranges), mode='lines', name='Límite Superior Control', line=dict(color='blue', dash='dash')))
        fig_ranges.add_trace(go.Scatter(x=np.arange(1, len(data_ranges) + 1), y=[limite_inferior_rango_moving] * len(data_ranges), mode='lines', name='Límite Inferior Control', line=dict(color='blue', dash='dash')))
        fig_ranges.update_layout(title="Gráfico de Rangos",
                          xaxis_title="Lecturas",
                          yaxis_title="Valores",
                          legend=dict(x=0, y=-0.2, orientation='h') )
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

    # Mostrar datos ingresados
    st.subheader("Datos Ingresados")
    if not data_df.empty:
        st.write(data_df)  # Mostramos la tabla con los datos

if __name__ == "__main__":
    main()
