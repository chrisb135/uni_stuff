import streamlit as st
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("Student_Performance.csv")
model_lr = joblib.load("lr_model.pkl")

st.title('Predição de Performance Estudantil')
st.sidebar.title('Orientação')
st.sidebar.write('Para realizar a previsão para vários alunos, atente-se que a tabela ou planilha enviada deve ter as seguintes 4 variáveis:')
st.sidebar.markdown(
"""
- Horas estudadas.
- Média de notas passadas.
- Média de horas dormidas.
- Quantidade de listas de exercícios resolvidas.
"""
)
selecao = st.selectbox('Escolha a forma de previsão:', ['Previsão em massa (CSV)', 'Previsão individual'])
if selecao == 'Previsão em massa (CSV)':
  st.header('Previsão em massa')
  uploaded_file = st.file_uploader('Escolha um arquivo CSV', type=['csv'])
  if uploaded_file is not None:
    alunos_csv = pd.read_csv(uploaded_file)
    alunos_csv = alunos_csv.drop('Performance Index', axis=1)
    alunos_csv = alunos_csv.drop('Extracurricular Activities', axis=1)
    alunos_csv = alunos_csv.values
  if st.button('Prever', 1):
    predicao = model_lr.predict(alunos_csv)
    resultado = pd.DataFrame(columns=['Índice de performance'])
    resultado['Índice de performance'] = predicao
    st.write(resultado)
elif selecao == 'Previsão individual':
  st.header('Previsão individual')
  horas_estudadas = st.slider("Horas estudadas", int(df['Hours Studied'].min()), int(df['Hours Studied'].max()))
  notas_passadas = st.slider("Média de notas passadas", int(df['Previous Scores'].min()), int(df['Previous Scores'].max()))
  horas_dormidas = st.slider("Média de horas dormidas", int(df['Sleep Hours'].min()), int(df['Sleep Hours'].max()))
  papers_practiced = st.slider("Listas de exercícios resolvidas", int(df['Sample Question Papers Practiced'].min()), int(df['Sample Question Papers Practiced'].max()))
  if st.button('Prever', 2):
    data = np.array([[horas_estudadas, notas_passadas, horas_dormidas, papers_practiced]])
    predicao = model_lr.predict(data)
    st.header('Resultado')
    st.write(predicao)