import streamlit as st
import joblib
import pandas as pd
import os

@st.cache_data
def carrega_modelo():
  caminho_modelo = os.path.join(os.path.dirname(__file__), 'modelo_titanic_logistic_regression.pkl')
  return joblib.load(caminho_modelo)

st.set_page_config(page_title='Titanic Logistic Regression', page_icon='🚢')

st.title('Informações do passageiro')
idade = st.number_input('Qual a idade do passageiro', min_value=0, max_value=100)
sexo_selecionado = st.selectbox('Qual o sexo do passageiro',['Feminino', 'Masculino'])
sexo = 1 if sexo_selecionado=='Feminino' else 0
classe_selecionada = st.selectbox('Selecione a classe do passageiro',['Primeira','Segunda', 'Terceira'])
mapa_classe = {'Primeira': 1, 'Segunda': 2, 'Terceira': 3}
classe = mapa_classe[classe_selecionada]
numero_irmaos = st.number_input('Quantos irmãos ou cônjuges tinha a bordo', min_value=0)
numero_filhos = st.number_input('Quantos pais ou filhos tinha a bordo', min_value=0)

if st.button('Analisar'):
  novo_passageiro = pd.DataFrame([{
    'age': idade,
    'sex': sexo,
    'pclass': classe,
    'parch': numero_irmaos,
    'sibsp': numero_filhos
  }])

  st.dataframe(novo_passageiro)

  lr = carrega_modelo()

  predicao = lr.predict(novo_passageiro)

  st.write('Sobreviveu' if predicao[0]==1 else 'Não Sobreviveu')