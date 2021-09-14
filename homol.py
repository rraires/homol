import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import investpy as inv
#import time
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import cufflinks as cf
import datetime
from datetime import datetime
import math
import statsmodels.api
import statsmodels as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sazonalidade():
    st.header('Análise de Sazonalidade')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('Escolha o País')
        pais = st.radio('', ('Brasil', 'Estados Unidos'))
    with col2:
        st.write('Escolha entre Ações ou Indices')
        opcao = st.radio('', ('Ações', 'Indices'))

    st.session_state.lista_tickers = ['WEGE3', 'VALE3', 'PETR4', 'BBDC4', 'ITUB4']

    if pais == 'Brasil' and opcao == 'Ações':
        # lista = st.session_state.lista_tickers
        lista = inv.get_stocks_list(country='brazil')
    if pais == 'Brasil' and opcao == 'Indices':
        lista = inv.get_indices_list(country='brazil')
    if pais == 'Estados Unidos' and opcao == 'Ações':
        lista = inv.get_stocks_list(country='united states')
    if pais == 'Estados Unidos' and opcao == 'Indices':
        lista = inv.get_indices_list(country='united states')
    retornos = ''  # Inicia a variável retornos, pois há o teste dela lá em baixo para ver se está nula.
    with st.form(key='Analise_Sazonalidade'):
        ticker = st.selectbox('Selecione a Ação ou Indice desejado', lista)
        if st.form_submit_button(label='Analisar Sazonalidade'):
            try:
                data_inicial = '01/12/1999'
                data_final = datetime.today().strftime('%d/%m/%Y')

                if pais == 'Brasil' and opcao == 'Ações':
                    data_inicial = '1999-12-01'
                    data_final = datetime.today().strftime('%Y-%m-%d')
                    retornos = yf.download(ticker + '.SA', start= data_inicial, end=data_final, interval='1mo', progress=False)["Adj Close"].pct_change()
                    preco = yf.download(ticker + '.SA', start= data_inicial, end=data_final, progress=False)["Adj Close"]

                if pais == 'Brasil' and opcao == 'Indices':
                    retornos = \
                    inv.get_index_historical_data(ticker, country='brazil', from_date=data_inicial, to_date=data_final,
                                                  interval='Monthly')['Close'].pct_change()
                    preco = \
                    inv.get_index_historical_data(ticker, country='brazil', from_date=data_inicial, to_date=data_final,
                                                  interval='Daily')['Close']
                if pais == 'Estados Unidos' and opcao == 'Ações':
                    retornos = \
                        inv.get_stock_historical_data(ticker, country='united states', from_date=data_inicial,
                                                      to_date=data_final,
                                                      interval='Monthly')['Close'].pct_change()
                    preco = \
                        inv.get_stock_historical_data(ticker, country='united states', from_date=data_inicial,
                                                      to_date=data_final,
                                                      interval='Daily')['Close']
                if pais == 'Estados Unidos' and opcao == 'Indices':
                    retornos = \
                        inv.get_index_historical_data(ticker, country='united states', from_date=data_inicial,
                                                      to_date=data_final,
                                                      interval='Monthly')['Close'].pct_change()
                    preco = \
                        inv.get_index_historical_data(ticker, country='united states', from_date=data_inicial,
                                                      to_date=data_final,
                                                      interval='Daily')['Close']
                preco = preco.fillna(method='bfill')

            except:
                st.error('Algo errado com o ativo escolhido! Provavelmente seus dados históricos apresentaram algum problema. Escolha outro Ativo.')

    if len(retornos) != 0:
        mapa_retornos(ticker, retornos)
        grafico_sazonalidade(ticker, preco)

def mapa_retornos(ticker, retornos):
    with st.expander("Retornos Mensais", expanded=False):
        # Separar e agrupar os anos e meses
        retorno_mensal = retornos.groupby([retornos.index.year.rename('Year'), retornos.index.month.rename('Month')]).mean()
        # Criar e formatar a tabela pivot table
        tabela_retornos = pd.DataFrame(retorno_mensal)
        try:
            tabela_retornos = pd.pivot_table(tabela_retornos, values='Close', index='Year', columns='Month')
        except:
            tabela_retornos = pd.pivot_table(tabela_retornos, values='Adj Close', index='Year', columns='Month')

        tabela_retornos.columns = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        col1, col2, col3 = st.columns([0.1,1,0.1])
        with col2:
            # HeatMap Seaborn
            fig, ax = plt.subplots(figsize=(12, 9))
            cmap = sns.color_palette('RdYlGn', 50)
            sns.heatmap(tabela_retornos, cmap=cmap, annot=True, fmt='.2%', center=0, vmax=0.02, vmin=-0.02, cbar=False,
                        linewidths=1, xticklabels=True, yticklabels=True, ax=ax)
            ax.set_title(ticker, fontsize=18)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, verticalalignment='center', fontsize='12')
            ax.set_xticklabels(ax.get_xticklabels(), fontsize='12')
            ax.xaxis.tick_top()  # x axis on top
            plt.ylabel('')
            st.pyplot(fig)

            # Media das rentabilidades
            media = pd.DataFrame(tabela_retornos.mean())
            media.columns = ['Média']
            media = media.transpose()
            fig, ax = plt.subplots(figsize=(12, 0.5))
            sns.heatmap(media, cmap=cmap, annot=True, fmt='.2%', center=0, vmax=0.02, vmin=-0.02, cbar=False,
                        linewidths=1, xticklabels=True, yticklabels=True, ax=ax)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, verticalalignment='center', fontsize='11')
            st.pyplot(fig)

def grafico_sazonalidade(ticker, preco):
    with st.expander("Gráfico Sazonalidade", expanded=False):
        decomposicao = sm.tsa.seasonal.seasonal_decompose(preco, model='additive',
                                                          period=252)  # Decomposição da Série de preços
        sazonalidade = pd.DataFrame(decomposicao.seasonal)  # Cria um dataframe com a parte de sazonalidade
        # Montar a tabela pivot, separando os anos.
        df_pivot = pd.pivot_table(sazonalidade, values='seasonal',
                                  index=[sazonalidade.index.month.rename('Mês'),
                                         sazonalidade.index.day.rename('Dia')], columns=sazonalidade.index.year)
        df_pivot = df_pivot.fillna(method='bfill')
        df_pivot['Sazonalidade'] = df_pivot.mean(axis=1)  # Criar coluna com a média da sazonalidade dos anos/mês
        df_pivot.drop(columns=df_pivot.columns[:-1],
                      inplace=True)  # Apagar todos os anos e deixar apenas a coluna da Media
        df_pivot.reset_index(level=[0, 1],
                             inplace=True)  # Resetar o Multi-Indice isolando o dia e mês em colunas separadas
        df_pivot['Dia'] = df_pivot['Dia'].astype(str)  # Muda a coluna para String para poder concatenar depois
        df_pivot['Mês'] = df_pivot['Mês'].astype(str)  # Muda a coluna para String para poder concatenar depois
        df_pivot['Data'] = df_pivot['Dia'] + '/' + df_pivot['Mês']  # Cria uma coluna com o Dia e Mês Concatenado
        df_pivot.drop(columns=['Dia', 'Mês'], inplace=True)  # Apaga as colunas que não serão usadas
        df_pivot.set_index('Data', inplace=True)  # Seta a coluna data como index
        df_pivot.drop('29/2',
                      inplace=True)  # Apaga o dia 29/02, pois dá problema nos calculos de data por só existir em alguns anos.
        # Loop para converter a coluna Data de String para Date Format
        count = 0
        df_pivot['DateTime'] = ''
        for i in range(len(df_pivot)):
            df_pivot['DateTime'][count] = datetime.strptime(df_pivot.index[count], '%d/%m')
            count += 1

        lista_dias = pd.Series(range(len(df_pivot['DateTime'])))
        for i in range(len(df_pivot['DateTime'])):
            # lista_dias[i]=df_pivot['DateTime'][i].strftime("%Y-%m-%d")
            lista_dias[i] = df_pivot['DateTime'][i].strftime("%d/%b")

        df_pivot.set_index('DateTime', inplace=True)  # Seta a coluna Data como index

        #######PLOTLY EXPRESS#######

        # fig = px.line(df_pivot, title='Sazonalidade Anual - ' + ticker, width=1000, height=500)
        # if st.checkbox('Mostrar Data Atual'):
        #     fig.add_vline(hoje)
        # fig.update_layout(xaxis_tickformatstops=[dict(dtickrange=[None, 'M1'], value='%b'),
        #                                          dict(dtickrange=['M1', None], value="%d")],
        #                   xaxis=dict(tickvals=['1900-01-02', '1900-02-01', '1900-03-01', '1900-04-02', '1900-05-01',
        #                                        '1900-06-01', '1900-07-02', '1900-08-01', '1900-09-01', '1900-10-02',
        #                                        '1900-11-01', '1900-12-01']),
        #                   showlegend=False, hovermode="x unified"
        #                   )
        # fig.update_traces(hovertemplate="<b>%{x|%d/%b}</b>")
        # fig.update_yaxes(title_text='Sazonalidade')
        # fig.update_xaxes(fixedrange=True, title=None)
        # fig.update_yaxes(fixedrange=True)
        # st.plotly_chart(fig)

        #######PLOTLY#######
        # Create figure with secondary y-axis


        # df_preco_ano=pd.DataFrame(preco)
        # df_preco_ano = df_preco["2021-01-01":"2021-09-14"]
        # df_preco_ano.reset_index(inplace=True)
        # # df_preco_ano['Date'] = df_preco_ano['Date'].apply(lambda x: x.replace(year=1900))
        # df_preco_ano.set_index('Date', drop=True, inplace=True)
        # df_preco_ano.index = df_preco_ano.index + pd.DateOffset(year=1900)
        # df_preco_series = df_preco_ano.stack()

        # preco_a = yf.download('^BVSP', start='2021-01-01', end='2021-09-14')['Adj Close']
        # preco_a.index = preco_a.index + pd.DateOffset(year=1900)

        preco_ano = preco.loc["2021-01-01":"2021-09-14"]
        preco_ano.index = preco_ano.index + pd.DateOffset(year=1900)


        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces

        fig.add_trace(
            go.Scatter(x=df_pivot.index, y=df_pivot['Sazonalidade'], name="Sazonalidade"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=preco_ano.index, y=preco_ano.values, name="Preço"),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text='Sazonalidade Anual - ' + ticker
        )
        # # Set x-axis title
        # fig.update_xaxes(title_text="xaxis title")
        # # Set y-axes titles
        # fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
        # fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

        fig.update_layout(xaxis_tickformatstops=[dict(dtickrange=[None, 'M1'], value='%b'),
                                                 dict(dtickrange=['M1', None], value="%d")],
                          xaxis=dict(tickvals=['1900-01-02', '1900-02-01', '1900-03-01', '1900-04-02', '1900-05-01',
                                               '1900-06-01', '1900-07-02', '1900-08-01', '1900-09-01', '1900-10-02',
                                               '1900-11-01', '1900-12-01']),
                          showlegend=True, hovermode="x unified",
                          width=1000,
                          height=500
                          )
        fig.update_traces(hovertemplate="<b>%{x|%d/%b}</b>")
        fig.update_yaxes(title_text="<b>Sazonalidade</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Preço</b>", secondary_y=True)
        fig.update_xaxes(fixedrange=True, title=None)
        fig.update_yaxes(fixedrange=True)
        st.plotly_chart(fig)




sazonalidade()

