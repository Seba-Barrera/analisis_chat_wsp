
###############################################################################################################
# App de analisis de chat wsp
###############################################################################################################

# Posibles Mejoras:
# - filtro global de fechas que afecte todos los otros analisis 
# - Evolutivo de palabras mas usadas (total y filtrable por algun emisor)

#**************************************************************************************************************
# [A] Importar LIbrerias a Utilizar
#**************************************************************************************************************

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime

from sklearn import preprocessing

import re
from unicodedata import normalize



# import nltk # nueva linea agregada 
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud # https://www.edureka.co/community/64068/installing-wordcloud-using-unable-getting-following-error

from apyori import apriori

from sklearn.cluster import KMeans

import streamlit as st


#**************************************************************************************************************
# [B] Crear funciones utiles para posterior uso
#**************************************************************************************************************

#**************************************************************************************************************
# B.2 Funcion para aplicar en reglas de asociacion 

def inspect(output):
    lhs        = [tuple(result[2][0][0])[0] for result in output]
    rhs        = [tuple(result[2][0][1])[0] for result in output]
    support    = [result[1] for result in output]
    confidence = [result[2][0][2] for result in output]
    lift       = [result[2][0][3] for result in output]
    return list(zip(lhs, rhs, support, confidence, lift))


#**************************************************************************************************************
# [Z] Comenzar a diseñar App
#**************************************************************************************************************

def main():
    
    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide")
    
 #=============================================================================================================
 # [01] Subir archivo, leerlo y hacer calculos 
 #=============================================================================================================   
    
    # titulo inicial 
    st.markdown('## Analisis de Conversaciones por whatsapp')
    
    # autoria 
    st.sidebar.markdown('**Autor: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')
    st.sidebar.markdown('Version SB_V20221225')
    
    # tutoriales de como descargar chats de wsp  
    st.sidebar.markdown('*[Como descargar Chat en Android](https://www.mundodeportivo.com/uncomo/tecnologia/articulo/como-descargar-las-conversaciones-de-whatsapp-en-android-35091.html)*')
    st.sidebar.markdown('*[Como descargar Chat en IOS](https://www.xataka.com/basics/como-descargar-exportar-conversaciones-whatsapp-para-guardarlos-leerlos-cualquier-lado)*')
        
    # subir archivo 
    Archivo = st.sidebar.file_uploader('Subir Conversacion.txt',type=['txt'])
    
    st.sidebar.markdown('*el archivo de conversacion ingresado no queda guardado luego de uso*')
        
    if Archivo is not None:
        
 #-------------------------------------------------------------------------------------------------------------
 # [01.1] Crear funcion 
 #-------------------------------------------------------------------------------------------------------------   

        
        # Definir funcion para tratamiendo de archivo y generacion de DFs
        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def tratamiento_txt_df(archivo):       
            
            #__________________________________________________________________________________________________
            # Lectura de archivo txt 
            
            df = pd.read_table(archivo, delimiter="$$$$$$xxt$$$yzz$$$$//",header=None)
            df.columns=['Texto']

            # quitar espacios 
            df['Texto'] = df['Texto'].apply(lambda x: x.strip()) 

            # determinar separador de fecha (guion o slash)
            df['sep_fecha']=df['Texto'].apply(lambda x: 
                '-' if x.find('-')>=0 and x.find('/')>=0 and x.find('-')<x.find('/') else 
                '-' if x.find('-')>=0 and x.find('/')<0 else
                '/' 
                )

            # encontrar primer caracter de fecha
            df['Hito_fecha1']=df.apply(lambda x: x.Texto.find(x.sep_fecha),axis=1)

            # encontrar segundo caracter de fecha
            df['Hito_fecha2']=df.apply(lambda x: x.Texto.find(x.sep_fecha,x.Hito_fecha1+1),axis=1)

            # encontrar dia             
            df['texto_dia'] = df.apply(lambda x: 
                x.Texto[0] if x.Hito_fecha1==1 else
                x.Texto[(x.Hito_fecha1-2):x.Hito_fecha1] if x.Hito_fecha1>=2 
                else 'NA' , axis=1 
                )

            # depurar texto dia para corroborar que sea numerico 
            df['texto_dia'] = df.apply(lambda x: 
                x.texto_dia if x.texto_dia.isnumeric()
                else 'NA' , axis=1 
                )
            
            # corregir casos donde solo tenga un digito
            df['texto_dia'] = df.apply(lambda x: 
                '0'+x.texto_dia if len(x.texto_dia)==1
                else x.texto_dia , axis=1 
                )

            # encontrar mes 
            df['texto_mes'] = df.apply(lambda x: 
                x.Texto[(x.Hito_fecha1+1):x.Hito_fecha2] if x.Hito_fecha1>=1 
                else 'NA' , axis=1 
                )

            # depurar texto mes para corroborar que sea numerico 
            df['texto_mes'] = df.apply(lambda x: 
                x.texto_mes if x.texto_mes.isnumeric()
                else 'NA' , axis=1 
                )
            
            # corregir casos donde solo tenga un digito
            df['texto_mes'] = df.apply(lambda x: 
                '0'+x.texto_mes if len(x.texto_mes)==1
                else x.texto_mes , axis=1 
                )

            # encontrar año 
            df['texto_ano'] = df.apply(lambda x: 
                x.Texto[(x.Hito_fecha2+1):(x.Hito_fecha2+5)] if x.Hito_fecha1>=1 
                else 'NA' , axis=1 
                )

            # depurar año en caso que solo sean 2 digitos 
            df['texto_ano2'] = df.apply(lambda x: 
                x.texto_ano if x.texto_ano.isnumeric() and x.Hito_fecha1>=1  else 
                '20'+x.texto_ano[0:2] if x.Hito_fecha1>=1 
                else 'NA' , axis=1 
                )

            # depurar texto ano para corroborar que sea numerico 
            df['texto_ano2'] = df.apply(lambda x: 
                x.texto_ano2 if x.texto_ano2.isnumeric()
                else 'NA' , axis=1 
                )

            # encontrar primer caracter de fecha
            df['Hito_hora']=df['Texto'].apply(lambda x: x.find(':'))

            # encontrar hora
            df['texto_hora'] = df.apply(lambda x: 
                x.Texto[(x.Hito_hora-2):x.Hito_hora] if x.Hito_hora>=2 and x.Texto[(x.Hito_hora-2):x.Hito_hora].isnumeric() else 
                x.Texto[(x.Hito_hora-1):x.Hito_hora] if x.Hito_hora>=2 
                else 'NA' , axis=1 
                )

            # depurar texto hora para corroborar que sea numerico 
            df['texto_hora'] = df.apply(lambda x: 
                x.texto_hora if x.texto_hora.isnumeric()
                else 'NA' , axis=1 
                )

            # encontrar minuto           
            df['texto_minuto'] = df.apply(lambda x: 
                x.Texto[(x.Hito_hora+1):(x.Hito_hora+3)] if x.Hito_hora>=2 and x.Texto[(x.Hito_hora+1):(x.Hito_hora+3)].isnumeric() else
                x.Texto[(x.Hito_hora+1):(x.Hito_hora+2)] if x.Hito_hora>=2
                else 'NA' , axis=1 
                )

            # depurar texto minuto para corroborar que sea numerico 
            df['texto_minuto'] = df.apply(lambda x: 
                x.texto_minuto if x.texto_minuto.isnumeric()
                else 'NA' , axis=1 
                )

            # rescatar hito de resto de mensaje ubicando espacio desde los : hacia adelante
            df['Hito_resto']=df.apply(lambda x: x.Texto.find(' ',x.Hito_hora+1), axis=1 )

            # rescatar resto de mensaje 
            df['texto_resto']=df.apply(lambda x: x.Texto[x.Hito_resto:666].strip(), axis=1 )

            # rescatar hito emisor
            df['Hito_emisor']=df.apply(lambda x: x.texto_resto.find(':'), axis=1 )

            # rescatar emisor 
            df['Emisor']=df.apply(lambda x: 
                x.texto_resto[0:x.Hito_emisor] if x.Hito_emisor>=0 else
                'NA'
                , axis=1)

            # correccion de Emisor para casos con separacion de guion
            df['Emisor']=df.apply(lambda x: 
                x.Emisor[2:666] if x.Emisor[0:2]=='- ' else 
                x.Emisor
                , axis=1)

            # rescatar mensaje 
            df['Mensaje']=df.apply(lambda x: 
                x.texto_resto[(x.Hito_emisor+1):666] if x.Hito_emisor>=0 else
                'NA'
                , axis=1)

            # Construir campos para uso posterior 
            df['Fecha']=df.apply(lambda x: x.texto_dia+'/'+x.texto_mes+'/'+x.texto_ano2, axis=1)
            df['Hora']=df.apply(lambda x: x.texto_hora+':'+x.texto_minuto, axis=1)

            #__________________________________________________________________________________________________
            # Quedarse solo con campos relevantes y comenzar a crear otros campos 

            # Quedarse solamente con registros correctos
            df2 = df.loc[
                (df['texto_dia']!='NA') & (df['Hito_emisor']>=0),
                ['Fecha','Hora','Emisor','Mensaje']
                ]
            
            
                # llevar fecha de string a fecha    
            df2['Fecha']=df2['Fecha'].apply(lambda x: 
                10000*int(x[6:10])+
                100*int(x[3:5])+
                int(x[0:2])
                )

            df2['Fecha']=df2['Fecha'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%Y/%m/%d') )
            
            # crear campo de nro de palabras 
            df2['Nro_Palabras']=df2['Mensaje'].apply(lambda x: len(x.split()) )
            
            # crear campo de hora 
            df2['Hora2']=df2['Hora'].apply(lambda x: int(x[0:2]) )
            
            # crear campo de dia de la semana 
            df2['dia_sem']=df2['Fecha'].apply(lambda x: 
                '1.Lunes' if pd.to_datetime(x).day_name()=='Monday' else
                '2.Martes' if pd.to_datetime(x).day_name()=='Tuesday' else
                '3.Miercoles' if pd.to_datetime(x).day_name()=='Wednesday' else
                '4.Jueves' if pd.to_datetime(x).day_name()=='Thursday' else
                '5.Viernes' if pd.to_datetime(x).day_name()=='Friday' else
                '6.Sabado' if pd.to_datetime(x).day_name()=='Saturday' else
                '7.Domingo' if pd.to_datetime(x).day_name()=='Sunday' else
                ''  
                )
            
            #__________________________________________________________________________________________________
            # Generar DF agrupado  
            
            df2_AGG_Resumen=df2.groupby(['Emisor']).agg(
                N_Mensajes = pd.NamedAgg(column = 'Emisor', aggfunc = len),
                Min_fecha = pd.NamedAgg(column = 'Fecha', aggfunc = min),
                Max_fecha = pd.NamedAgg(column = 'Fecha', aggfunc = max),
                N_Dias =  pd.NamedAgg(column = 'Fecha', aggfunc = lambda x: len(x.unique())),
                Hora_Prom = pd.NamedAgg(column = 'Hora2', aggfunc = np.mean),
                N_Palabras = pd.NamedAgg(column = 'Nro_Palabras', aggfunc = sum)                
                )
            df2_AGG_Resumen.reset_index(level=df2_AGG_Resumen.index.names, inplace=True) # pasar indices a columnas

            # ordenar 
            df2_AGG_Resumen=df2_AGG_Resumen.sort_values(by=['N_Mensajes'], ascending=False)

            # resetear indices 
            df2_AGG_Resumen=df2_AGG_Resumen.reset_index(drop=True)

            # obtener algunas metricas 
            df2_AGG_Resumen['Mjs_x_dia']=df2_AGG_Resumen.apply(lambda x: round(x['N_Mensajes']/x['N_Dias'],2), axis=1 )

            df2_AGG_Resumen['Palabras_x_Mjs']=df2_AGG_Resumen.apply(lambda x: round(x['N_Palabras']/x['N_Mensajes'],2), axis=1 )

            df2_AGG_Resumen['Peso_Mensajes']=round((df2_AGG_Resumen['N_Mensajes'] / df2_AGG_Resumen['N_Mensajes'].sum()) * 100,2) # https://www.geeksforgeeks.org/how-to-calculate-the-percentage-of-a-column-in-pandas/

            df2_AGG_Resumen['Hora_Prom']=df2_AGG_Resumen['Hora_Prom'].apply(lambda x: round(x,2))

            df2_AGG_Resumen['Rango_fecha']=df2_AGG_Resumen.apply(
                lambda x: 
                    (datetime.strptime(x['Max_fecha'],"%Y/%m/%d")-datetime.strptime(x['Min_fecha'],"%Y/%m/%d")).days
                    , axis=1 
                    )

            df2_AGG_Resumen['Peso_Dias']=df2_AGG_Resumen.apply(lambda x: round(100*x['N_Dias']/(1+x['Rango_fecha']),2), axis=1)

            # para acumulado de porcentaje de mensajes 
            df2_AGG_Resumen['Peso_Mensajes2']=df2_AGG_Resumen['Peso_Mensajes'].cumsum() # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cumsum.html

            # agregar correlativo de top de quienes envian mas mensajes 
            df2_AGG_Resumen.insert(0, 'Top', range(1, 1 + len(df2_AGG_Resumen)))
    
    
            # Pegar en gran tabla el top de usuario (para efectos de filtro posteriores)
            df2=pd.merge(
                df2, 
                df2_AGG_Resumen[['Emisor','Top']], 
                on='Emisor', 
                how='left'
                )
    
            #__________________________________________________________________________________________________
            # Generar DF separado por palabras     
            
            # separar df palabra por palabra 
            df2_palabras = df2[['Emisor','Mensaje']].assign(palabra=df2['Mensaje'].str.split()).explode('palabra')

            df2_palabras = df2_palabras.drop('Mensaje', axis = 1) # eliminar campo de mensaje 
            
            # limpiar palabra 
            df2_palabras['palabra2']=df2_palabras['palabra'].apply(lambda x:
                normalize( 
                        'NFC', 
                        re.sub(
                            r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", 
                            r"\1", 
                            normalize( 
                                        "NFD",
                                        re.sub(r'[^\w\s]', '', x.lower().strip())
                                        ), 
                            0, 
                            re.I
                            )
                        )
                )


            # Crear nuevo DF apilando el total 
            df2_palabras = pd.concat(
                [
                    pd.DataFrame({
                        'Emisor': 'TOTAL',
                        'palabra':df2_palabras['palabra2'] 
                        }),
                    pd.DataFrame({
                        'Emisor': df2_palabras['Emisor'] ,
                        'palabra':df2_palabras['palabra2'] 
                        })        
                    ], 
                axis=0, 
                ignore_index=True
                )

        
            # nltk.download('stopwords') # necesario para tener listado de stopwords
            # https://discuss.streamlit.io/t/nltk-dependencies/5749/5
            # stop_words = stopwords.words('spanish')             
            stop_words = pd.read_csv('stop_words_spanish.txt', sep=" ", header=None,encoding='latin-1')
            stop_words = stop_words.iloc[:,0].tolist()
            
            # crear marca de stopword
            df2_palabras['StopWord']=df2_palabras['palabra'].apply(
                lambda x: 1 if 
                x in stop_words
                or len(x)<4 
                or len(re.sub('j|a', '',x))==0 # quitar todos los jajajajajaja
                else 0
                )
            
            # quedarse sin palabras de mas 
            df2_palabras=df2_palabras.loc[df2_palabras['StopWord']==0,['Emisor','palabra']]
            
            # crear bista agrupada 
            df2_palabras_Agg=df2_palabras.groupby(['Emisor','palabra']).agg(
                Conteo = pd.NamedAgg(column = 'palabra', aggfunc = len)
                )
            df2_palabras_Agg.reset_index(level=df2_palabras_Agg.index.names, inplace=True) # pasar indices a columnas
        

            #__________________________________________________________________________________________________
            # Generar DF de interacciones 

            # crear tabla en blanco de interacciones a poblar posteriormente 
            df2_interaccion = pd.DataFrame(
                data=[ 
                    (i, j)
                    for i in df2['Emisor'].unique()
                    for j in df2['Emisor'].unique() 
                    ], 
                columns=['quien_envia','quien_responde']
                )

            df2_interaccion['N']=0

            # crear tabla en blanco de monologos a poblar posteriormente 
            df2_monologo = pd.DataFrame(
                data=[i for i in df2['Emisor'].unique()], 
                columns=['Emisor']
                )
            
            df2_monologo['N']=0


            # recorrer df rescatando casos de interacciones 
            for i in range(1,df2.shape[0]):
                
                quien_envia = df2.iloc[i-1,2]
                quien_responde = df2.iloc[i,2]
                
                fecha_envia = df2.iloc[i-1,0]
                fecha_responde = df2.iloc[i,0]
                
                if quien_envia!=quien_responde and fecha_envia==fecha_responde:
                    
                    df2_interaccion.loc[
                        (df2_interaccion['quien_envia']==quien_envia) & 
                        (df2_interaccion['quien_responde']==quien_responde),
                        'N'
                        ]+=1
                
                if quien_envia==quien_responde and fecha_envia==fecha_responde:
                    
                    df2_monologo.loc[
                        (df2_monologo['Emisor']==quien_envia),
                        'N'
                        ]+=1
                
                
            # ordernar base 
            df2_interaccion = df2_interaccion.sort_values(by=['N'], ascending=False)

            # ordenar base de monologos 
            df2_monologo = df2_monologo.sort_values(by=['N'], ascending=False)
            
            # Pegar variable de N_Msj_Seguidos en DF agregado 
            df2_AGG_Resumen=pd.merge(
                df2_AGG_Resumen, 
                df2_monologo, 
                on='Emisor', 
                how='left'
                )
            
            # renombrar variable 
            df2_AGG_Resumen.rename(columns={'N':'N_Msj_Seguidos'}, inplace=True)

            return df2, df2_AGG_Resumen,df2_palabras,df2_palabras_Agg,df2_interaccion,df2_monologo

 #-------------------------------------------------------------------------------------------------------------
 # [01.2] Invocar funcion para generar DFs 
 #-------------------------------------------------------------------------------------------------------------   

        df2, df2_AGG_Resumen,df2_palabras,df2_palabras_Agg,df2_interaccion,df2_monologo = tratamiento_txt_df(Archivo)


 #=============================================================================================================
 # [02] Mostrar cifras de los mensajes post titulo inicial (no sujeto a subida de archivo)
 #=============================================================================================================   
 
        st.markdown(
            'Se consideraron '+
            str(len(df2['Emisor']))+
            ' mensajes (de '+
            str(len(df2['Emisor'].unique()))+
            ' emisores distintos) entre el '+
            str(min(df2['Fecha']))+
            ' hasta el '+
            str(max(df2['Fecha']))
            )

 #=============================================================================================================
 # [03] Mostrar Detalle de toda la conversacion 
 #=============================================================================================================   
        
        st.markdown('### 1. Detalle de conversacion')
        
        texto_filtro = st.text_input('Ingresa texto para buscar')
                
        # https://docs.streamlit.io/library/api-reference/data/st.dataframe
        st.dataframe(
            df2.loc[
                (df2['Mensaje'].str.lower().str.contains(str.lower(texto_filtro))) | 
                (df2['Emisor'].str.lower().str.contains(str.lower(texto_filtro))) | 
                (df2['Fecha'].str.lower().str.contains(str.lower(texto_filtro))),
                ['Fecha','Hora','Emisor','Mensaje']
                ]
            )
        
        # dejar disponible para descagar df
        st.download_button(
            "Presiona para descargar tabla",
            df2.loc[
                (df2['Mensaje'].str.lower().str.contains(str.lower(texto_filtro))) | 
                (df2['Emisor'].str.lower().str.contains(str.lower(texto_filtro))) | 
                (df2['Fecha'].str.lower().str.contains(str.lower(texto_filtro))),
                ['Fecha','Hora','Emisor','Mensaje']
                ].to_csv().encode('utf-8'),
            "Detalle.csv",
            "text/csv",
            key='download-csv'
            )
           
    
 #=============================================================================================================
 # [04] Cuadro con estadisticas de cada Emisor
 #=============================================================================================================   
    
        st.markdown('### 2. Resumen de indicadores por Emisor')
        
        texto_filtro2 = st.text_input('Ingresa Emisor para buscar')
                
        # https://docs.streamlit.io/library/api-reference/data/st.dataframe
        st.dataframe(
            df2_AGG_Resumen.loc[
                (df2_AGG_Resumen['Emisor'].str.lower().str.contains(str.lower(texto_filtro2))),
                ['Emisor','N_Mensajes','Min_fecha','Max_fecha','N_Dias','N_Palabras','Mjs_x_dia','Palabras_x_Mjs','Peso_Mensajes','Hora_Prom']                
                ]
            )
        
        # dejar disponible para descagar df
        st.download_button(
            "Presiona para descargar tabla",
            df2_AGG_Resumen.loc[
                (df2_AGG_Resumen['Emisor'].str.lower().str.contains(str.lower(texto_filtro2))),
                ['Emisor','N_Mensajes','Min_fecha','Max_fecha','N_Dias','N_Palabras','Mjs_x_dia','Palabras_x_Mjs','Peso_Mensajes','Hora_Prom']                
                ].to_csv().encode('utf-8'),
            "Resumen.csv",
            "text/csv",
            key='download-csv'
            )


 #=============================================================================================================
 # [05] Grafico de top Emisores (en cuando a cantidad de mensajes) y cuanto acumulan del total
 #=============================================================================================================   

 #-------------------------------------------------------------------------------------------------------------
 # [05.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------


        # Titulo del capitulo 
        st.markdown('### 3. Grafico de peso de los mensajes por Emisor')
    
        # Generar Slider 
        slider5 = st.slider(
            label = 'Seleccionar Top Emisores', 
            min_value=1, 
            max_value=len(df2['Emisor'].unique()), 
            value=15
            )
    
 #-------------------------------------------------------------------------------------------------------------
 # [05.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------

        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_5(top_emisores):         
            
            # Crear Tabla y Grafico respectivamente 
            df2_AGG_Resumen_Graf5 = pd.concat(
                [
                    pd.DataFrame({
                        'Top':df2_AGG_Resumen['Top'],
                        'Emisor':df2_AGG_Resumen['Emisor'],
                        'Peso':df2_AGG_Resumen['Peso_Mensajes'],
                        'Acumulado': 'No'
                        }),
                    pd.DataFrame({
                        'Top':df2_AGG_Resumen['Top'],
                        'Emisor':df2_AGG_Resumen['Emisor'],
                        'Peso':df2_AGG_Resumen['Peso_Mensajes2'],
                        'Acumulado': 'Si'
                        })      
                    ], 
                axis=0, 
                ignore_index=True
                )

            fig5 = px.bar(
                df2_AGG_Resumen_Graf5[df2_AGG_Resumen_Graf5['Top']<=top_emisores], 
                x="Emisor", 
                y="Peso",
                color='Acumulado', 
                barmode='group',
                title="Top Emisores de Mensajes"
                )

            fig5.update_layout(
                legend=dict(orientation="h",yanchor="bottom",y=1.02, xanchor="right",x=1),
                width=900,
                height=500
                )
            
            return fig5
        
 #-------------------------------------------------------------------------------------------------------------
 # [05.3] Generar entregable y mostrar 
 #-------------------------------------------------------------------------------------------------------------

        fig5 = entregable_5(slider5)

        st.plotly_chart(fig5, use_container_width=True)

 #=============================================================================================================
 # [06] Distribucion segun fecha 
 #=============================================================================================================   
 
 #-------------------------------------------------------------------------------------------------------------
 # [06.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------
    
        # Titulo del capitulo 
        st.markdown('### 4. Grafico de Distribucion segun Fecha, Hora o Dia')

        # Ingresar widgets         
        col6_1, col6_2, col6_3 = st.columns((1,1,1))
        
        radio6 = col6_1.radio(
            'Metrica a revisar',
            ['Fecha', 'Dia', 'Hora']
            )
        
        slider6 = col6_2.slider(
            label = 'Seleccionar Top Emisores', 
            min_value=1, 
            max_value=len(df2['Emisor'].unique()), 
            value=10
            )
        
        number_input6 = col6_3.number_input(label = 'Alto del grafico',value = 650)
        
 #-------------------------------------------------------------------------------------------------------------
 # [06.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------
   
        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_6(metrica,top_emisores,alto_grafico):  
                    
            # Crear variable a considerar 
            if metrica == 'Hora':
                Var_medir = 'Hora2'
            elif metrica == 'Dia':
                Var_medir = 'dia_sem'
            else:
                Var_medir = 'Fecha'
                        
            # Generar Tabla y Grafico         
            df2_Graf6 = pd.concat(
                [
                    pd.DataFrame({
                        'Top': 0,
                        'Emisor': 'TOTAL',
                        'Variable':df2[Var_medir] 
                        }),
                    pd.DataFrame({
                        'Top': df2.loc[df2['Top']<=top_emisores,'Top'],
                        'Emisor': df2.loc[df2['Top']<=top_emisores,'Emisor'],
                        'Variable':df2.loc[df2['Top']<=top_emisores,Var_medir] 
                        })        
                    ], 
                axis=0, 
                ignore_index=True
                )

            df2_Graf6 = df2_Graf6.sort_values(['Variable','Top']).reset_index(drop=True)

            # Crear objeto grafico (# https://plotly.com/python/violin/#ridgeline-plot)
            fig6 = go.Figure()

            for emi in df2_Graf6['Emisor'].unique():
                
                fig6.add_trace(go.Violin(
                    x=df2_Graf6.loc[df2_Graf6['Emisor']==emi,'Variable'].values.tolist(),
                    name=emi
                    ))

    
            fig6.update_traces(orientation='h', side='negative', width=3, points=False)
            
            # Setear detalles del grafico 
            if metrica == 'Hora':
                
                fig6.update_layout(
                    title = 'Distribucion de mensajes por hora top Emisores',
                    xaxis_showgrid=False, xaxis_zeroline=False,showlegend=False,
                    width=700,height=alto_grafico,
                    xaxis = dict(tickmode = 'linear',tick0 = 8,dtick = 2,range=[6, 25]),
                    xaxis_title='Hora'
                    )
                
            elif metrica == 'Dia':
                
                fig6.update_layout(
                    title = 'Distribucion de mensajes por dia top Emisores',
                    xaxis_showgrid=False, xaxis_zeroline=False,showlegend=False,
                    width=700,height=alto_grafico,
                    xaxis = dict(tickmode = 'linear',tick0 = 8,dtick = 1),
                    xaxis_title='Dia de la semana'
                    )
                
            else:
                
                fig6.update_layout(
                    title = 'Distribucion de mensajes por Fecha top Emisores',
                    xaxis_showgrid=False, xaxis_zeroline=False,showlegend=False,
                    width=700,height=alto_grafico,
                    xaxis = dict(tickmode = 'linear',tick0 = 8,dtick = 5,range=[min(df2['Fecha']), max(df2['Fecha'])]),
                    xaxis_title='Fecha'
                    )            
            
            # reversar eje 
            fig6['layout']['yaxis']['autorange'] = "reversed"
            
            return fig6
        
        
        
        
 #-------------------------------------------------------------------------------------------------------------
 # [06.3] Generar entregable y mostrar 
 #-------------------------------------------------------------------------------------------------------------

        fig6 = entregable_6(radio6,slider6,number_input6)
        
        # mostrar grafico         
        st.plotly_chart(fig6, use_container_width=True)
   
 #=============================================================================================================
 # [07] Grafico de dispersion 2D y 3D segun caracteristicas de cada Emisor
 #=============================================================================================================   
 
 #-------------------------------------------------------------------------------------------------------------
 # [07.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------

        # Titulo del capitulo 
        st.markdown('### 5. Grafico de Dispersion de Emisores segun metricas')
        
        # generar columnas 
        col7_1, col7_2, col7_3,col7_4,col7_5,col7_6 = st.columns((1,2,2,2,2,2)) 
        
        
        radio7 = col7_1.radio(
            'Tipo Grafico',
            ['3D', '2D']
            )
        
        # eje x
        selectbox7_ejex=col7_2.selectbox(
            'Eje X',
            df2_AGG_Resumen.select_dtypes([np.number]).columns,
            index = 1
            )
        
        # eje y
        selectbox7_ejey=col7_3.selectbox(
            'Eje Y',
            df2_AGG_Resumen.select_dtypes([np.number]).columns,
            index = 7
            )
        
        # eje z
        selectbox7_ejez=col7_4.selectbox(
            'Eje Z',
            df2_AGG_Resumen.select_dtypes([np.number]).columns,
            index = 5
            )
                    
        # color 
        selectbox7_color=col7_5.selectbox(
            'Color',
            df2_AGG_Resumen.select_dtypes([np.number]).columns,
            index = 2
            )
        
        # size 
        selectbox7_size=col7_6.selectbox(
            'Tamaño',
            df2_AGG_Resumen.select_dtypes([np.number]).columns,
            index = 4
            )
        
 #-------------------------------------------------------------------------------------------------------------
 # [07.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------
                        
        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_7(tipo_graf,p_ejex,p_ejey,p_ejez,p_color,p_size):
            
            if tipo_graf=='2D':
                
                fig7 = px.scatter(
                    df2_AGG_Resumen, 
                    x=p_ejex, 
                    y=p_ejey, 
                    color=p_color,
                    size=p_size, 
                    hover_data=['Emisor'] 
                    )                      
                
            else:              
                    
                fig7 = px.scatter_3d(
                    df2_AGG_Resumen, 
                    x=p_ejex, 
                    y=p_ejey, 
                    z=p_ejez,
                    color=p_color, 
                    size=p_size, 
                    hover_data=['Emisor'],
                    opacity=0.7
                    )            
            
            fig7.update_layout(
                width=650,
                height=650
                )
            
            return fig7           
        
 #-------------------------------------------------------------------------------------------------------------
 # [07.3] Generar entregable y mostrar 
 #-------------------------------------------------------------------------------------------------------------


        fig7 = entregable_7(radio7,selectbox7_ejex,selectbox7_ejey,selectbox7_ejez,selectbox7_color,selectbox7_size)


        # mostrar grafico         
        st.plotly_chart(fig7, use_container_width=True)

 #=============================================================================================================
 # [08] Grafico de radar comparando emisores que se vayan ingresando
 #=============================================================================================================   

 #-------------------------------------------------------------------------------------------------------------
 # [08.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------

 
        # Titulo del capitulo 
        st.markdown('### 6. Grafico de Radar segun emisor')
        
        # generar columnas 
        col6_1, col6_2 = st.columns((1, 1))            
        
        # Ingresar widget de seleccion emisores 
        multiselect8 = col6_1.multiselect(
            'Seleccionar Emisores',
            df2_AGG_Resumen['Emisor'].unique(),
            df2_AGG_Resumen['Emisor'].unique()[0:3],
            key = 1
            )
        
        # Ingresar widget de seleccion variables  
        multiselect8b = col6_2.multiselect(
            'Seleccionar Variables',
            df2_AGG_Resumen.select_dtypes([np.number]).columns,
            ['N_Mensajes','N_Dias','N_Palabras','Mjs_x_dia','Palabras_x_Mjs'],
            key = 1
            )
        
 #-------------------------------------------------------------------------------------------------------------
 # [08.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------
         
        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_8(emisores,variables):            
            
            # normalizar tabla 
            df2_AGG_Resumen_norm = pd.DataFrame(
                preprocessing.MinMaxScaler().fit_transform(
                    df2_AGG_Resumen[variables].values
                    )
                )

            # pegar dato de emisor 
            # https://www.datasciencemadesimple.com/append-concatenate-columns-python-pandas-column-bind/
            df2_AGG_Resumen_norm=pd.concat([df2_AGG_Resumen['Emisor'], df2_AGG_Resumen_norm], axis=1, ignore_index=True)

            # cambiar nombre a columnas 
            df2_AGG_Resumen_norm.columns = ['Emisor']+list(variables)

            # definir lista de sujetos a mostrar en radar 
            lista_emisor=emisores
            atributos=variables

            # crear objeto grafico (# https://plotly.com/python/radar-chart/)
            fig8 = go.Figure()

            for emi in lista_emisor:

                fig8.add_trace(go.Scatterpolar(
                    r=df2_AGG_Resumen_norm.loc[df2_AGG_Resumen_norm['Emisor']==emi,atributos].values.tolist()[0],
                    theta=atributos,
                    fill='toself',
                    name=emi
                ))


            # ultimas correcciones antes de mostrar 
            fig8.update_layout(
                polar=dict(radialaxis=dict(visible=False,range=[0, 1])),showlegend=True
                )
            
            return fig8

 #-------------------------------------------------------------------------------------------------------------
 # [08.3] Generar entregable y mostrar 
 #-------------------------------------------------------------------------------------------------------------

        fig8=entregable_8(multiselect8,multiselect8b)

        # mostrar grafico         
        st.plotly_chart(fig8, use_container_width=True)
        
 #=============================================================================================================
 # [09] Palabras mas frecuentes segun emisor ingresado
 #=============================================================================================================   

 #-------------------------------------------------------------------------------------------------------------
 # [09.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------
  

        # Titulo del capitulo 
        st.markdown('### 7. Top Palabras por Emisor')
        
        # Ingresar widget 
        col9_1, col9_2 = st.columns((3, 1))
        
        text_input9 = col9_1.text_input(
            'Ingresa palabras separadas por coma para excluir',
            value='omitido,multimedia',
            key=9)
        
        slider9 = col9_2.slider(
            label = 'Seleccionar Top palabras', 
            min_value=5, 
            max_value=20, 
            value=10
            )
        
        multiselect9 = st.multiselect(
            'Seleccionar Emisores',
            ['TOTAL']+list(df2_AGG_Resumen['Emisor'].unique()),
            ['TOTAL']+list(df2_AGG_Resumen['Emisor'].unique())[0:3],
            key = 2
            )

 #-------------------------------------------------------------------------------------------------------------
 # [09.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------

        # definir palabras a excluir 
        palabras_excluir = text_input9.split(',')
        
        # filtrar palabras a usar 
        df2_palabras_Agg1 = df2_palabras_Agg[~df2_palabras_Agg['palabra'].isin(palabras_excluir)]
        
        
        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_9(top_palabras,emisores):  
          
            fig9 = make_subplots(
                rows=1, 
                cols=len(emisores),
                horizontal_spacing = 0.1
                )

            for i in range(0,len(multiselect9)):

                df2_palabras_Agg2=df2_palabras_Agg1.loc[
                    df2_palabras_Agg1['Emisor']==emisores[i],
                    ['palabra','Conteo']   
                    ]

                df2_palabras_Agg2 = df2_palabras_Agg2.sort_values(by=['Conteo'], ascending=True)
                
                
                fig9.append_trace(
                    go.Bar(
                        x=df2_palabras_Agg2.tail(top_palabras)['Conteo'], 
                        y=df2_palabras_Agg2.tail(top_palabras)['palabra'],
                        name = emisores[i],
                        orientation='h'
                        ),
                    row=1, 
                    col=i+1
                    )
                

            fig9.update_layout(
                legend=dict(orientation="h",yanchor="bottom",y=1.02, xanchor="right",x=1),
                width=1300,
                height=500
                )
            
            fig9.for_each_xaxis(lambda x: x.update(showgrid=False))
            fig9.for_each_yaxis(lambda x: x.update(showgrid=False))
            
            return fig9

 #_______________________________________________________________
 # Generar entregables
 
        # Titulo del capitulo 
        st.markdown('#### 7.1 Grafico de Barras')
        
        # generar entregable 
        fig9 = entregable_9(slider9,multiselect9)
        
        # mostrar grafico         
        st.plotly_chart(fig9, use_container_width=True)
        

 #-------------------------------------------------------------------------------------------------------------
 # [09.3] Arrojar Vistas de Nube de Palabras 
 #-------------------------------------------------------------------------------------------------------------   

        # PENDIENTE COMO HACER OBJETOS PLT ENTREGABLES
        
        # para quitar advertencia de que ya no se usa asi (sin argumento) el comando st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # https://discuss.streamlit.io/t/how-to-add-wordcloud-graph-in-streamlit/818/2
        # Titulo del capitulo 
        st.markdown('#### 7.2 Nube de Palabras')

        
        plt.figure()
        plt.subplots(1,4,figsize=(25,25))

        for i in range(len(multiselect9)):
            
            df2_palabras_Agg2=df2_palabras_Agg1.loc[
            df2_palabras_Agg1['Emisor']==multiselect9[i],
            ['palabra','Conteo']   
            ]
            
            plt.subplot(1,len(multiselect9),i+1).set_title(multiselect9[i])
            plt.plot()
            
            wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='salmon', 
            colormap='Pastel1',
            max_words=50
            ).generate_from_frequencies(
                df2_palabras_Agg2.set_index('palabra').to_dict()['Conteo']
                )
            


            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')


        plt.show()

        st.pyplot()

 #=============================================================================================================
 # [10] Matriz de relaciones (segun quien responde a quien los mensajes)
 #=============================================================================================================   

 #-------------------------------------------------------------------------------------------------------------
 # [10.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------


        # titulo del capitulo 
        st.markdown('### 8. Tabla de interacciones (mensaje - respuesta)')
    
        # Generar Slider 
        slider8 = st.slider(
            label = 'Seleccionar Top Interacciones', 
            min_value=1, 
            max_value=55, 
            value=10
            )


 #-------------------------------------------------------------------------------------------------------------
 # [10.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------

        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_10(Top_Interacciones):   

            # pivotear top de la tabla 
            # https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html
            df2_interaccion2 = pd.pivot_table(
                df2_interaccion.head(Top_Interacciones), 
                values='N', 
                index=['quien_envia'],
                columns=['quien_responde'], 
                aggfunc=np.sum
                ).fillna(0)


            # https://stackoverflow.com/questions/49827096/generating-a-plotly-heat-map-from-a-pandas-pivot-table
            fig10 = go.Figure(data=go.Heatmap({
                'z': df2_interaccion2.values.tolist(),
                'x': df2_interaccion2.columns.tolist(),
                'y': df2_interaccion2.index.tolist()
                        }))
            
            fig10.update_layout(
                xaxis_title='Quien responde mensaje',
                yaxis_title='Quien envia mensaje'
                )
            
            return fig10
            
        
 #-------------------------------------------------------------------------------------------------------------
 # [10.3] Generar entregable y mostrar 
 #-------------------------------------------------------------------------------------------------------------

        fig10 = entregable_10(slider8)
        
        # mostrar grafico         
        st.plotly_chart(fig10, use_container_width=True)

    
 #=============================================================================================================
 # [11] Top Monologos (quienes mas hablan sobre sus propios mensajes)
 #=============================================================================================================   

 #-------------------------------------------------------------------------------------------------------------
 # [11.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------

        # titulo del capitulo 
        st.markdown('### 9. Quienes Envian mas mensajes encadenados/consecutivos')
    
        # Generar Slider 
        slider11 = st.slider(
            label = 'Seleccionar Top', 
            min_value=1, 
            max_value=55, 
            value=10,
            key = 11
            )
    
 #-------------------------------------------------------------------------------------------------------------
 # [11.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------

        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_11(top_monologos): 
    
            fig11 = px.bar(
                df2_monologo.head(top_monologos), 
                x="Emisor", 
                y="N",
                barmode='group',
                title="Top Emisores de Mensajes Consecutivos"
                )

            fig11.update_layout(
                legend=dict(orientation="h",yanchor="bottom",y=1.02, xanchor="right",x=1),
                width=900,
                height=500
                )
            
            fig11.update_traces(marker_color='mediumorchid') # https://developer.mozilla.org/en-US/docs/Web/CSS/color_value
            
            return fig11



 #-------------------------------------------------------------------------------------------------------------
 # [11.3] Generar entregable y mostrar 
 #-------------------------------------------------------------------------------------------------------------

        fig11 = entregable_11(slider11)

        st.plotly_chart(fig11, use_container_width=True)    
    
 #=============================================================================================================
 # [12] Tabla de asociacion de Top palabras (algoritmo apriori)
 #=============================================================================================================   
    
    
 #-------------------------------------------------------------------------------------------------------------
 # [12.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------


        # titulo del capitulo 
        st.markdown('### 10. Relacion entre palabras (algoritmo apriori)')
      
        # definir columnas 
        col12_1, col12_2, col12_3 = st.columns((1, 1, 3))
    
    
        # Generar Slider a
        slider12a = col12_1.slider(
            label = 'min_support', 
            min_value=0.05, 
            max_value=0.12, 
            value=0.09,
            step= 0.01
            )
        
        # Generar Slider b
        slider12b = col12_2.slider(
            label = 'min_confidence', 
            min_value=0.1, 
            max_value=0.5, 
            value=0.3,
            step= 0.02
            )
        
        # listar palabras a omitir 
        text_input12 = col12_3.text_input(
            'Ingresa palabras separadas por coma para excluir',
            value='omitido,multimedia',
            key=12)


 #-------------------------------------------------------------------------------------------------------------
 # [12.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------

        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_12(p1_min_support,p2_min_confidence,p3_exclusiones):   
            
            # filtrar palabras a usar 
            df2_palabras_apriori = df2_palabras.loc[
                ~df2_palabras['palabra'].isin(p3_exclusiones.split(',')),
                ['Emisor','palabra']
                ]
                

            # https://stackoverflow.com/questions/62270442/how-to-convert-a-dataframe-into-the-dataframe-for-apriori-algorithm
            lista_apriori = df2_palabras_apriori.groupby('Emisor')['palabra'].apply(list).values

            # arrojar regla de priori 
            reglas = apriori(
                transactions = lista_apriori, 
                min_support = p1_min_support, 
                min_confidence = p2_min_confidence, 
                min_lift = 3, 
                min_length = 2, 
                max_length = 2
                )


            df_reglas = pd.DataFrame(
                inspect(list(reglas)), # usar funcion propia "inspect" creada en el comienzo
                columns = ['Antecedente', 'Consecuente', 'Soporte', 'Confianza', 'Lift']
                )

            # ordenar relaciones entre palabras 
            df_reglas = df_reglas.sort_values(by=['Confianza'], ascending=False)
            
            return df_reglas
            
        
        
 #-------------------------------------------------------------------------------------------------------------
 # [12.3] Generar entregable y mostrar 
 #-------------------------------------------------------------------------------------------------------------
 
        df_reglas = entregable_12(slider12a,slider12b,text_input12)       
        
        # https://docs.streamlit.io/library/api-reference/data/st.dataframe
        st.dataframe(df_reglas)            
        
        # dejar disponible para descagar df
        st.download_button(
            "Presiona para descargar tabla",
            df_reglas.to_csv().encode('utf-8'),
            "Reglas.csv",
            "text/csv",
            key='download-csv'
            )
            


 #=============================================================================================================
 # [13] Grafico de quienes mas usan alguna palabra (parametro por ingresar)
 #=============================================================================================================   
     
 #-------------------------------------------------------------------------------------------------------------
 # [13.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------


        # titulo del capitulo 
        st.markdown('### 11. Top emisores que usan alguna palabra')
      
        # definir columnas 
        col13_1, col13_2 = st.columns((1, 1))
    
        text_input13 = col13_1.text_input(
            'Ingresa palabra a buscar',
            value='gracias',
            key=17)
    
    
        # Generar Slider
        slider13 = col13_2.slider(
            label = 'Top Emisores', 
            min_value=2, 
            max_value=20, 
            value=10
            )


 #-------------------------------------------------------------------------------------------------------------
 # [13.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------

        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_13(palabra_buscar,top_emisores):   

            # agrupar tabla 
            df2_AGG_busqueda=df2[df2['Mensaje'].str.contains(palabra_buscar)].groupby(['Emisor']).agg(
                Conteo = pd.NamedAgg(column = 'Emisor', aggfunc = len)
                )
            df2_AGG_busqueda.reset_index(level=df2_AGG_busqueda.index.names, inplace=True) # pasar indices a columnas

            # ordenar y contar
            df2_AGG_busqueda = df2_AGG_busqueda.sort_values(by=['Conteo'], ascending=False)


            fig13 = px.bar(
                df2_AGG_busqueda.head(top_emisores), 
                x="Emisor", 
                y="Conteo",            
                barmode='group',
                title="Top Emisores que usan palabra: "+palabra_buscar
                )
            
            fig13.update_traces(marker_color='goldenrod')
            
            fig13.update_layout(
                legend=dict(orientation="h",yanchor="bottom",y=1.02, xanchor="right",x=1),
                width=900,
                height=500
                )
            
            return fig13

 #-------------------------------------------------------------------------------------------------------------
 # [13.3] Generar entregable y mostrar 
 #-------------------------------------------------------------------------------------------------------------

        fig13 = entregable_13(text_input13,slider13)

        st.plotly_chart(fig13, use_container_width=True)
        
        
 #=============================================================================================================
 # [14] Agrupar Emisores usando k-means 
 #=============================================================================================================

 #-------------------------------------------------------------------------------------------------------------
 # [14.1] Titulo + Widgets de inputs 
 #-------------------------------------------------------------------------------------------------------------
 

        # titulo del capitulo 
        st.markdown('### 12. Clasificacion de Emisores segun k-means')
      
        # definir columnas 
        col14_1, col14_2 = st.columns((4, 1))
    
    
        # Ingresar widget de seleccion variables  
        multiselect14 = col14_1.multiselect(
            'Seleccionar Variables',
            df2_AGG_Resumen.select_dtypes([np.number]).columns,
            ['N_Mensajes','N_Dias','N_Palabras','Mjs_x_dia','Palabras_x_Mjs'],
            key = 15
            )    
    
        # Generar Slider
        slider14 = col14_2.slider(
            label = 'N de Clusters', 
            min_value=2, 
            max_value=6, 
            value=4
            )
        
 #-------------------------------------------------------------------------------------------------------------
 # [14.2] Procesamiento previo al entregable (funcion)
 #-------------------------------------------------------------------------------------------------------------

        @st.cache(suppress_st_warning=True) # https://docs.streamlit.io/library/advanced-features/caching
        def entregable_14(variables,N_cluster):   
       
            # normalizar tabla en forma de arreglo 
            df2_AGG_Resumen_KM_norm = np.array(
                preprocessing.MinMaxScaler().fit_transform(
                    df2_AGG_Resumen[variables].values
                    )
                )

            # Se ejecuta Algoritmo
            kmeans = KMeans(n_clusters=N_cluster).fit(df2_AGG_Resumen_KM_norm)
                    
            # Crear duplicado de Base (solo columnas relevantes)
            df2_AGG_Resumen_KM = df2_AGG_Resumen.loc[:,['Emisor']+list(variables)]

            # Asignar Prediccion
            df2_AGG_Resumen_KM['Cluster_KM']=kmeans.predict(df2_AGG_Resumen_KM_norm)

            # Se edita la variable (se suma 1 y se pasa a string agregando prefijo)
            df2_AGG_Resumen_KM['Cluster_KM'] = df2_AGG_Resumen_KM['Cluster_KM'].apply(lambda x: 'C'+str(x+1))
                        
            # agrupar tabla por cluster 
            df2_AGG_Resumen_KM_c=df2_AGG_Resumen_KM.groupby('Cluster_KM').agg(np.mean)
            df2_AGG_Resumen_KM_c.reset_index(level=df2_AGG_Resumen_KM_c.index.names, inplace=True) # pasar indices a columnas
            
            return df2_AGG_Resumen_KM,df2_AGG_Resumen_KM_c


 #-------------------------------------------------------------------------------------------------------------
 # [14.3] Generar entregable y mostrar 
 #-------------------------------------------------------------------------------------------------------------

        df2_AGG_Resumen_KM,df2_AGG_Resumen_KM_c = entregable_14(multiselect14,slider14)

 #____________________________________
 # Entregable de tabla detallada 
  
        # Titulo del capitulo 
        st.markdown('#### 12.1 Tabla detallada con asignacion de Cluster')
        st.markdown('Las variables fueron normalizadas previa ejecucion de k-medias y luego reversadas')


        # Mostrar df resultante 
        st.dataframe(df2_AGG_Resumen_KM)            
        
        # dejar disponible para descagar df
        st.download_button(
            "Presiona para descargar tabla",
            df2_AGG_Resumen_KM.to_csv().encode('utf-8'),
            "Detalle_k_means.csv",
            "text/csv",
            key='download-csv'
            )

 #____________________________________
 # Entregable de tabla resumen 
      
        # Titulo del capitulo 
        st.markdown('#### 12.2 Tabla resumen de promedios del centroide')
        
        
        # Mostrar df resultante 
        st.dataframe(df2_AGG_Resumen_KM_c)            
        
        # dejar disponible para descagar df
        st.download_button(
            "Presiona para descargar tabla",
            df2_AGG_Resumen_KM_c.to_csv().encode('utf-8'),
            "Centroides_k_means.csv",
            "text/csv",
            key='download-csv'
            )

 #-------------------------------------------------------------------------------------------------------------
 # [14.4] Grafico de dispersion 
 #-------------------------------------------------------------------------------------------------------------   
      
        # Titulo del capitulo 
        st.markdown('#### 12.3 Grafico de dispersion de segmentos')
        
        
        # definir columnas 
        col14c_1, col14c_2,col14c_3,col14c_4 = st.columns((1,1,1,1))
    
        # eje x 
        selectbox14_ejex=col14c_1.selectbox(
            'Eje X',
            df2_AGG_Resumen_KM.select_dtypes([np.number]).columns
            )
        
        # eje y 
        selectbox14_ejey=col14c_2.selectbox(
            'Eje Y',
            df2_AGG_Resumen_KM.select_dtypes([np.number]).columns
            )
        
        # tamaño
        selectbox14_size=col14c_3.selectbox(
            'Tamaño',
            df2_AGG_Resumen_KM.select_dtypes([np.number]).columns
            )
                
        # color 
        selectbox14_color=col14c_4.selectbox(
            'Color',
            df2_AGG_Resumen_KM.columns,
            index = len(df2_AGG_Resumen_KM.columns)-1
            )
    
        # generar grafico 
        fig14 = px.scatter(
            df2_AGG_Resumen_KM, 
            x=selectbox14_ejex, 
            y=selectbox14_ejey, 
            color=selectbox14_color,
            size=selectbox14_size,
            symbol = 'Cluster_KM',
            marginal_x='box', 
            marginal_y='box',
            hover_data=['Emisor'] 
            )    
         
        fig14.update_layout(
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=0.5
                ), 
            width=650,
            height=650
            )

        # mostrar grafico         
        st.plotly_chart(fig14, use_container_width=True)
        

# arrojar main para lanzar App
if __name__=='__main__':
    main()
    
# Escribir en terminal: streamlit run App_chat_WSP_V20221225.py
# !streamlit run App_chat_WSP_V20221225.py

