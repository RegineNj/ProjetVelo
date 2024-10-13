import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
#@st.cache_data

st.set_page_config(
    page_title="Projet Regression Trafic Cycliste à Paris",
    layout="wide",
    initial_sidebar_state="expanded",
    )  

# Header
img = Image.open("Bikes.jpg")
st.image(img)
st.markdown("<h1 style='text-align: center; color: goldenrod;'>Analyse du Trafic Cycliste à Paris</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Projet Analyse du Trafic Cycliste à Paris")
st.sidebar.title("Sommaire")
pages=["Projet","Jeux de données","Data Visualisation", "Modélisation et prédiction", "Conclusions"]
page=st.sidebar.radio("Aller vers", pages)
# Encadrement du sidebar
texte = """
<div style="text-align: center; border: 0.75px solid #040925; padding: 7px; border-radius: 5px; background-color: #9ec9e';">
**TEAM**\n

Regine **NJAWE OUENDJI**\n
Abderrahmane **AZAMI IDRISSI**\n 
Silvana **DIAZ PAREDES**\n
**BDA Février 2024**
 </div>
"""
st.sidebar.markdown(texte, unsafe_allow_html=True)

# Dataset principal
df=pd.read_csv("dataset_projet_velo.csv")

list_compteurs = df['nom_compteur'].unique().tolist()
cols=['id_compteur','nom_compteur']
df_compteurs = df[cols].drop_duplicates()
def get_id_compteur(df, nom):
     index = df.index[df['nom_compteur']== nom]
     i= index[0]
     return df['id_compteur'][i]

# PAGES
# Premier page : PROJET
if page==pages[0]:
    # Définir le texte à afficher dans l'encadré
     texte = """
     <div style="text-align: center; border: 2px solid #000000; padding: 20px; border-radius: 10px; background-color: #ccc8b2;">
     Projet réalisé dans le cadre de la formation Data Analyst de DataScientest\n
     **Promotion BDA Février 2024**\n
     **TEAM :**\n
     Regine NJAWE OUENDJI\n
     Abderrahmane AZAMI IDRISSI\n
     Silvana DIAZ PAREDES\n
     **Source de données :** [Comptage vélos | Open Data | Ville de Paris](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/api/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJjb2x1bW4iLCJmdW5jIjoiQVZHIiwieUF4aXMiOiJzdW1fY291bnRzIiwic2NpZW50aWZpY0Rpc3BsYXkiOnRydWUsImNvbG9yIjoiI0ZBOEM0NCJ9XSwieEF4aXMiOiJpZCIsIm1heHBvaW50cyI6IiIsInRpbWVzY2FsZSI6IiIsInNvcnQiOiIiLCJjb25maWciOnsiZGF0YXNldCI6ImNvbXB0YWdlLXZlbG8tZG9ubmVlcy1jb21wdGV1cnMiLCJvcHRpb25zIjp7ImRpc2p1bmN0aXZlLmlkX2NvbXB0ZXVyIjp0cnVlLCJkaXNqdW5jdGl2ZS5ub21fY29tcHRldXIiOnRydWUsImRpc2p1bmN0aXZlLmlkIjp0cnVlLCJkaXNqdW5jdGl2ZS5uYW1lIjp0cnVlfX19XSwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZSwidGltZXNjYWxlIjoiIn0%3D)
     </div>
     """
     st.markdown(texte, unsafe_allow_html = True)
     st.markdown("\n")
     st.markdown("<h3 style='text-align: center; color: #55562d;'>Introduction au projet</h3>", unsafe_allow_html=True)
     # Afficher l'encadré avec le texte centré
     st.write("")

     # Texte à afficher avec une mise en forme spécifique
     texte = """
     **CONTEXTE** 

     En octobre 2021, la  Mairie de Paris a présenté son plan vélo 2021-2026, qui consiste en la 
     construction de 180km de nouvelles pistes cyclables sécurisées, de 100.000 places de stationnement, et zone à trafic limité 
     de voitures dans le centre de Paris. Son objectif, c’est de faire une ville 100% cyclable.
     Notre analyse sera portée sur quelques axes :  

     1. **<span style="color: #55562d;"> Analyse du trafic </span>** (Zone d'affluence ) 
     2. **<span style="color: #55562d;"> Évolution temporelle :</span>** quels facteurs influencent le trafic ?
     3. **<span style="color: #55562d;"> Prédiction du trafic :</span>** modèles de machine learning

     """
     # Affichage du texte dans Streamlit
     st.markdown(texte, unsafe_allow_html=True)
     # Texte encadré
     st.info("Pour accéder à nos analyses détaillées et interactives (Jeux de données, Data visualisation, Modélisation et Prédictions), cliquez dans le menu de gauche.")
    

# Deuxième page : JEUX DE DONNÉES
if page == pages[1]:  
    st.markdown("<h3 style='text-align: center; color: #55562d;'>Jeux de Données</h3>", unsafe_allow_html=True)
    dataset_selectionne = st.radio("Sélectionner :", ["Dataset Principal", "Datasets Secondaires"])

    if dataset_selectionne == "Dataset Principal":     
         # Contenu du Dataset Principal
         st.markdown("""
         **DATASET PRINCIPAL**

         **1. SOURCE**\n
         Le jeu de données provient du site de la Mairie de Paris : Comptage Vélo – Données compteurs

         **2. PÉRIODE**\n
         Les données sont mises à jour quotidiennement et remontent sur 13 mois glissants. Nous avons récupéré toutes les données du 1er 
         Janvier 2023 au 25 Février 2024 (14 mois).

         **3. REMARQUES**\n
         Les données sont fournies par un prestataire : Eco-Compteur.
         Le nombre de compteurs fluctue. Certains sont créés, d’autres arrêtés en cas de travaux ou de panne.

         **4. EXPLORATION DES DONNÉES:**\n
         Langage utilisé : Python\n
         Librairies utilisées : Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

         **a. Taille du DataFrame:**\n
         992 782 lignes et 16 variables

         **b. Les variables**
         """)
         col1,col2,col3 = st.columns([1,2,1])
         with col1:
             st.write()
         with col2:
            st.markdown("""                 
                | N° colonne | Nom de la colonne           | Type           |
                |------------|-----------------------------|----------------|
                | 1          | Identifiant du compteur     | string         |
                | 2          | Nom du compteur             | string         |
                | 3          | Identifiant du site de comptage | string      |
                | 4          | Nom du site de comptage     | string         |
                | 5          | Comptage horaire            | integer        |
                | 6          | Date et heure de comptage   | string         |
                | 7          | Date d'installation du site de comptage | string |
                | 8          | Lien vers photo du site de comptage | string |
                | 9          | Coordonnées géographiques   | string         |
                | 10         | Identifiant technique compteur | string      |
                | 11         | ID photos                   | string         |
                | 12         | test_lien_vers_photos_du_site_de_comptage | string |
                | 13         | id_photo_1                  | string         |
                | 14         | url_sites                   | string         |
                | 15         | type_image                  | string         |
                | 16         | mois_année_comptage         | string         |           
            """)
         with col3:
            st.write()
         st.markdown(""" \n""")
         st.markdown(""" \n""")
         st.markdown("""
         **c. Doublons**\n
         Le dataset ne contient pas de doublons.

         **d. Valeurs manquantes**\n
         Il y a 64816 données, soit **6,58% du dataset**. Elles concernent 6 variables, toutes liées à l'identification des compteurs.

         **e. Valeurs extrêmes de la variable “Comptage horaire”**\n
         Les statistiques descriptives nous montrent un range des valeurs  avec un minimum de 0 vélo/heure/site et un maximum de 8190. 
         Car la moyenne  a une valeur d’environ d'environ 76 et une médiane de 42, nous pouvons conclure qu'il y a des valeurs extremes 
         ou aberrantes dans notre ensemble de données. Pour le moment, nous les gardons. Un boxplot nous permet de les visualiser.
                    
         **5. AJOUT DE VARIABLES**\n
         Pour alimenter notre analyse, nous créons 9 variables.

         **a. Périodicité** : année, mois, jour, jour de la semaine.

         **b. Événements récurrents** : jours fériés, vacances scolaires, météo (pluie, temperature).

         **c. Evènements exceptionnels** : grève.
                     
         **6. LA VARIABLE CIBLE**\n
         La variable cible est le “Comptage horaire”, c’est-à-dire le nombre de vélos/heure/site. C’est la seule variable numérique continue.
            
         """)
         col4, col5 = st.columns([1,2])
         with col4:
            st.markdown('La description de **Comptage horaire**')
            st.dataframe(df['comptage_horaire'].describe())   
         with col5:
            st.markdown ("Boxplot de **'Comptage horaire'** entre [ 0 , 500 ]")
            st.image("Boxplot_Comptage_horaire.png")
            #fig = plt.figure()
            #sns.boxplot(df.loc[df['comptage_horaire']<500]['comptage_horaire'])
            #plt.ylabel('Nombre de vélos par heure et site')
            #st.pyplot(fig)
         st.markdown("\n")
         # Extrait du dataset
         st.markdown("""
         Extrait du DataFrame           
         """)
         st.dataframe(df.head(5))
         if st.checkbox("Afficher les NA"):
                st.dataframe(df.isna().sum())
        
    # Afficher le bouton pour le Dataset Secondaire
    elif dataset_selectionne == "Datasets Secondaires":
         st.markdown ("""
         **A. DATASET METEO**\n  
                      
         **1. SOURCE:**\n             
         Le deuxième jeu de donnes provient de la plateforme ouverte de données publiques française https://donneespubliques.meteofrance.fr.

         **2. TRAITEMENT DE DONNÉES:**\n
         Ce jeu de données comprend plusieurs variables,  mais nous avons sélectionné la température minimale, la température maximale et 
         les précipitations en millimètre (mm) de pluie pour une date et une heure données. Cela nous permettra de savoir comment ces facteurs
         affectent le trafic cycliste.\n 

         Il est constitué de 10468 lignes et 5 variables.\n
         Il n’y a ni valeur manquante ni doublon.\n
                      
         Extrait du DataFrame :
         """)
         df_meteo=pd.read_csv("meteo_paris_2023_2024.csv",sep=';')
         st.dataframe(df_meteo.head(5))
         st.markdown("\n")

         st.markdown ("""
         **B. DATASET CALENDRIER**\n

         Le troisième jeu de données a été cree par nous, pour identifier les jours exceptionnels comme les jours fériés, les vacances 
         scolaires et les jours de grève de transport.\n 
         Il est formé de 456 lignes et 4 variables.\n
         Extrait du DataFrame :
         """)
         df_calendrier=pd.read_csv("db_paris_calendrier_2023_2024.csv",sep=';')
         st.dataframe(df_calendrier.head(5))
                          
# Troisième page : DATA VISUALISATION
if page == pages[2]:
    st.markdown("<h3 style='text-align: center; color: #55562d;'>Data Visualisation</h3>", unsafe_allow_html=True)
    # On defini les parametres de visualisation
    sns.set_theme(style='whitegrid',context='paper',palette='Set2')
    dataviz_selectionne = st.radio("Sélectionner :", ["Première Analyse du trafic", "Influences des variables sur la 'comptage_heure'","Top compteurs par trafic"])

    if dataviz_selectionne == "Première Analyse du trafic":
        st.markdown("\n")
        st.markdown("""                           
        **1. ANALIYSE DU TRAFIC DES VÉLOS À PARIS**
        """)
        col1,col2 = st.columns([1,3])
        with col1: 
            st.markdown("\n")
            st.markdown("\n")
            st.markdown("\n")
            st.markdown("""
            Cette carte montre la répartition des comptages horaires par compteur et localisation <span style='text-decoration: underline;'> **sur les trois derniers mois.**</span> La taille du cercle
            est proportionnelle au nombre de vélos comptés.\n
            """, unsafe_allow_html=True)
        with col2:
            ## Graphique: Carte de dispersion du compteurs à Paris 
            df_filtre = df[df['mois_annee_comptage'].isin(['2024-02', '2024-01', '2023-12'])]
            fig2 = px.scatter_mapbox(df_filtre, lat='latitude', lon='longitude', color='comptage_horaire',hover_name='nom_compteur', color_continuous_scale='matter',
                                size='comptage_horaire', zoom=11, center=dict(lat=48.8566, lon=2.3522), mapbox_style="carto-positron")
            fig2.update_layout(margin=dict(l=70, r=70, t=60, b=50))
            st.plotly_chart(fig2)

        st.markdown("""
         **2. EVOLUTION DE TRAFIC DES VÉLOS MOYENNE PAR HEURE PAR SITE**\n
         """)
        col1,col2 = st.columns([1,3])
        with col1:
            st.markdown("\n")
            st.markdown("""
            Ce graphique montre l'évolution du trafic moyen de vélos par heure par site sur la période janvier 2023 au février 2024. On observe une certaine saisonnalité du trafic.
            """)
        with col2:
            st.markdown("\n")
            st.image("Evolution_Trafic_Velos.png", width=10, use_column_width="always")
            st.markdown("\n")

        st.markdown("""
         **3. DISTRIBUTION DE LA VARIABLE "Comptage horaire"**\n
        """)
        st.markdown("\n")
         ## Graphique: Distribution de la variable "comptage_horaire"
        col1, col2,col3 = st.columns([2,3,1])
        with col1:
            limite = st.select_slider('Sélectionner la valeur maximum : ', options=[100,200,500,1000,8000])
            st.markdown("\n")
            st.markdown("\n")
            st.markdown("""
            Ce graphique montre la distribution des valeurs du comptage de vélos par heure par site sur la période janvier 2023 au février 2024. On observe que la majorité des valeurs se trouvent entre 0 et 100.             """)
        with col2:
            fig3 = sns.displot(df.loc[df['comptage_horaire']<limite]['comptage_horaire'],bins=50 , color= "#8D8741")
            plt.xlabel('Comptage horaire')
            plt.ylabel('Nombre de registres')
            st.pyplot(fig3)
        with col3:
            st.write("")

    elif dataviz_selectionne == "Influences des variables sur la 'comptage_heure'":
        st.markdown("""
          **4. INFLUENCE DES VARIABLES DESCRIPTIVES PÉRIODIQUES AU "Comptage horaire"**\n
          """)
        st.markdown("\n")
          ## Graphique: Relation des variables descriptives au "comptage_horaire" part 1
        col1, col2 = st.columns([2,4])
        with col1:
            st.markdown("\n")
            lista_var=['mois_comptage','heure_comptage','jour_sem_comptage','temperature','pluie']
            choix_var = st.selectbox('Choix de la variable',lista_var)
            st.markdown("\n")
            if choix_var=='mois_comptage':
                st.markdown("""
                Nous pouvons constater que le mois de Juin et le mois de Septembre ont les valeurs moyennes des comptages horaires les plus élevés. 
                Le mois d'Août quant à lui a le comptage le plus faible. La période hivernale (Décembre, Janvier, Février) enregistre également moins de comptage.
                """)
            elif choix_var=='heure_comptage':
                st.markdown("""
                Ce graphique nous permet de déduire le fait qu’on enregistre un grand nombre moyenne de vélos aux heures de pointe (8h et 18h) , on remarque que sur les 
                périodes de sommeil, les enregistrements sont très bas ( entre 00h et 6h ).
                """)
            elif choix_var=='jour_sem_comptage':
                st.markdown("""
                Dans ce graphique, nous pouvons visualiser que  le nombre de vélos par heure par site  est entre 0 et 100. Et que les jours mardi, mercredi et jeudi ont a peu plus traffic. 
                Aussi, il y a des valeurs extremes dans chaque jour.
                """)
                st.markdown("\n")
                st.markdown("**Remarque :** lundi : 0 , mardi : 1, ....")
            elif choix_var=='temperature':
                st.markdown("""
                Dans ce graphique, nous pouvons visualiser que l’affluence des vélos augmente quand la temperature est entre 18 °C et 27 °C. Nous pouvons également observer que lorsque les 
                températures sont basses, le trafic est moins important que lorsque les températures sont très élevées.
                """)
            elif choix_var=='pluie':
                st.markdown("""
                On peut clairement voir que lorsqu’il ne pleut pas du tout (précipitations = 00 mm), on a beaucoup plus de vélos qui circulent à Paris.
                """)
        with col2:
            df_small= df[df['comptage_horaire']<500]
            if choix_var == 'mois_comptage':
                fig4 = plt.figure()
                sns.lineplot(x = 'mois_comptage', y = 'comptage_horaire', marker = 'o', color= '#8d8741',  data = df)
                plt.xlabel(choix_var)
                plt.ylabel('Comptage horaire')
                #plt.xticks(rotation=90)
                st.pyplot(fig4)
            elif choix_var == 'pluie':
                fig4 = plt.figure(figsize=(8, 6))
                sns.scatterplot(x='pluie', y='comptage_horaire', hue='pluie',data=df_small)
                plt.xlabel('Pluie (mm)')
                plt.ylabel('Comptage Horaire')
                st.pyplot(fig4)
            else:
                fig4 = plt.figure()
                sns.lineplot(x = choix_var, y = 'comptage_horaire', marker = 'o', color= '#659dbd',data = df)
                plt.xlabel(choix_var)
                plt.ylabel('Comptage horaire')
                #plt.xticks(rotation=90)
                st.pyplot(fig4)

    #      st.markdown("""
    #      **5. INFLUENCE DES VARIABLES DESCRIPTIVES RÉCCURENTS ET EXCEPTIONNELS AU "Comptage horaire"**\n
    #      """)
    #      st.markdown("\n")
    #      ## Graphique: Relation des variables descriptives au "comptage_horaire" part 2
    #      col1, col2 = st.columns([2,4])
    #      with col1:
    #          st.markdown("\n")
    #          lista_var=['vacances_scolaire','jour_feriee','greve']
    #          choix_var = st.selectbox('Choix de la variable',lista_var)
    #          st.markdown("\n")
    #          st.markdown("""
    #          Nous pouvons voir ici que le facteur grève impacte énormément le trafic de vélos vers la hausse, et pendant les jours fériés on compte le moins de vélos.
    #          """)
    #      with col2:
    #          fig5 = plt.figure()
    #          sns.swarmplot(x=choix_var, y='comptage_horaire', hue=choix_var, data=df)
    #          plt.ylabel('Comptage Horaire')
    #          plt.xlabel('')
    #          x = [0,1]
    #          label =['normal',choix_var]
    #          plt.xticks(x, label)
    #          st.pyplot(fig5)

    elif dataviz_selectionne == "Top compteurs par trafic":
        st.markdown("""
         **6. TOP 10 DES COMPTEURS AVEC LA MOYENNE HORAIRE LA PLUS ÉLEVÉE**\n
         """)
        st.markdown("\n")
         # Graphique: Top 10 des compteurs avec la moyenne horaire la plus élevée
        col1, col2 = st.columns([1,4])
        with col1:
            st.markdown("\n")
            st.markdown("""
             Ce graphique montre les 10 compteurs avec le nombre moyenne de vélos comptés le plus élevé entre janvier 2023 et février 2024. 
             Ces compteurs peuvent être situés sur des axes cyclables majeurs.
             """)
        with col2:
            st.image("TOP10_Trafic_Elevee.png")
            #df_top_moyenne = df.groupby('nom_compteur')['comptage_horaire'].mean().sort_values(ascending=False).head(10)
            #fig6, ax6 = plt.subplots(figsize=(15, 10))
            #df_top_moyenne.plot(kind='barh', ax=ax6, figsize=(15, 10), color= "#8d8741", title='Top 10 des Compteurs avec la Moyenne Horaire la plus élevée')
            #ax6.set_xlabel('Nombre moyen de vélos par heure')
            #st.pyplot(fig6)

        st.markdown("""
         **7. TOP 10 DES COMPTEURS AVEC LA MOYENNE HORAIRE LA PLUS FAIBLE**\n
         """)
        st.markdown("\n")
        ## Graphique: Top 10 des compteurs avec la moyenne horaire la plus faible
        col1, col2 = st.columns([1,4])
        with col1:
            st.write()          
        with col2:
            st.image("TOP10_Trafic_Faible.png")
            #df_top_moyenne_faible = df.groupby('nom_compteur')['comptage_horaire'].mean().sort_values(ascending=True).head(10)
            #fig7, ax7 = plt.subplots(figsize=(15, 10))
            #df_top_moyenne_faible.plot(kind='barh', ax=ax7, figsize=(15, 10), color="#659dbd",title='Top 10 des Compteurs avec la Moyenne Horaire la plus faible')
            #ax7.set_ylabel('Nombre moyen de vélos par heure')
            #st.pyplot(fig7)

        #st.markdown("""
        # **8. TOP 10 DES COMPTEURS AVEC LE PLUS GRAND NOMBRE DES VÉLOS**\n
        #   """)
        #st.markdown("\n")
         ## Graphique: Top 10 des compteurs avec les plus grand nombre de velos
        #col1, col2 = st.columns([1,4])
        #with col1:
        #     st.markdown("\n")
        #     st.markdown("""
        #     Ce graphique montre les 10 compteurs avec le nombre total de vélos comptés le plus élevé entre janvier 2023 et février 2024. 
        #     Ces compteurs peuvent être situés sur des axes cyclables majeurs.
        #      """)
        #with col2:
        #     df['total_velos_comptes'] = df.groupby('nom_compteur')['comptage_horaire'].transform('sum')
        #     df_top_total = df.groupby('nom_compteur')['total_velos_comptes'].sum().sort_values(ascending=False).head(10)
        #     fig8, ax8 = plt.subplots(figsize=(15, 10))
        #     df_top_total.plot(kind='barh', ax=ax8, figsize=(15, 10), color="#daad86", title='Top 10 Compteurs avec le Plus Grand Nombre de Vélos')
        #     ax8.set_ylabel('Nombre total de vélos comptés (Janvier 2023 - Février 2024)')
        #     st.pyplot(fig8)

# Quatrième page : MODÉLISATION et PRÈDICTION 
if page == pages[3]:
    st.markdown("<h3 style='text-align: center; color: #55562d;'>Modélisation et Prédiction</h3>", unsafe_allow_html=True)
    selection = st.radio("Sélectionner :", ["Modélisation", "Prédiction"])

    import sklearn
    import category_encoders as ce
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib

    df = pd.read_csv("db2023_modele_projet_velo.csv")
    df_test = pd.read_csv("db2024_projet_velo.csv")
        
    df = df.loc[df['comptage_horaire']<1000]
    df_test = df.loc[df['comptage_horaire']<1000] 
        
    #On defini l'ensemble d'entraînement et de test
    y_train = df['comptage_horaire']
    X_train = df.drop('comptage_horaire',axis=1)
    y_test = df_test['comptage_horaire']
    X_test = df_test.drop('comptage_horaire',axis=1)

    # On encode la varible cible
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    #On encode la variable 'id_compteur'
    categorical_columns = ['id_compteur']
    loo_encoder = ce.leave_one_out.LeaveOneOutEncoder(cols=categorical_columns)
    X_train = loo_encoder.fit_transform(X_train, y_train)
    X_test = loo_encoder.transform(X_test)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)       

    if selection == "Modélisation":
        st.markdown("""
         **MODÈLISATION**\n            
         **1. CHOIX DES VARIABLES**\n
         En ayant déjà identifié la variable ‘comptage_horaire’ comme le variable cible, nous avons sélectionnés comme variables 
         descriptives : ‘id_compteur’,  ’annee_comptage', ‘mois_comptage', ‘jour_comptage', ‘jour_sem_comptage’, ‘heure_comptage’, 
         ‘vacances_scolaire' , ‘jour_feriee', ’greve', ‘temperature' et ‘pluie’.\n

         **2. ELIMINATION DES VALEURS EXTRÊMES OU ABERRANTES**\n
         Étant donné que la médiane de la variable "comptage_horaire" est de 40 vélos par heure et par site et la moyenne  est 74.745 ,
         et qu'il n'y a seulement que 379 valeurs supérieures à 1000 vélos/heure/site, on a décidé de supprimer ces valeurs afin que le 
         modèle ne soit pas influencé par elles. On peut remarquer, que nous avons confirmé que ces valeurs de compteurs, ont été 
         enregistré par les compteurs pendant jours de manifestations ( telles que, par exemple, les manifestations pour la retraite). 
         C'est-à-dire que les compteurs n'ont pas seulement enregistré des vélos, mais aussi des personnes qui marchent. 
         C'est pourquoi les chiffres sont si élevés.

         **3. ENCODAGE DE LA VARIABLE 'id_compteur'**\n
         Nous avons encodé la variable "id_compteur", la seule variable qualitative, à l'aide de la technique LOO (Leave-One-Out Encoding). 
         On utilise cette technique, parce qu’elle fonctionne très bien quand la variable qualitative comporte un grand nombre de données, 
         comme le cas de “id_compteur" ( 101 compteurs).
                                         
         **4. CHOIX DU TYPE DE MÉTHODOLOGIE**\n
         Comme la **variable cible Comptage horaire est une variable continue**, on va utiliser des **méthodes de Regression** pour la modèlisation.\n
         
         **5. ENTRAÎNEMENT DU MODÉLE**\n                      
         """)
        def prediction(classifier):
            if classifier =='Régression Linéaire':
                clf = joblib.load('model_rl')
            if classifier =='Decision Tree Regressor':
                clf = DecisionTreeRegressor(max_depth=30)
                clf.fit(X_train, y_train)
            #if classifier =='Random Forest Regressor':
            #    clf = RandomForestRegressor()
            #    clf.fit(X_train, y_train)
            if classifier =='XGBoost Regressor':
                clf = XGBRegressor(learning_rate=1.0, n_estimators=100,max_dept=1, random_state=0)
                clf.fit(X_train, y_train)
            return clf
    
        def scores(clf):
            # Les scores d'entraînement
            r2_ent = r2_score(y_train,clf.predict(X_train))
            mse_ent = mean_squared_error(y_train,clf.predict(X_train))
            rmse_ent = mean_squared_error(y_train,clf.predict(X_train),squared=False)
            mae_ent = mean_absolute_error(y_train,clf.predict(X_train))
            list_ent = [r2_ent,mse_ent,rmse_ent,mae_ent]

            #Les scores de test
            r2_test = r2_score(y_test,clf.predict(X_test))
            mse_test = mean_squared_error(y_test,clf.predict(X_test))
            rmse_test = mean_squared_error(y_test,clf.predict(X_test),squared=False)
            mae_test = mean_absolute_error(y_test,clf.predict(X_test))
            list_test = [r2_test,mse_test,rmse_test,mae_test]

            metrics = ['R^2','MSE','RSME','MAE']
            df_scores = pd.DataFrame(zip(list_ent,list_test), index=metrics, columns=['Entraîment', 'Test'])
            return (st.dataframe(df_scores))
    
        def impPlot(imp, name):
            figure = px.bar(imp, x=imp.values, y=imp.keys(), labels = {'x':'importance', 'index':'variables'},
                    text=np.round(imp.values, 2), title="L'importance des variable du "+ name, width=700, height=400)
            figure.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            st.plotly_chart(figure)
        
        #Chosir le classifier
        choix = ['Régression Linéaire', 'Decision Tree Regressor', 'Random Forest Regressor', 'XGBoost Regressor']
        option = st.selectbox('Choix du modèle',choix)

        #Entraîner le modèle choisi
        if option == 'Random Forest Regressor':
            st.markdown("\n")
            col1,col2,col3 = st.columns([1,2,1])
            with col1:
                st.write()
            with col2:
                df_RF_scores=pd.read_csv('dataset_RF_scores.csv',index_col=0)
                st.dataframe(df_RF_scores)
            with col3:
                st.write()
            st.markdown("\n")
            col4, col5 = st.columns([1,2])
            with col4:
                # L'importance de las variables
                if st.checkbox("Afficher l'importance des variables"):
                    coefs=pd.read_csv('dataset_RF_features.csv',index_col=0)   
                    st.dataframe(coefs)               
            with col5:
                if st.checkbox("Afficher le graphique de l'importance des variables"):
                    st.image("Modele_RF.png")
        else:  
            clf = prediction(option)
            st.markdown("\n")
            col1,col2,col3 = st.columns([1,2,1])
            with col1:
                st.write()
            with col2:
                scores(clf)
            with col3:
                st.write()
            st.markdown("\n")
            col4, col5 = st.columns([1,2])
            with col4:
                # L'importance de las variables
                if st.checkbox("Afficher l'importance des variables"):
                    if option == choix[0]:
                        feats_ = df.drop('comptage_horaire',axis=1)
                        coeffs = list(clf.coef_)
                        coeffs.insert(0,clf.intercept_)
                        feats = list(feats_.columns)
                        feats.insert(0, 'intercept')
                        coefs = pd.DataFrame({'valeur estimée':coeffs}, index=feats)
                        st.markdown("\n")
                        st.dataframe(coefs)               
                    else:
                        feats_ = df.drop('comptage_horaire',axis=1)
                        importances = list(clf.feature_importances_)
                        feats = list(feats_.columns)
                        coefs = pd.DataFrame({'valeur estimée': importances}, index = feats)
                        st.markdown("\n")
                        st.dataframe(coefs)
            with col5:
                if st.checkbox("Afficher le graphique de l'importance des variables"):
                    if option == choix[0]:
                        coeffs = list(clf.coef_)
                        feats_ = df.drop('comptage_horaire',axis=1)
                        feats = list(feats_.columns)
                        coeff = pd.Series(coeffs, index=feats)
                    else:
                        coeff = pd.Series(clf.feature_importances_, index=feats)
                    #Graphique de l'importance de las varaibles
                    impPlot(coeff,option)

    if selection == "Prédiction":
        st.markdown("""
        **PRÈDICTIONS**\n   
        **1. MODÉLE CHOISI : XGBoost Regressor**\n
        
        La classe GridSearch a révélé que le **modèle XGBoostRegressor** avec un **score de 0.954**  et les paramètres :\n
        """)            
        col1,col2,col3 = st.columns([1,2,1])
        with col1:
            st.write()
        with col2:
            st.markdown("""
                                       
            | Paramètres       | Valeur       |
            |------------------|--------------|
            | colsample_bytree | 0.7          |
            | learning_rate    | 1            |
            | max_depth        | 5            |
            | min_child_weight | 5            |
            | n_estimators     | 300          |
            | random_state     | 0            |
            | objective        |count:poisson |
      
            """)
        with col3:
            st.write()
        st.markdown("\n")
            
        clf_choisi = joblib.load("model_xgbC")
        
        if st.checkbox("Afficher les metriques et l'importance des variables du XGBoost amélioré"):
            st.markdown ("\n")
            #scores du modéle choisi
            col1,col2,col3 = st.columns([1,2,1])
            with col1:
                st.write()
            with col2:
                # Les scores d'entraînement
                r2_ent = r2_score(y_train,clf_choisi.predict(X_train))
                mse_ent = mean_squared_error(y_train,clf_choisi.predict(X_train))
                rmse_ent = mean_squared_error(y_train,clf_choisi.predict(X_train),squared=False)
                mae_ent = mean_absolute_error(y_train,clf_choisi.predict(X_train))
                list_ent = [r2_ent,mse_ent,rmse_ent,mae_ent]
                #Les scores de test
                r2_test = r2_score(y_test,clf_choisi.predict(X_test))
                mse_test = mean_squared_error(y_test,clf_choisi.predict(X_test))
                rmse_test = mean_squared_error(y_test,clf_choisi.predict(X_test),squared=False)
                mae_test = mean_absolute_error(y_test,clf_choisi.predict(X_test))
                list_test = [r2_test,mse_test,rmse_test,mae_test]
                metrics = ['R^2','MSE','RSME','MAE']
                df_scores = pd.DataFrame(zip(list_ent,list_test), index=metrics, columns=['Entraîment', 'Test'])
                st.dataframe(df_scores)
            with col3:
                st.write()
            st.markdown("\n")
            
            #L'importances des variables du modéle choisi
            col4, col5 = st.columns([1,2])
            with col4:
                feats_ = df.drop('comptage_horaire',axis=1)
                importances = list(clf_choisi.feature_importances_)
                feats = list(feats_.columns)
                coefs = pd.DataFrame({'valeur estimée': importances}, index = feats)
                st.markdown("\n")
                st.dataframe(coefs)
            with col5:
                def impPlot(imp, name):
                    figure = px.bar(imp, x=imp.values, y=imp.keys(), labels = {'x':'importance', 'index':'variables'},
                        text=np.round(imp.values, 2), title="L'importance des variable du "+ name, width=700, height=400)
                    figure.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    })
                    st.plotly_chart(figure)
                coeff = pd.Series(clf_choisi.feature_importances_, index=feats)
                impPlot(coeff,"XGBoost Regressor")

        st.markdown("""
        **2. REPRÉSENTATION GRAPHIQUE DES PRÉDICTIONS PAR VARIABLE EXPLICATIF**\n
        """)
        lista_var=list(df.columns)
        lista_var.pop(1)
        lista_var.pop(10)
        choix_var = st.selectbox('Choix de la variable',lista_var)
        st.markdown("\n")
        list1 = df_test[choix_var].to_numpy().tolist()
        dict1={choix_var: list1, 'predictions_xgb': clf_choisi.predict(X_test)}
        data_g= pd.DataFrame(dict1)
        if choix_var in ["id_compteur",'temperature','pluie']: rot=90
        else: rot=0
        if choix_var=="id_compteur": font_s=5
        else: font_s=10    
        fig = plt.figure(figsize=(10,5))
        sns.pointplot(x=choix_var, y='comptage_horaire', data=df_test, color='#659DBD',label='Comptage horaire réel')
        sns.pointplot(x=choix_var, y='predictions_xgb', data=data_g,color='orange', label='Prédictions')
        plt.title('Relation entre le comptage horaire et'+" "+ choix_var)
        plt.ylabel('Comptage horaire')
        plt.xlabel(choix_var)
        plt.xticks(rotation=rot, fontsize=font_s)
        plt.legend()
        st.pyplot(fig) 
        st.markdown("""
        **3. REPRÉSENTATION GRAPHIQUE DES PRÉDICTIONS PAR COMPTEUR PAR JOUR DE L'ANNÉE 2024**\n
        """)
        col1, col2,col3 = st.columns([2,1,1])
        list_compteurs.sort()
        with col1:
            choix_compteur = st.selectbox('Choix du compteur',list_compteurs)
        with col2:
            list_mois = ['janvier','février']
            choix_nommois = st.selectbox('Choix du mois',list_mois)
        with col3:
            list_jour = df_test['jour_comptage'].unique().tolist()
            choix_jour = st.selectbox('Choix du mois',list_jour)
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        import category_encoders as ce

        choix_idcompteur = get_id_compteur(df_compteurs,choix_compteur)
        if choix_nommois=='janvier': choix_mois=1
        else : choix_mois=2

        if choix_nommois =='février' and choix_jour in [26,27,28,29,30,31] :
            st.warning ("Le dataset ne contient que les données de 2024 jusqu'au 25 février.")
        else:
            # Définition des données de test (un seul compteur et une date)
            df_new = df[(df['id_compteur'] != choix_idcompteur) & (df["mois_comptage"] != choix_mois) & (df["jour_comptage"] != choix_jour)]
            df_test_new = df_test[(df_test['id_compteur'] == choix_idcompteur) & (df["mois_comptage"] == choix_mois) & (df["jour_comptage"] != choix_jour)]
            
            y_train = df_new['comptage_horaire']
            X_train = df_new.drop('comptage_horaire',axis=1)
            y_test = df_test_new['comptage_horaire']
            X_test = df_test_new.drop('comptage_horaire',axis=1)
            # On encode la varible cible
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)     
            #On encode la variable explicative'id_compteur'
            categorical_columns = ['id_compteur']
            loo_encoder = ce.leave_one_out.LeaveOneOutEncoder(cols=categorical_columns)
            X_train = loo_encoder.fit_transform(X_train, y_train)
            X_test = loo_encoder.transform(X_test)
            # On fait la normalisation de la dataset 
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.fit_transform(X_test)

            y_pred_test = clf_choisi.predict(X_test)

            #Creation d'un dataframe avec la variable "heure_comptage" et les predictions du dataset d'entraînement
            list1 = df_test_new['heure_comptage'].to_numpy().tolist()
            dict1={'heure_comptage': list1, 'predictions_xgb': y_pred_test}
            data_g= pd.DataFrame(dict1)

            #Visualisation des prédictions et du comptage horaire réel
            fig = plt.figure(figsize=(10,5))
            sns.lineplot(x='heure_comptage', y='comptage_horaire',  marker = 'o', data=df_test_new, color='#659DBD', label='Comptage horaire réel')
            sns.lineplot(x='heure_comptage', y='predictions_xgb',  marker = 'o', data=data_g, color='#FFD64B', label='Prédictions')
            plt.title("Compteur du " + choix_compteur + " de la journée du " + str(choix_jour) +" "+ choix_nommois + " 2024")
            plt.ylabel('Comptage horaire')
            plt.xlabel('Heure')
            plt.legend() 
            st.pyplot(fig) 

# Cinquième page : CONCLUSIONS
if page == pages[4]:
    st.markdown("<h3 style='text-align: center; color: #55562d;'>Conclusions</h3>", unsafe_allow_html=True)

    st.markdown("\n")
    st.markdown("""
    Lors de la période étudiée (Janvier 2023 - Février 2024), nous utiliserons comme indicateur du trafic le Comptage moyen/site/heure.\n
        
    **1. TOP/FLOP : UNE FORTE DISPARITÉ GÉOGRAPHIQUE**\n
    Le site du **Totem 73 boulevard de Sébastopol** arrive en tête du classement avec un cumul de <span style='text-decoration: underline;'>**281 vélos/heure en moyenne**</span>. 
    Le trafic y est 20 fois plus important qu’au **90 Rue de Sèvres**, le site le moins fréquenté, avec <span style='text-decoration: underline;'>**14 vélos/heure en moyenne**\n

    **2. DE FORTS CONSTRASTES CENTRE/PÉRIPHÉRIE ET RIVE DROITE/RIVE GAUCHE**\n
    Le trafic est <span style='text-decoration: underline;'>**plus intense dans le centre et rive droite, plus faible dans les quartiers périphériques et rive gauche**</span>. D’un côté des quartiers plus denses 
     en activités économiques, commerciales et lieux de sortie, avec des distances à parcourir plus courtes et un trafic routier congestionné. De l’autre des 
     quartiers plus résidentiels et plus éloignés des nœuds d’activité. Les axes Nord-Sud, Est-Ouest et les quais de Seine sont quant à eux des points de passage 
     obligés pour traverser Paris à vélo.\n
    """, unsafe_allow_html=True)
   
    st.markdown("""
    **3. QUEL FACTEURS INFLUENCENT LE TRAFIC À PARIS ?**\n
    Voici ce qu'il faut retenir sur l’évolution du trafic cycliste en fonction des facteurs périodiques, récurrents et exceptionnels.\n
  
    **a. Périodicité :** dissemblance semaine/week-end\n
    **- Du lundi au vendredi :** 2 pics aux heures de pointe (7-9h et 17-19h) qui disparaissent le week-end\n
    **- Week-end : -35,4 %** (**-30,65 %** le samedi, **-40,08 %** le dimanche vs du lundi au vendredi)
    """)
    st.markdown("\n")
    col1,col2, col3 = st.columns([1,4,1])
    with col1:
        st.write("")
    with col2:    
        st.image("Trafic_Velos_Jour_Sem.png")
    with col3:
        st.write("")
    st.markdown("\n")

    st.markdown("""
     **b. Facteurs réccurents et exceptionnels**\n
     **- Facteurs réccurents :** Comme vous pouvez le voir sur ce graphique, On observe une **baisse** du trafic pendant les jours fériés, également pendant les 
     vacances de Noel et à la fin de l'été. On remarque une **hausse** significaticative du trafic lorsqu'il fait beau temps.\n
     **- Facteurs exceptionnels :** On remarque que les grèves/manifestations influent le trafic vers la **hausse**.
        """)
    st.markdown("\n")
    col1,col2, col3 = st.columns([1,4,1])
    with col1:
        st.write("")
    with col2:    
        st.image("Evolution_Trafic_Velos.png", caption="Carte des événements de Jan23-Feb24")
    with col3:
        st.write("")
    st.markdown("\n")

    st.markdown("""
     **4. PRÉDICTION DU TRAFIC CYCLISTE À PARIS**\n
     **a. Variable Cible :** comptage_horaire comptage/heure/site\n
                
     **b. Les problèmes :**\n 
     - 16 variables à disposition, mais aucune véritable variable numérique. Elles sont toutes catégorielles et ont peu de relations linéaires avec la variable cible.\n
     - La existence des valeurs extrêmes ou aberrantes entre 201 et 8190 qui representent le **9,255 %** du dataset.\n
     - La existence de valeurs equal à 0 et 1 qui represente le **6,52 %** du dataset.\n
                
     **c. Les solutions :** 
     - Ajouter 5 variables explicatives : temperature, pluie, jour feriée, vacances scoilaires et greve. Et eliminer les variables liées à les photos du site de comptage.
     - Eliminer les valeurs supérieures à 1000. Mais conserver les valeurs equals à 0 et 1.
                
     **d. Méthodologie et  Algorithmes essayés:**\n 
     Regression Lineraire, Decision Tree, Random Forest et XGBoost Regressors.\n
                
     **e. Choix du modéle:**\n 
     **XGBoost Regressor** avec un **R^2 train de **95.58%** et R^2 test de 91.48%**.\n
                
     **f. Représentation graphique :**\n
     """)
    st.markdown("\n")
    col1,col2, col3 = st.columns([1,4,1])
    with col1:
        st.write("")
    with col2:    
        st.image("Modelisation.png")
    with col3:
        st.write("")
    st.markdown("\n")
    
    st.markdown("""
     **5. CONCLUSIONS DU MODÉLE**\n
      Bien que le modèle ait un bon score, lorsque nous avons effectué les représentations graphiques sur trois  compteurs différents (un compteur aléatoire, le deuxième avec 
     la plus grande affluence et le troisième avec la plus faible) pendant trois situations spéciales,  nous avons constaté que le modèle n’est pas performant:\n
     - Sur **les heures de pointe**, quand les valeurs sont extremes et aléatoires.\n
     - Sur **les compteurs avec les plus faible trafic** de vélos par heure par site.\n
     - Sur **les situations exceptionnelles** comme jour de neige or jour fériée.\n
      """)
    st.markdown("\n")
    st.info("Si le développement de notre projet vous intéresse, n'hésitez pas à nous contacter !")
    