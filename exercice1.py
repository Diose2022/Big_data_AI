"""
#################### EXERCICE1 ###############################
"""
#import des bibliotheque necessaire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du fichier csv avec la methode read_csv et on lui donne le chemin du fichier
data = pd.read_csv('winequality-red.csv', sep=';')

#Affichage des 5 premiers ligne on utilise la methode head(5) pour voir les 5 premiers ligne
print(data.head(5))


"""
#################### EXERCICE2 ###############################
"""#1 Affichage des statistique descriptiives du dataframe
print(data.describe())
"""
Nous avons 1599 valeur pour chaque observation
Moyenne: Nous pouvons constater que l'alcohol a la moyenne la plus eleve avec 10,42 
            suivi du "fixed acidity" qui a une moyenne de 5,63
            et la plus faible moyenne est celui de 0.527821

Ecart Type: On constate que l'ecart type le plus faible est "volatile acidity" avec 0.179060 
suivi de la qualite qui a une ecartype de 0.807569 puis viens alcohol avec 1.065668 puis enfin
fixed acidity qui a une ecart type de 1.741096

Moyenne: La moyenne la plus eleve est celui de "alcohol" avec 10.200000
puis viens "fixed acidity" avec 7.900000
la moyenne de la qualite est de : quality
celui de "volatile acidity" = 0.520000
"""


"""
#################### EXERCICE3 ###############################
"""

"""
#Question1:  pour l'identification des valeurs nulles dans le dataframe
on appel le nom de notre dataset(data) suivi de .isnull() qui nous affiche les valeur nulles
"""
valeur_null = data.isnull()
print(valeur_null)

"""
#Question2: Pour afficher le nombre total de valeur nulle par colone on appel le nom de notre
dataset et on lui affecte la methode isnull() suivi de .sum() et print() pour afficher le resultat
"""
nombre_valeur_null = data.isnull().sum()
print(nombre_valeur_null)

"""
Question3: Pour les valeurs null je propose la suppression pour eviter des erreur lors de l'entrainement de mon modele
"""
sup = data.isnull().dropna()
print(sup)

"""
#################### EXERCICE4 ###############################
"""
"""
Question1:Les notes de qualite son proche parce qu'il  tendent vers 1
"""

sns.countplot(x='quality', data=data)
plt.show()

"""
Question2: Pour visualiser les correlations entre les variable on utilise heatmap avec  .corr() ensuite on 
fait un plt.show() pour afficher le resultat
"""
sns.heatmap(data.corr(), annot=True, linecolor='white', cmap=plt.cm.Blues)
plt.show()

"""
Question3: evaluation la multicolinearite: on constate une multicolinéarité entre 'qualite' et 'volatile_acidy' qui tend vers 0.8
"""


#####entrainer un model de random Forest####

# Question4: on entraine les donnees en utilisant sklearn pour faire les test
from sklearn.model_selection import train_test_split
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
Entrainemt des donnees en utilisant sklearn
on import d'abord RandomForestRegressor
puis on entraine le modele avec la methde .fit() et on lui donne les 2 parametre
'''
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

#Entrainement des donnees avec random forest et on voit bien que
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Calcul des importances des caractéristiques
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Affichage des importances triées en utilisant print
print(feature_importances)


"""
 Alcohol (0.281362) est clairement la variable la plus importante, 
 car elle a la valeur la plus élevée. 
 Cela signifie que cette caractéristique 
 a le plus grand effet sur la prédiction du modèle.
 
Nous pouvons dire que le modele est tres influence par la colone alcohol
"""