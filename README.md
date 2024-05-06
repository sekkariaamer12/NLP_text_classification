# NLP_text_classification

# Préparation des données :

Le dataset est divisé en trois jeux de données :
- **Le dataset train pour l’apprentissage**
- **Le dataset validation pour mesurer la capacité de généralisation du réseau**
- **Le dataset test pour vérifier que la généralisation des performances mesurées**

Dans un premier temps, on sépare les données fournies selon deux catégories : les phrases à analyser et les labels.
On coupe les phrases faisant plus de 10 mots afin qu’elles ne fassent pas trop de caractères et on ajoute des caractères vides pour aux phrases trop courtes pour qu’elles fassent toutes la même taille pour passer dans le réseau de neurones.
Après avoir réalisé une telle opération, on crée une correspondance entre les mots et un identifiant. Pour cela, on utilise la bibliothèque torchtext.vocab.
De plus, on utilise un encodage onehot qui va permettre de transformer l’identifiant du mot en tensor faisant la taille du nombre de mots présent dans le vocabulaire et qui possède un 1 pour l’identifiant du mot et des 0 dans le reste des indexes.

# Entrainement et Evaluation du modèle :

On utilise un DataLoader afin d’alimenter notre réseau de neurones mot par mot.
On réalise une phase d’apprentissage que l’on valide à chaque epochs pour vérifier que l’on n’a pas de overfitting.

Dans un premier temps, on a trouvé un problème d'overfitting pour cela on a modifier les hyper paramètre :
- **Le taux d'apprentissage**
- **Le nombre de mots par phrase**

# Test du modèle :
On a utilisé les données test pour tester notre modèle sur des nouvelles données. On a mis en place une matrice de confusion pour voir les performances de notre modèle.
