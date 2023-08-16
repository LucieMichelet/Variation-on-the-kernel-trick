# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:04:16 2022

@author: Utilisateur
"""

import mabiblio as acp
import numpy as np
import pandas as pd
import os
import cv2

# %% Chargement données iris

# On charge les données
Ebrut = np.genfromtxt("iris.csv", dtype=str, delimiter=',')
labelscolonne = Ebrut[0, :-1]
labelsligne = Ebrut[1:, -1]
E = Ebrut[1:, :-1].astype('float')

# %% Chargement données mnist

# données entrainement
train_data = pd.read_csv("mnist_train.csv", delimiter=",")
E = train_data.to_numpy()

# %% Test ACP

#run la cellule iris ou mnist_train avant celle-ci en fonction de celle que l'on veut éudier

ACPvals, q = acp.ACP(E, 2)
acp.ACP_plot(ACPvals, q, "ACP_normal")

# %% Test ACPpond

d = E.shape[1]                           #Création de W 
W = np.eye(d)

ACPpondvals, q = acp.ACPpond(E, 0, W)    #On calcule les valeurs de notre ACP et le qkaiser avec pondération
acp.ACP_plot(ACPvals, q, "ACP_Pondérée") #On affiche l'ACP pondérée

W = np.eye(d)                            #Différence notable lors d'un changement de coefficient  
W[0, 0] = 9

ACPpondvals, q = acp.ACPpond(E, 0, W)
acp.ACP_plot(ACPvals, q, "ACP_Pondérée")

# %% Choix des points 

#!!! n'oubliez pas d'indiquer le chemin dans lequel se trouve les photos nécessaires pour appliquer les fonctions
#ou mettez les photos dans le même dossier que le code
# transformez Lena.png en Lena.jpg 

L1 = []
Image = []

for file in os.listdir("./"): #récupère toutes les fichier du dossier
    if file.endswith(".jpg"):   #selectionnes les fichier jpg
        img = cv2.imread(file)
        Image.append(file)
        L = acp.choixpts(img, 10000) #selectionne dix milles points sur l'image
        L1.append(L) #les auvegarde dans une liste

# %% Application ACP normale

for i in range(len(L1)):
    img = cv2.imread(Image[i])  #Lecture de l'image
    data_img = acp.data_pixels(img, L1[i])  #On génère la matrice data_img contenant l'intensité des points
    nfile = Image[i][:-4]
    ACP_vals, qkaiser = acp.ACPimg(data_img)    #On calcule les valeurs de notre ACP et le qkaiser
    acp.ACP_plot(ACP_vals, 2, "ACP_normal") #On plot l'ACP 
    
    if qkaiser > 1:
        Kmoy = acp.Kmoyimg(data_img, qkaiser)
        imgmasqueponctuel = acp.Masque(Kmoy, img)
        imgmasqueponctuel = np.uint8(imgmasqueponctuel)
        imgpaint = acp.RemplissageMasque(imgmasqueponctuel)
        mask = "{}_mask.png".format(nfile)
        paint = "{}_paint.png".format(nfile)
        cv2.imwrite(mask, imgmasqueponctuel)
        cv2.imwrite(paint, imgpaint)

# %% Application avec Covariance pondérée

for i in range(len(L1)):
    img = cv2.imread(Image[i])   #Lecture de l'image
    data_img = acp.data_pixels(img, L1[i])   #on génère la matrice data_img contenant l'intensité des points
    d = data_img.shape[1]    #nombre de colonnes de data_img
    W = np.eye(d)   #matrice identité de taille d
    W[0, 0] = 10    #premier terme de W = 10
    W[1,1] = 1      #deuxième terme sur la diagonale de W = 1
    nfile = Image[i][:-4] #récupère toutes les lignes de toutes les colonnes sauf les 4 dernières
    ACP_vals_pond, qkaiser = acp.ACPimg_pond(data_img, W)    #On calcule les valeurs de notre ACP et le qkaiser avec pondération
    acp.ACP_plot(ACP_vals, 2, "ACP_pondérée")  #affichage de l'ACP pondérée
    
    if qkaiser > 1:         #on exclu le cas particulier ou qkaiser = 1
        Kmoy = acp.Kmoyimg(data_img, qkaiser)
        imgmasqueponctuel = acp.Masque(Kmoy, img)
        imgmasqueponctuel = np.uint8(imgmasqueponctuel)
        imgmasque = acp.RemplissageMasque(imgmasqueponctuel)
        imgpaint = acp.RemplissageMasque(imgmasqueponctuel)
        mask = "{}_mask_pond.png".format(nfile)
        paint = "{}_paint_pond.png".format(nfile)
        cv2.imwrite(mask, imgmasqueponctuel)
        cv2.imwrite(paint, imgpaint)