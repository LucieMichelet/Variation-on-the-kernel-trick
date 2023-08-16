# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:37:55 2022

@author: Utilisateur
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import seaborn as sns

# %% Partie 1 Centrage-réduction et ACP



def esperance(Xi):
    """
    fonction qui calcul un estimateur sans biais de l'esperence d'un vecteur Xi
    :param Xi: un vecteur de taille (m,)
    :return: l'esperence du vecteur Xi
    """

    m = Xi.shape[0]
    return np.sum(Xi) / m


def variance(Xi):
    """
    fonction qui calcul un estimateur avec un biais asymptotique de la variance d'un vecteur Xi
    :param Xi: un vecteur de taille (m,)
    :return: la variance du vecteur Xi
    """

    m = Xi.shape[0]
    Xi_bar = esperance(Xi)
    return np.sum((Xi - Xi_bar)**2) / (m - 1)


def centre_red(R):
    """
    Permet de créer la matrice centrée-réduite à partir d'un tableau de données'

    Parameters
    ----------
    R : TYPE np.array
        DESCRIPTION. Tableau de données qu'on veut étudier

    Returns
    -------
    Rcr : TYPE np.array
        DESCRIPTION. Tableau de la même taille que R avec des valeurs centrée-réduites

    """
    Rcr = np.zeros(R.shape) #On crée une matrice vide de la taille de nos données que l'on va remplir
    indg = np.mean(R, axis=0)   #On calcule la somme des lignes divisé par le nombre de lignes
    for i in range(R.shape[0]): #On soustrait la ligne "moyenne" calculée à la ligne du dessus à  chaque ligne de notre tableau 
        Rcr[i, :] = R[i, :]-indg
    Rc = Rcr.copy() #On garde la matrice centré en mémoire
    for i in range(np.shape(R)[1]): #On divise chaque colonne par sa norme 
        colonne = Rcr[:, i]
        colonne_r = np.linalg.norm(colonne)
        if colonne_r != 0:
            Rcr[:, i] = colonne/colonne_r
    return Rcr, Rc


def ACP(X, q):
    """
    Permet de calculer une projection de nos données 

    Parameters
    ----------
    X : TYPE np.array
        DESCRIPTION. Tableau de données qu'on veut étudier
    q : TYPE int
        DESCRIPTION. The default is 0. Le nombre de directions principales à prendre en compte car elle contiennent de l'information
    
    Returns
    -------
    Xq : TYPE np.array 
        DESCRIPTION. Matrice de la projection de nos données 
    qkaiser : TYPE int
        DESCRIPTION. nombre de directions principales obtenues avec la règle de kaiser
        
    """
    Xcr, Xc = centre_red(X) #On calcule nos matrices centrée et centrée-réduite
    C = Xcr.T@Xcr #On calcule la matrice des covariance
    U, D, Ut = np.linalg.svd(C) #On applique la SVD sur notre matrice de covariance
    d = Xcr.shape[1]
    qkaiser = len(D[D >= sum(D)/d]) #On applique la règle de kaiser (cf énoncé)
    Xq = Xcr@(U[:d, :q]) #On calcule notre projection selon la matrice orthognal Uq
    return Xq, qkaiser


def CovPond(Xcr, W):
    """
    Génère une matrice de covariance à partir d'une matrice W permettant de donner plus d'importance
    à une information que l'on veut faire ressortir (ex : les couleurs, la position des pixels, ...)'

    Parameters
    ----------
    Xcr : TYPE np.array
        DESCRIPTION. Tableau de données centré-réduit qu'on veut étudier
    W : TYPE np.array
        DESCRIPTION. Matrice diagonale dont certains coefficients priment sur d'autres pour cibler l'analyse

    Returns
    -------
    Cov : TYPE np.array
        DESCRIPTION. Matrice de covariance

    """
    raceW = W**0.5 #Les coefficients sont à la racine !!! faire attention à avoir que des coefficients positifs
    Cov = (1/(np.sum(W)))*(Xcr@raceW).T@Xcr@raceW
    return Cov


def ACPpond(X, q, W):
    """
    Permet de calculer une projection de nos données avec pondération

    Parameters
    ----------
    X : TYPE np.array
        DESCRIPTION. Tableau de données qu'on veut étudier
    q : TYPE int
        DESCRIPTION. The default is 0. Le nombre de directions principales à prendre en compte car elle contiennent de l'information
        
    Returns
    -------
    Xq : TYPE np.array 
        DESCRIPTION. Matrice de la projection de nos données 
    qkaiser : TYPE int
        DESCRIPTION. nombre de directions principales obtenues avec la règle de kaiser
        
    """
    Xcr, Xc = centre_red(X) #Même principe que pour l'ACP normale 
    C = CovPond(Xcr, W)
    U, D, Ut = np.linalg.svd(C)
    d = Xcr.shape[1]
    qkaiser = len(D[D > sum(D)/d])
    Xq = Xcr@(U[:d, :q])
    return Xq, qkaiser


def ACP_plot(Xq, q, title):
    """
    Permet de tracer la projection de nos données avec ou sans pondération

    Parameters
    ----------
    X : TYPE np.array
        DESCRIPTION. Tableau de données qu'on veut étudier
    q : TYPE int
        DESCRIPTION. The default is 0. Le nombre de directions principales à prendre en compte car elle contiennent de l'information
    title : TYPE str
        DESCRIPTION. Donne un titre au plot (permet de savoir de quel type d'ACP il s'agit)

    Returns
    -------
    None.

    """
    if q >= 2:
        Xq = Xq[:, :2]  #On ne sélectionne que les 2 dimensions principales 
        plt.title(title)    #On donne un titre à notre code
        plt.scatter(Xq[:, 0], Xq[:, 1]) #On trace nos directions principales
        plt.show()  #On affiche le plot

# %% Partie 2 Algorithme des K-moyens

def Kmoy2(A, k, epsilon):
    """
    Permet de calculer et créer des catégories à partir de données

    Parameters
    ----------
    A : TYPE np.array
        DESCRIPTION. Tableau numpy ayant le même nombre de lignes que R et k colonnes
    k : TYPE int
        DESCRIPTION. Le nombre de directions principales à prendre en compte car elle contiennent de l'information
    epsilon : TYPE float
        DESCRIPTION. La précision

    Returns
    -------
    TYPE tuple
        DESCRIPTION. renvoie la matrice composée des barycentres et leurs catégories'


    """
    if len(A) <= 2: #cas particulier à éviter
        print("Ensemble trop petit")
        return False, False
    
    else:
        ligne, colonne = np.shape(A)    #On récupère le nombre de lignes et de colonnes
        Aavecind = np.concatenate((np.arange(0, ligne, 1)[np.newaxis].T, A), axis=1)    #On ajoute une première colonne à A contenant les indices
        LL = [[] for x in range(0, k)]  #On créer des listes pour stocker les lignes et les indices
        LLind = [[] for x in range(0, k)]
        ind = np.random.randint(0, ligne, k)
        testunicite = len(ind) == len(np.unique(ind))   #On regarde si c'est le seul point

        mu = A[ind]  #on définie la matrice muu qui contient les coordonnées
        cent = np.zeros(np.shape(mu))

        normtest = np.zeros((ligne, k))
        l = 0
        compteur = 0
        
        if testunicite == True:  #si la condition test unicité est vérifiée
            while np.linalg.norm(cent-mu) > epsilon:
                cent = copy.deepcopy(mu)
                LL = [[] for x in range(0, k)]
                LLind = [[] for x in range(0, k)]
                
                for r in range(0, k):
                    for i in range(0, ligne):
                        normtest[i, r] = np.linalg.norm(A[i, 2:]-mu[r, 2:]) #On calcule la norme entre ce point et les autres
                
                for i in range(0, ligne):
                    l = np.where(normtest[i, :] == np.amin(normtest[i, :]))[0][0]   #On regarde à quel indice est reliée chaque norme minimale
                    LL[l].append(A[i, :])
                    LLind[l].append((Aavecind[i, 0]).astype('int'))
                
                for r in range(0, k):
                    s = 0
                    for j in range(0, len(LL[r])):
                        s = s+LL[r][j]
                    mu[r] = (1/len(LL[r]))*s
                compteur = compteur+1
        
        else:
            Kmoy2(A, k, epsilon)
        S2 = np.insert(np.array(LL[0]), np.array(LL[0]).shape[1], 1, axis=1) #On crée nos blocks pour contenir la dernière colonne avec le numéro de la catégorie
        S3 = np.insert(np.array(LL[1]), np.array(LL[1]).shape[1], 2, axis=1)
        S2 = np.block([[S2],[S3]])
        
        for i in range(2, len(LL)): #Boucle dans le cas où il y a plus de 2 catégories
            bloc = np.insert(np.array(LL[i]), np.array( LL[i]).shape[1], (i+1), axis=1)
            S2 = np.block([[S2],[bloc]])
            
        return S2

# %% Partie 3


def choixpts(img, nbpts):
    """
    Permet de choisir une liste de coordonnées sur une image

    Parameters
    ----------
    img : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels
    nbpts : TYPE int
        DESCRIPTION. nombre de points à choisir sur l'image'

    Returns
    -------
    L : TYPE list 
        DESCRIPTION. Liste des coordonnées

    """
    L = []
    while len(L) < nbpts:   #Boucle while pour être sûr de prendre des points différents/optimisation possible avec .pop en sortant les valeurs déjà utilisées
        couple = (int(np.random.randint(low=1, high=img.shape[0])-2), int(np.random.randint(low=1, high=img.shape[1])-2))   #On génère le couple d'indices
        if couple not in L:  #on intègre dans L chaque couple d'indice sans doublon
            L.append(couple)
    return L


def Moyenne_pixel(img, coord_x, coord_y, couche):
    """
    Fonction peu élégante permettant de calculer la moyenne d'intensité d'une couleur pour un pixel

    Parameters
    ----------
    img : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels
    coord_x : TYPE int 
        DESCRIPTION. La coordonnée en x
    coord_y : TYPE int
        DESCRIPTION. La coordonnée en y
    couche : TYPE int 
        DESCRIPTION. la couche rouge, verte ou bleu 

    Returns
    -------
    TYPE float
        DESCRIPTION. Moyenne d'intensité d'un couleur pour un pixel en prenant les pixels autour de lui

    """
    p_hg = img[coord_x-1, coord_y+1, couche]    #On prends la valeur de chaque pixel
    p_hm = img[coord_x, coord_y+1, couche]
    p_hd = img[coord_x+1, coord_y+1, couche]
    p_cg = img[coord_x-1, coord_y, couche]
    p_cm = img[coord_x, coord_y, couche]
    p_cd = img[coord_x+1, coord_y, couche]
    p_bg = img[coord_x-1, coord_y-1, couche]
    p_bm = img[coord_x, coord_y-1, couche]
    p_bd = img[coord_x+1, coord_y-1, couche]
    sum_h = int(p_hg) + int(p_hm) + int(p_hd)
    sum_c = int(p_cg) + int(p_cm) + int(p_cd)
    sum_b = int(p_bg) + int(p_bm) + int(p_bd)
    return (sum_h + sum_c + sum_b)/9 #On fait la moyenne


def data_pixels(img, L):
    """
    Permet de générer une matrice contenant les données sur les intensités de pixels

    Parameters
    ----------
    img : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels
    L : TYPE list 
        DESCRIPTION. Liste des coordonnées

    Returns
    -------
    data_img : TYPE np.array
        DESCRIPTION. Tableau à 2 dimensions contenant les intensités et instensités moyennes des pixels choisis aléatoirement

    """
    data_img = np.zeros((len(L), 8))    #On crée notre matrice data_img
    for i in range(len(L)):
        C = np.zeros(8)
        C[0], C[1] = L[i][0], L[i][1]   #On associe les coordonnées
        C[2], C[3], C[4] = img[L[i][0], L[i][1], 0], img[L[i][0], L[i][1], 1], img[L[i][0], L[i][1], 2] #On associe les intensités des pixels
        C[5] = Moyenne_pixel(img, L[i][0], L[i][1], 0) #On associe les intensités moyenne des pixels
        C[6] = Moyenne_pixel(img, L[i][0], L[i][1], 1)
        C[7] = Moyenne_pixel(img, L[i][0], L[i][1], 2)
        data_img[i, :] = C #on ajoute la ligne
    return data_img

def ACPimg(data_img):
    """
    Permet de calculer une projection de nos données issues d'une image'

    Parameters
    ----------
    data_img : TYPE np.array
        DESCRIPTION. Tableau à 2 dimensions contenant les intensités et instensités moyennes des pixels choisis aléatoirement

    Returns
    -------
    Xq : TYPE np.array 
        DESCRIPTION. Matrice de la projection de nos données 
    qkaiser : TYPE int
        DESCRIPTION. nombre de directions principales obtenues avec la règle de kaiser

    """
    return ACP(data_img, 2)


def ACPimg_pond(data_img, W):
    """
    Permet de calculer une projection pondérée de nos données issues d'une image'

    Parameters
    ----------
    data_img : TYPE np.array
        DESCRIPTION. Tableau à 2 dimensions pondérée contenant les intensités et instensités moyennes des pixels choisis aléatoirement

    Returns
    -------
    Xq : TYPE np.array 
        DESCRIPTION. Matrice pondérée de la projection de nos données 
    qkaiser : TYPE int
        DESCRIPTION. nombre de directions principales obtenues avec la règle de kaiser

    """
    return ACPpond(data_img, 2, W)


def Kmoyimg(data_img, qkaiser):
    """
    Permet de calculer et créer des catégories à partir de données

    Parameters
    ----------
    data_img : TYPE np.array
        DESCRIPTION. Tableau à 2 dimensions pondérée contenant les intensités et instensités moyennes des pixels choisis aléatoirement
    qkaiser : TYPE int
        DESCRIPTION. Le nombre de directions principales à prendre en compte car elle contiennent de l'information

    Returns
    -------
    TYPE tuple
        DESCRIPTION. renvoie la matrice composée des barycentres et leurs catégories'


    """
    return Kmoy2(data_img, qkaiser, 1*10**-16)


def Masque(S, img):
    """
    Permet de créer un masque en coloriant les pixels d'une certaine couleur en fonction de leurs catégories' 

    Parameters
    ----------
    S : TYPE np.array
        DESCRIPTION. Tableau par blocs contenant les catégories et les pixels 
   img : TYPE np.array
       DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels

    Returns
    -------
    imgmasqueponctuel : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels

    """
    N = int(S[-1, -1])  #On récupère le nombre de catégorie
    palette = sns.color_palette(None, N)    #Création de N couleurs différentes
    imgmasqueponctuel = np.zeros(img.shape)
    for i in range(S.shape[0]):
        imgmasqueponctuel[int(S[i][0]), int(S[i][1]),0] = palette[int(S[i, -1])-1][0]*255 #On colorie chaque pixels de la couleur de sa catégorie
        imgmasqueponctuel[int(S[i][0]), int(S[i][1]),1] = palette[int(S[i, -1])-1][1]*255
        imgmasqueponctuel[int(S[i][0]), int(S[i][1]),2] = palette[int(S[i, -1])-1][2]*255
    return imgmasqueponctuel


def RemplissageMasque(imgmasqueponctuel):
    """
    algorithme d'impainting permet de "peindre" à partir d'un masque'

    Parameters
    ----------
    imgmasqueponctuel : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels

    Returns
    -------
    imgmasqueponctuel : TYPE np.array
        DESCRIPTION. Tableau à 3 dimensions contenant les intensités des pixels

    """
    mask = cv2.cvtColor(imgmasqueponctuel, cv2.COLOR_BGR2GRAY)  #Conversion en niveau de gris
    mask[mask > 0] = 255
    mask = 255*np.ones(np.shape(mask))-mask #on applique le masque
    imgmasque = cv2.inpaint(imgmasqueponctuel, np.uint8(mask), 3, cv2.INPAINT_NS)
    return imgmasque
 