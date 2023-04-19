import pandas as pd
import numpy as np
import missingno as msno



##############################################################################
#  initializePandas() :
#         Aucun paramètres
#
#         Initialise les options pandas
# 
#         Return : None
##############################################################################

def initializePandas() :
    pd.set_option('display.max_columns', 10)  # or 1000
    pd.set_option('display.max_rows', 100)  # or 1000
    pd.set_option('display.max_colwidth', 30)  # or 199
    return None
    
    
##############################################################################
#  compareColumns(df, L) :
#         df : pd.dataframe
#         L : liste de string de noms de colomnes de data
#
#         Affiche le nombre de valeurs présente dans une colonnes et absentes dans l'autre
#          
#
#         Return : None
##############################################################################

def compareColumns(df, L) :
    for e1 in L :
        for e2 in L:
            if e1 != e2 :
                try :
                    mask = df[e1].notna()
                    print(f'il y a {df[mask][e2].isna().sum()} valeurs dans {e1} qui sont manquantes dans {e2}.')
                except KeyError :
                    print(f"Erreur de clé, couple {e1} - {e2} non traité.")
            else :
                pass
    return None

##############################################################################
#  missingValuesInfos(df) :
#         df : pd.dataframe
#
#         Affiche le nombre de valeurs manquantes, totales, le taux de remplissage et la shape du dataframe
#         Affiche la msno.matrix du dataframe          
#
#         Return : None
##############################################################################

def missingValuesInfos(df) :
    nbRows, nbCols = df.shape
    print(f"Il y a {df.isna().sum().sum()} valeurs manquantes sur {nbRows * nbCols} valeurs totales.")
    print(f"Le taux de remplissage est de : {int(((nbRows*nbCols - df.isna().sum().sum())/(nbRows*nbCols))*10000)/100} %")
    print("Dimension du dataframe :",df.shape)
    msno.matrix(df)
    return None

##############################################################################
#  missingValuesInfos(df) :
#         df : pd.dataframe
#
#         A RENSEIGNER (et a comprendre)
#
#         Return : None
##############################################################################

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """Display correlation circles, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig, ax = plt.subplots(figsize=(10,10))

            # Determine the limits of the chart
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Add arrows
            # If there are more than 30 arrows, we do not display the triangle at the end
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Display variable names
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # Display circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Define the limits of the chart
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Display grid lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))
            plt.show(block=False)