# IMPORT
import matplotlib.pyplot as plt
import pandas as pd

"""
# Classe permettant de génerer 4 graphiques de suivit de métriques durant l'entrainement d'un modèle
# Train accuracy, Train loss, Validation accuracy, Validation loss
"""

def displayGraph(pathLog,pathSaveGraph):
    """
    # Fonction permettant de creer nos graph de suivi de metriques
    :param pathLog: chemin du CSV contenant nos metrics
    :param pathSaveGraph: chemin de destination pour sauvegarder nos 4 graphiques en jpg
    """

    data = pd.read_csv(pathLog, sep=',')
    print(data)
    # split into input (X) and output (Y) variables
    plot(data['epoch'], data['dice_coef_loss'], data['val_dice_coef_loss'], 'Accuracy metrics', 'Epoch', 'Accuracy', 'upper left',pathSaveGraph)
    #plot(data['epoch'], data['binary_accuracy'], 'Accuracy metrics', 'Epoch', 'Accuracy', 'upper left',pathSaveGraph)
    plot(data['epoch'], data['loss'], data['val_loss'], 'Loss metrics', 'Epoch', 'Loss', 'upper left',pathSaveGraph)
    #plot(data['epoch'], data['loss'], 'Loss metrics', 'Epoch', 'Loss', 'upper left',pathSaveGraph)


def plot(X, Y, Y2, title, xLabel, yLabel, legendLoc, pathSaveGraph):
#def plot(X, Y, title, xLabel, yLabel, legendLoc, pathSaveGraph):
    """
    # Fonction d'affichage de graph
    :param X: correspond au nombre d'époch
    :param Y: correspond a la courbe accuracy
    :param Y2: correspond a la courbe loss
    :param title: titre du graphique
    :param xLabel: label des abcisses
    :param yLabel: label des ordonnees
    :param legendLoc: legende
    :param pathSaveGraph: chemin de sauvegarde pour les graphiques
    """

   #On trace nos differentes courbes
    plt.plot(Y)
    plt.plot(Y2)
   #titre du graph, legende...
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(['train', 'val'], loc=legendLoc)
   #Pour avoir un courbe propre qui demarre à 0
    plt.xlim(xmin=0.0, xmax=max(X))
    plt.savefig(pathSaveGraph +'\\' + title)
    plt.figure()
    #plt.show()


def main():
    """
    # Fonction main
    """

    #Definition des chemins d'acces a notre fichier log
    pathLogs = 'result/log/metric/metrics.csv'
    pathSaveGraph = 'result/log/graph'
    displayGraph(pathLogs,pathSaveGraph)


if __name__ == "__main__":
    """
    # MAIN
    """
    main()