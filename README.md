# Tooth Decay Semantic Segmentation

## Description
Logiciel de détection, localisation & segmentation de carries sur des radiographies dentaires.  
  
Basé sur le réseau de neurones U-Net++.  

L'article complet du projet est disponible sur :  
https://deeplylearning.fr/cours-pratiques-deep-learning/segmentation-semantique-dimages/

## Informations

### Dataset original
Les images utilisées sont en noir et blanc (channel=1)

### Data augmentation

### Annotation
Nous avons utilisé la version gratuite de SuperAnnotate pour labeliser nos données.
Nous traçons pour chaque carrie présente sur une radio un polygone.

## Installation
Testé seulement sur Python 3.6.12

1. Cloner le répo :  
`$ git clone https://github.com/Momotoculteur/Tooth-decay-semantic-segmentation.git`

2. Installer les modules externes :  
`pip install -r requirements.txt`
   
## Utilisation
### Répo architecture
```
.
├── data                
│   └── img
│        └── mask                 # Mask associé aux radiographies
│        └── ori                  # Radiographie originale
│   └── label           
│        └── annotations.json     # Fichier contenant les annonations des images originale
│        └── classes.json         # Contient la définition des classes
│        └── config.json          # Fichier de configuration de SuperAnnotate
├── masksMaker.py                 # Script permettant de générer les masques des images originales a partir du fichier annotations.json
├── datasetLoader.py              # Generateur custom pour charger, transformer et envoyer au DNN
├── datasetAugmenter.py           # Scripts de data augmentation 
├── segmentation_models/..        # Contient divers modèles de réseaux de neurones 
├── train.py                      # Script permettant d'entrainer le réseau
├── utils.py                      # Contient diverses fonction d'aide à l'entrainement/visualization/etc.
```


### Création des masques
Pour chaque image du dataset, nous récupérons les polygones représentant les carries via le fichier annotations.json.
Les masques générés sont format PNG, évitant toute compression et perte de données.

### Augmentation des données
Requis : les masques des images originales doivent déjà être généré.
Pour garantir de ne pas changer les informations contenues dans les images originales, nous n'utilisons que des déformations non
destructives :
- flip
- rotation 90°
- transpose

### Entrainement

### Prédiction

## Remerciements
```
@article{zhou2019unetplusplus,
  title={UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation},
  author={Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  journal={IEEE Transactions on Medical Imaging},
  year={2019},
  publisher={IEEE}
}
@incollection{zhou2018unetplusplus,
  title={Unet++: A Nested U-Net Architecture for Medical Image Segmentation},
  author={Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  booktitle={Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support},
  pages={3--11},
  year={2018},
  publisher={Springer}
}
@misc{Yakubovskiy:2019,
  Author = {Pavel Yakubovskiy},
  Title = {Segmentation Models},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models}}
}
@misc{ronneberger2015unet,
      title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, 
      author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
      year={2015},
      eprint={1505.04597},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Licence
Copyright (c) 2020 Bastien MAURICE & Dr.Van-Hoan NGUYEN

This project is licensed under the terms of the MIT [license](LICENSE).