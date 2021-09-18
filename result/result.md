|   | baseDS | dataAugPerImg | augmentedDS | transformation | metric | metricVal | loss   | lossVal  | maxEpoch | res  | backBone | pre-weight | opti | Lr   | earlyStop            | redLr                 | Notes|
|---|--------|---------------|-------------|----------------|--------|-----------|--------|----------|----------|------|----------|------------|------|------|----------------------|-----------------------|------|
| 1. | 1089   | /             | /           | /              | mIOU   | 0.495299  | lovasz | 2.324899 | 2        | unet | resnet50 | imageNet   | Adam | 0.01 | pat:5 // minDelta: 0 | pat:3 // minLr:0.0001 |  Gradient qui explose ? Cause mIoU ?    |
| 2. | 1089   | /             | /           | /              | Dice   | 0.9930    |FocalTverskyLoss|0.7093| 11   | unet | resnet50 | imageNet   | Adam | 0.01 | pat:5 // minDelta: 0 | pat:3 // minLr:0.0001 |       |
| 3. | 1089   | /             | /           | /              | Dice   | 0.9920    |FocalTverskyLoss |0.8056|  6  | unet | resnet50 | None   | Adam | 0.01 | pat:5 // minDelta: 0 | pat:3 // minLr:0.0001 |       |
| 4. | 1089   |5|6534|C[Flip(0.5)/Transpose(0.5)/RRotate90(0.5)]|Dice|0.9955     |FocalTverskyLoss | 1.0000| 10   | unet | resnet50 | None   | Adam | 0.01 | pat:5 // minDelta: 0 | pat:3 // minLr:0.0001 |       |
| 5. | 1089   |5|6534|C[Flip(0.5)/Transpose(0.5)/RRotate90(0.5)]|Dice| 0.99568   |FocalTverskyLoss |0.8485 | 15   | unet | resnet50 | imageNet   | Adam | 0.01 | pat:5 // minDelta: 0 | pat:3 // minLr:0.0001 |       |
| 6. | 1089   |5|6534|C[Flip(0.5)/Transpose(0.5)/RRotate90(0.5)]|Dice | 0.9958   |FocalTverskyLoss |0.55126 | 30   | unet | resnet50 | imageNet   | Adam | 0.01 | pat:5 // minDelta: 0 | pat:3 // minLr:0.0001 |  callback loss based (before Dice)     |
| 7. | 1089   |5|6534|C[Flip(0.5)/Transpose(0.5)/RRotate90(0.5)]|Dice | 0.9958   |FocalTverskyLoss |0.5211 | 35   | unet | resnet18 | imageNet   | Adam | 0.01 | pat:5 // minDelta: 0 | pat:3 // minLr:0.0001 |  callback loss based (before Dice)     |



# A TEST HYPER PARAM

**x5 augmented:**
dice, iou ?
focal, lovazs ?
Lr 0.01, 0.001 ?
unet, unet++ ?
BB resnet34, resnet50,....
!!!!!! loss comme metric ????



# OBSERVATIONS

++
image net base
data aug

Overfit de ouf ?
diminuer complexité du model OU + de data ( more data aug transformation type ?)



# TO DO
ROC / AUC sur data test => faire les fonctions sur *predictTestDataset.py*
Refaire *datasetExplorerInfos.py* avec dataset train 1.1 & dataset test 1.1 ( /!\ SANS DAT AUG )

Data aug ~~ 20K de photos 
Script data aug: ajouter options d'augmenter QUE photos avec carry
Plus de transformation non rigid
