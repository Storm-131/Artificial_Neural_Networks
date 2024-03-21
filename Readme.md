# Artificial Neural Networks

This is a sample project for using Pytorch or Pytorch Lightning as clean single-file-implementations to train a basic 
ANN model on MNIST dataset. 

To run the models, simply select one of the files (pytorch.py, lightning.py) and execute it.

```bash
python pytorch.py
python lightning.py
```

### Organisation of Files

```txt
📦ANN
 ┣ 📂data
 ┃ ┣ 📂MNIST    -> Will be downloaded by the code
 ┃ ┗ 📂samples  -> Sample images for custom prediction
 ┃ ┃ ┣ 📜2.png
 ┃ ┃ ┗ 📜5.png
 ┣ 📂utils
 ┃ ┗ 📜pred.py  -> Code for custom prediction
 ┣ 📜Readme.md
 ┣ 📜lightning.py   -> Pytorch Lightning implementation
 ┗ 📜pytorch.py     -> Pytorch implementation
```
---
