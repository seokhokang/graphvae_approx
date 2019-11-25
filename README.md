# graphvae_approx
Tensorflow implementation of the model described in the paper [Efficient Learning of Non-Autoregressive Graph Variational Autoencoders for Molecular Graph Generation](https://link.springer.com/article/10.1186/s13321-019-0396-x)

## Components
- **preprocessing.py** - script for preprocessing data
- **train.py** - script for model training
- **test.py** - script for model evaluation (molecular graph generation)
- **GVAE.py** - model architecture

## Dependencies
- **Python**
- **TensorFlow**
- **RDKit**
- **NumPy**
- **scikit-learn**
- **sparse**

## Citation
```
@Article{Kwon2019,
  title={Efficient learning of non-autoregressive graph variational autoencoders for molecular graph generation},
  author={Kwon, Youngchun and Yoo, Jiho and Choi, Youn-Suk and Son, Won-Joon and Lee, Dongseon and Kang, Seokho},
  journal={Journal of Cheminformatics},
  volume={11},
  pages={70},
  year={2019},
  doi="10.1186/s13321-019-0396-x"
}
```
