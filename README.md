# Deep Learning with Catalyst [![Stepik](https://img.shields.io/badge/DLS-Stepik-success)](https://stepik.org/course/83344/syllabus) [![Slack](https://img.shields.io/badge/Catalyst-slack-success)](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw)

[![dls-catalyst-course](https://github.com/catalyst-team/catalyst-pics/blob/master/pics/catalyst-dl-course-poster-eng.png)](https://github.com/catalyst-team/dl-course)

This is an open deep learning course made by [Deep Learning School](https://dlschool.org), [Tinkoff](https://tinkoff.ru), and [Catalyst team](https://github.com/catalyst-team). 
Lectures and practice notebooks located in ```./week*``` folders. Homeworks are in ```./homework*``` folders.

> *Note: the course is under update: 
> weeks with colab barge are ready to go, weeks with [WIP] label are still in progress. 
> You could use the `v20.12` branch for the earlier version of the full course.*

## Syllabus

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/dl-course/blob/master/week-01/seminar.ipynb) week 1: Deep learning intro
  - Deep learning – introduction, backpropagation algorithm. Optimization methods.
  - Neural Network in numpy.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/dl-course/blob/master/week-02/seminar.ipynb) week 2: Deep learning frameworks
  - Regularization methods and deep learning frameworks.
  - Pytorch basics & extras.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/dl-course/blob/master/week-03/seminar.ipynb) week 3: Convolutional Neural Network
  - CNN. Model Zoo.
  - Convolutional kernels. ResNet. Simple Noise Attack.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/dl-course/blob/master/week-04/seminar_done.ipynb) week 4: Object Detection, Image Segmentation*
  - Object Detection. (One, Two)-Stage methods. Anchors.
  - Image Segmentation. Up-scaling. FCN, U-net, FPN. DeepMask.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/dl-course/blob/master/week-05/seminar_done.ipynb) week 5: Metric Learning*
  - Metric Learning. Contrastive and Triplet Loss. Samplers.
  - Cross Entropy Loss modifications. SphereFace, CosFace, ArcFace.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/dl-course/blob/master/week-06/seminar_done.ipynb) week 6: Autoencoders*
  - AutoEncoders. Denoise, Sparse, Variational.
  - Generative Models. Autoregressive models.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catalyst-team/dl-course/blob/master/week-07/seminar_done.ipynb) week 7: Generative Adversarial Models*
  - Generative Adversarial Networks. VAE-GAN. AAE.
  - Energy based model.
- *[WIP] week 8: Natural Language Processing*
  - Embeddings.
  - RNN. LSTM, GRU.
- *[WIP] week 9: Attention and transformer model*
  - Attention Mechanism.
  - Transformer Model.
- *[WIP] week 10: Transfer Learning in NLP*
  - Pretrained Transformers. BERT. GPT.
  - Data Augmentation in Texts. Domain Adaptation.
- *[WIP] week 11: Recommender Systems*
  - Collaborative Filtering. FunkSVD.
  - Neural Collaborative Filtering.
- *[WIP] week 12: Reinforcement Learning for RecSys*
  - Reinforcement Learning. DQN Algorithm. 
  - DDPG Algorithm. Wolpertinger.
- *[WIP] week 13: Extras*
  - Research & Deploy.
  - Config API. Reaction.
  
## Environment

### Anaconda setup
```bash
# setup - env
conda create -n catalyst-dl python=3.7 anaconda
source activate catalyst-dl
conda remove nb_conda_kernels -y
conda install -c conda-forge nb_conda_kernels -y
conda install notebook jupyter nb_conda -y
conda remove nbpresent -y

# setup - jupyter
jupyter notebook password

# jupyter run
jupyter notebook --no-browser --ip 0.0.0.0 --port 8888
```

### Requirements
```bash
pip install -U catalyst==21.04.2 torch==1.8.0 albumentations==0.5.0
```

## Course staff & contributors

- [@AlexeySh](https://github.com/AlekseySh)
- [@artek0chumak](https://github.com/artek0chumak)
- [@elephantmipt](https://github.com/elephantmipt)
- [@Inkln](https://github.com/Inkln)
- [@lordofprograms](https://github.com/lordofprograms)
- [@Scitator](https://github.com/Scitator)
- [@zelcookie](https://github.com/zelcookie)

