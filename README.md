# BCGen

BCGen is a project to automatically generate comments for given code snippets with their contexts.
The source code and dataset are opened.

# Requirements

Python 3.8

Pytorch 1.6

NLTK 3.5

# Config

Every model has its own configuration, we can change the configuration in the `main.py` by setting the field values of the class `Config`.

# Model Training

Train the BCGen: `python main.py relate [main_folder]`

Train the Deepcom: `python main.py deepcom [main_folder]`

Train the Hybrid-Deepcom: `python main.py mix [main_folder]`

Train the Seq2Seq: `python main.py seq2seq [main_folder]`

Train the model with using java method as context: `python main.py addmethod [main_folder]`

In every training epoch, if the model has the greater performance, this model will be saved in "`[main_folder]`/model/`[model_name]`_`[training_epoch]`".

# Dataset

Our dataset is opened in [https://drive.google.com/drive/folders/1M_YZE_ykpy3iJMm_rheOzI-B42ACph_l?usp=sharing](https://drive.google.com/drive/folders/1M_YZE_ykpy3iJMm_rheOzI-B42ACph_l?usp=sharing)