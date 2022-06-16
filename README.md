## Model description
The RST-LSTM module is used to refine predictions of a high-performance sequential text classifier (BERT) on documents with rhetorical structure.

### Training pipeline
1. The first stage involves fine-tuning the sequential model on the dataset including texts of different lengths and complexity. 
2. In the second stage, we freeze the base model and then train a discourse-aware neural module on top of it for the classification of texts with discourse structure.

### Prediction pipeline
1. The text is parsed with end-to-end RST parser
2. Predictions are obtained on each discourse unit in the structure with the BERT
3. Non-elementary discourse structures with assigned BERT predictions go through the trained RST-LSTM 

## RuARG-2022

This repository is for applying this method on [RuARG-2022](https://github.com/dialogue-evaluation/RuArg) argument mining shared task. 

### Requirements

 - AllenNLP == 2.9.3
 - [IsaNLP RST parser for Russian](https://github.com/tchewik/isanlp_rst)

### Code
 - ``*.ipynb`` - Data analysis, scripts for training and evaluation.
 - ``models_scripts/`` - BERT-based and RST-LSTM-based classifiers scripts for AllenNLP. 
   - Both classifiers predict two labels (Stance and Premise) jointly.
   - RST-LSTM includes both Child-sum and Binary options for Tree LSTM (no significant difference was found for the current task, Binary by default).

## Reference

```bibtex
@INPROCEEDINGS{chistova2022dialogue,
      author = {Chistova, E. and Smirnov, I.},
      title = {Discourse-aware text classification for argument mining}},
      booktitle = {Computational Linguistics and Intellectual Technologies. Papers from the Annual International Conference "Dialogue" (2022)},
      year = {2022},
      number = {21},
      pages = {93--105}
}
```
