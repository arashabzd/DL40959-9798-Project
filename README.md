# DL40959-9798-Project
Required material for Deep Learning Course Project

Instructor: Mahdieh Soleymani

Course Page: [http://ce.sharif.edu/courses/97-98/2/ce959-1/](http://ce.sharif.edu/courses/97-98/2/ce959-1/)


## Implementation
Implementation is based on [What You Get Is What You See: A Visual Markup Decompiler](https://arxiv.org/pdf/1609.04938v1.pdf) paper.

## Prerequsites
* torch
* torchvision
* Distance
* nltk

```
pip3 install -r requirements.txt
```

## Dataset
Dataset folder in this repository is a template.
You can download dataset from [here](https://drive.google.com/file/d/1Et9gOAAzHFt9dehw9BzNNQjItVsWlw7a/view?usp=sharing) and replace the Dataset folder with the one you've downloaded.



## Usage

#### For Evaluating BLEU Score
```bash
python3 Evaluation/bleu_score.py --target-formulas target.txt --predicted-formulas predicted.txt --ngram 5
```

#### For Evaluating Edit Distance Accuracy

```bash
python3 Evaluation/edit_distance.py --target-formulas target.txt --predicted-formulas predicted.txt
```
