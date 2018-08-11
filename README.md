# W266 - Text summarization using pointer generator networks

## File Reference
- exploration - Contains code for data exploration as well as preprocessing
- baseline: Contains code for the baseline model 
- pointergen: Contains code for the final model 
- Analyze Results.ipynb: Notebook of analysis of the results from each model
- reference_data: 
	- Generated output: https://drive.google.com/file/d/1NiSXv3LYwttZBjBy-nTYBzlbBkHfPrMa/view?usp=sharing
	- Model: https://drive.google.com/file/d/184Db7Lty9ACbSxf7glV2ePQ8M3Jl0ORu/view?usp=sharing


## Baseline model (Sequence to sequence with attention)
- baseline_train.ipynb: Notebook with code to train the model.
- baseline_view_sample.ipynb: Notebook to view a random example.This uses the trained model to generate a summary.
- baseline_compute_results.ipynb: Notebook to compute the ROUGE scores for the generated summary.
- baseline_view_specific_example.ipynb: Notebook to view a specific example.Provide a number between 0 to 10000 (depending on how the data is setup)


## Pointer generator model with coverage (Baseline + Pointer generator with coverage)
- pointergen_train.ipynb: Notebook with code to train the model.
- pointergen_view_sample.ipynb: Notebook to view a random example.This uses the trained model to generate a summary.
- pointergen_compute_results.ipynb: Notebook to compute the ROUGE scores for the generated summary.
- pointergen_view_specific_example.ipynb: Notebook to view a specific example. Provide a number between 0 to 10000 (depending on how the data is setup) to view a generated summary for that example.


## Dependencies
- Tensorflow
- Python 3
- ROUGE : https://github.com/pltrdy/rouge
- Stanford core NLP: https://stanfordnlp.github.io/CoreNLP/
- Original Data: https://cs.nyu.edu/%7Ekcho/DMQA/
