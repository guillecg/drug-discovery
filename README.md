
# Machine Learning for Drug Discovery

## 0. Architecture

- **Modular pipelines**
	- **Data enrichment (see ideas)**

	- **Data preprocessing**
		- SMILES: canonizing, cleaning, etc.
		- TODO: check other data sources/types
		- Imbalanced classes

	- **Feature engineering** (ideas from [here](http://www.bioinf.jku.at/research/DeepTox/))
		- ECFP
		- DFS
		- 3D features based on MOPAC
		- Quantum-mechanical descriptors
		- Tanimoto
		- Minmax
		- Various 2D, 3D and pharmacophore kernels
		- In-house toxicophore and scaffold features
		- **Dimensionality reduction:**
			- PCA
			- FastICA
			- Manifolds

	- **Feature selection**:
		- Variance Inflation Factor (VIF)
		- F2 (Scikit-learn)
		- Forest importance

	- **ML setup:**
		- Validation algorithm:
			- Hold-out + CV on train
			- Nested CV

	- **ML algorithms:**
		- Scikit-learn (RF, SVM, Elastic Nets, Gradient boosting, etc.)
		- Ensembles (XGBoost, LightGBM, stacking, etc.)
		- PyTorch (LSTM, 1D-CNN, GNN, GANs, etc.)

	- **Evaluation:**
		- Metrics
		- Explainability:
			- Chemical space visualization
			- Shapley
			- ELI5
			- LIME


## 1. Ideas

- **Data Enrichment:**
	- Genomic/transcriptomic data
	- Interactomic data
	- Polypharmacology
	- [KSPA](https://www.frontiersin.org/articles/10.3389/fenvs.2015.00080/full)
	- Latent Dirichlet Allocation ([Predictive Toxicogenomics Space, PTGS](https://youtu.be/qH3z5GwccxE?t=1431))

- **Active Learning for increasing labelled samples**

- **Generative models:**
	- **Examples:**
		- [ReLeaSE](https://arxiv.org/abs/1711.10907)
		- [REINVENT](https://github.com/MarcusOlivecrona/REINVENT)
	- **GANs**
	- **RL:**
		- Actions = add fragment, state = SMILES, reward = activity/similarity
		- Actions = add letter, state = SMILES, reward = activity/similarity

- **Generative models + labelling with Active Learning**

- **Supervised Learning:**
	- LSTM with SMILES
	- GNN with graph representations (maybe SDFs?)

- **Assessment/evaluation:**
	- Similarity (assumption: it equals to activity)
	- Direct mapping to activity (using Supervised Learning)
	-
- **DeepTox:**
	- **Machine Learning methods:**
		- SVMs with various kernels
		- Random Forests
		- Elastic Nets
	- **Features and kernels:**
		- ECFP
		- DFS
		- 3D features based on MOPAC
		- Quantum-mechanical descriptors
		- Tanimoto
		- Minmax
		- Various 2D, 3D and pharmacophore kernels
		- In-house toxicophore and scaffold features

- **Resources:**
	- **Datasets:**
		- [Tox21](http://bioinf.jku.at/research/DeepTox/tox21.html)
