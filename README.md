# IntuRec
This is the pytorch implementation of the paper at KDD 2026:
> **Intuition-Guided Latent Reasoning for LLM-Based Recommendation**
>
> Chang Liu, Yimeng Bai, Xiaoyan Zhao, Yang Zhang, Qifan Wang, Fuli Feng, Wenge Rong.

## Usage

### Data for stage 1
- Step1: Download the required subset **X** from the Amazon Review dataset. Then, under the path `your_path/IntuRec/data/`, create a folder named **X**. Inside this folder, create the following eight subdirectories: `train`, `valid`, `test`, `train_extra`, `valid_extra`, `test_extra`, `info`, and `raw`. After that, place the two original files of the downloaded subset—`X_5.json.gz` and `meta_X.json.gz`—into the `raw` folder.
- Step 2: Run the two code cells in `your_path/IntuRec/data/data_process.ipynb` sequentially. This will prepare all the required files in the `train`, `valid`, `test`, `info`, and `raw` directories.

### Training & Evaluation for stage 1
Run the following script based on the hyperparameter ranges provided in the paper
```
bash run_inturec_f1.sh
```

### Data for stage 2

- Step1: Run the following script according to the number of intu items you need.

	```
	bash genf1midres.sh
	```

- Step2: Run the first cell in `your_path/IntuRec/preparedata2f2.ipynb` to determine the balance ratio for the selected subset **X**. Then, based on the logic in the second cell, fill in the corresponding balance ratio. Finally, execute the second and third cells sequentially.

### Training & Evaluation for stage 2

Run the following script based on the hyperparameter ranges provided in the paper

```
bash run_inturec_f2.sh
```

