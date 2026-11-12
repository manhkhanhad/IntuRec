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

- Step2: Run the standalone preparation script from the IntuRec root. It creates
  `train_extra/extra.json`, `valid_extra/extra.json`, and
  `test_extra/extra.json`, calculates the training balance ratio, and enriches
  each split with the generated intuition lists and scores.

	```
	python prepare_stage2_data.py \
	  --dataset X \
	  --source-data-root /path/to/stage1/data \
	  --output-data-root data \
	  --beam-num 15 \
	  --balance-ratio Y \ get this balance ratio from preparedata2f2.ipynb
	```

  Here, `--source-data-root` must contain
  `X/{train,valid,test}/X*.csv`. The output data root must contain
  `train_beam_16.pt`, `valid_beam_15.pt`, and `test_beam_15.pt` under `X/`
  when `--beam-num 15` is used. To use a fixed balance ratio instead of the
  automatically calculated value, add `--balance-ratio VALUE`.

### Training & Evaluation for stage 2

Run the following script based on the hyperparameter ranges provided in the paper

```
bash run_inturec_f2.sh
```


python prepare_stage2_data.py \
  --dataset CDs_and_Vinyl \
  --source-data-root /common/home/users/m/mk.ngohuu.2024/research/R3/data \
  --output-data-root data \
  --beam-num 15 \
  --balance-ratio 0.3120
