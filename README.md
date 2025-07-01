# Video Joint Embedding Predictive Architectures for Facial Expression Recognition

This respository is a fork of the original [V-JEPA I repository](https://github.com/facebookresearch/jepa). For information how to setup and other information regarding the usage of the code, we refer the reader there.

## Data preparation

1. You need to acquire the CREMA-D and RAVDESS datasets.

   1. You can acquire CREMA-D [here](https://github.com/CheyneyComputerScience/CREMA-D).
   2. You can acquire the RAVDESS dataset [here](https://zenodo.org/records/1188976).
2. Before you can do any training you will need to create two types of `.csv` files (sorry for the inconvenience):

   1. For CREMA-D you need to create one `.csv` file per data split, containing the paths and labels to each video
   2. For RAVDESS you need to create two `.csv` files per training run, one for training, one for validation.
3. You need to place the paths to these `.csv` files in the corresponding configuration files under `configs/evals` and `config/full_video_evals.`

## Models

You can use the V-JEPA backbone used in our experiments [here](https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar).

Similar to the steps for data preparation, you will need to place the path to this model checkpoint in all configuration files.

## Training

The configuration files used to train a classification head are the ones in `configs/evals`. To run them (in a distributed data parallell setting) use:

```bash
python -m evals.main
	--fname=configs/evals/path_to_config_file.yaml
	--devices DEVICES
```

Make sure that only devices you want to use are visible to the process by setting `CUDA_VISIBLE_DEVICES`.
Most of our experiments were performed only using a single Nvidia RTX A5000 GPU.

### The `scripts` and `visualizations` directories...

... contain a handful of different scripts that were used to combine output logs produced by multiple devices during distributed data parallell training runs, scripts to handle `calm` and `neutral` classes and also produce the embedding visualizations presented in the paper.

## Citation

If you find this repository useful in your research, please consider giving a star ⭐️ and a citation.
If you have any questions please contact me at [lennart.eing@uni-a.de](mailto:lennart.eing@uni-a.de).
