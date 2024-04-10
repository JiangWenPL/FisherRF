# FisherRF: Active View Selection and Uncertainty Quantification for Radiance Fields using Fisher Informations
Wen Jiang, Boshu Lei, Kostas Daniilidis<br>
| [Project page](https://jiangwenpl.github.io/FisherRF/) | [arxiv](https://arxiv.org/abs/2311.17874) | [full paper](https://arxiv.org/abs/2311.17874) <br>
<!-- ![Teaser image](assets/teaser-cropped.gif) -->
<img src="assets/teaser-cropped.gif" alt="drawing" width="400"/>

## Environment setup
This repo is heavely build on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/main). Please follow their repo to set-up and compile most things about 3D Gaussian Splatting. Beside, this code was tested with the following dependencies:

```
einops==0.7.0
Pillow==10.2.0
pymeshlab==2023.12.post1
scipy==1.12.0
wandb==0.15.12
```

### Clone our repo with submodules:
```bash
git clone git@github.com:JiangWenPL/FisherRF.git --recursive
```

### Install extensions:
```bash
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install -e ./diff/ -v
pip install -e ./var_diff -v # install ActiveNeRF version
```

### Download Datasets (Optional)
Please get the NeRF-synthetic from: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 (nerf_synthetic.zip).
The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/)

## Running our code

Please use scripts under `./scripts/` to run different experiments with different configurations. The first arguments of the script is the path to the scene that contains `trainsforms_*.json` and the second is the path which you would like to save your experiment. For example:
```bash
# Run 3D GS + FisherRF
bash scripts/blender_seq1.sh /PATH/TO/YOUR/DATASET/lego YOUR_EXP_PATH H_reg

# Run 3D GS + ActiveNeRF
bash scripts/blender_seq1.sh /PATH/TO/YOUR/DATASET/lego YOUR_EXP_PATH variance
```


## Citing
If you find this code useful for your research or the use data generated by our method, please consider citing our paper:
```
@article{Jiang2023FisherRF,
      title={FisherRF: Active View Selection and Uncertainty Quantification for Radiance Fields using Fisher Information},
      author={Wen Jiang and Boshu Lei and Kostas Daniilidis},
      journal={arXiv},
      year={2023}
  }
```

## Acknowledgements
This project builds heavily on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/main) and [Plenoxels](https://github.com/sxyu/svox2). 
We thanks the authors for their excellent works!
If you use our code, please also consider citing their papers as well.