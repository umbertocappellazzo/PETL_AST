<div align="center">

[![](https://img.shields.io/badge/Code-github.umbertocappellazzo%2FPETL_AST-blue)](https://umbertocappellazzo.github.io/)
[![](https://img.shields.io/badge/Paper-arxiv.2312.03694-red)](https://arxiv.org/abs/2312.03694)

# Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers

[Umberto Cappellazzo](https://umbertocappellazzo.github.io/), [Daniele Falavigna](https://scholar.google.com/citations?user=LEaCpUMAAAAJ&hl=en), [Alessio Brutti](https://scholar.google.it/citations?user=dS643iQAAAAJ&hl=en), [Mirco Ravanelli](https://sites.google.com/site/mircoravanelli/)

</div>

This is the repository of the paper "**Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers**". The paper is currently under review. It explores the use of different parameter-efficient transfer-learning methods (**PETL**) applied to the Audio Spectrogram Transformer model for various audio and speech processing tasks. Our experiments reveal that *LoRA* and *Houlsby adapters* achieve the best results. We also show that adapters exhibit excellent performance in few-shot settings, whereas LoRA showcases superior scalability by leveraging an increasing number of parameters. We finally ablate the best configuration for both adapters and LoRA.

This repo also contains the code for additional experiments we carried out in the appendix, namely combining multiple PETL methods together and finding the best adapter configuration given a fixed boudget of parameters.



<div align="center">

|     <img src="images/AST_main.png" width='400'/>    |
| :-------------------------------------------------: |
| **Illustration of the AST model and the PETL methods.** |


</div>



## Environment setup
The requested libraries for running the experiments are listed in the requirements.txt file. Run the command below to install them.   

```
pip install -r requirements.txt
```

I use weights and biases (https://wandb.ai/site) for tracking my experiments (I warmly recommend it). Nonetheless, you can deactivate it by setting `--use_wandb = False` in the command line.

I expect that the user has already downloaded the datasets by him/herself.


# Running an experiment

To run an experiment, everything you need is to use the command ``` python3 main.py ``` followed by some arguments passed on to the command line to specify the setting. The mandatory parameters are:

- `--data_path`: the path to the folder containing the dataset. 
- `--dataset_name`: the selected dataset. As of now, 4 datasets are available: `['FSC', 'ESC-50', 'urbansound8k', 'GSC']`.
- `--method`: the selected PETL method. A list of supported PETL methods follows: `['linear', 'full-FT', 'adapter', 'prompt-tuning', 'prefix-tuning', 'LoRA', 'BitFit', 'adapter-Hydra', 'adapter+LoRA', 'adapter+prompt', 'adapter+prompt+LoRA']`.
- Other arguments can be passed to the command line, so please have a look at the `main.py` script for a detailed description.
- Hyper-parameters related to the optimization process and datasets can be inspected and modified at `hparams/train.yaml`. The current values correspond to the ones we used for our experiments and that led to the best results.

Each PETL method comes with some specific parameters. We provide a brief description below. Note that here we avoid including the references for brevity, please refer to the paper.

- **adapter**: `reduction_rate_adapter` --> it rules the bottleneck dim of the adapter module (e.g., if *d* is the hidden dimension and RR is the reduction rate, then the dim of the adapter is *d*/RR); `seq_or_par` --> whether to insert the adapter parallel or sequentially; `adapter_type` --> either Pfeiffer or Houlsby configuration; `adapter_block` --> either Bottleneck or Convpass; `apply_residual` --> whether to apply residual connections or not. As reported in the paper, parallel adapter should dispense with residuals, whereas sequential adapter benefits from residuals.
- **prompt-tuning**: `prompt_len_prompt` --> how many prompts to use; `is_deep_prompt` --> set to `True` if you want to enable *deep prompt-tuning* (DPT), otherwise *shallow prompt-tuning* (SPT); `drop_prompt` --> the dropout rate for the prompts. In our experiments we set it to `0.`.
- **LoRA**: `reduction_rate_lora` --> please see `reduction_rate_adapter`; `alpha_lora` --> the LoRA_alpha as defined in the original paper. This is used for scaling (e.g., s = alpha_lora/RR).

For example, suppose we want to test the adapter with configuration Bottleneck, parallel, Pfeiffer, RR = 64 to the ESC-50 dataset. Then, the command to run is:

```bash
python3 main.py --data_path '/path_to_your_dataset' --dataset_name 'FSC' --method 'adapter' --seq_or_par 'parallel' --reduction_rate_adapter 64 --adapter_type 'Pfeiffer' --apply_residual False --adapter_block 'Bottleneck'
```

### Few-shot Experiments
If you want to run few-shot learning experiments, you just need to set the flag `--is_few_shot_exp` to `True` and specify the # of samples per class `--few_shot_samples`.

### Ablation studies on the optimal configuration for adapter/LoRA
Finally, if you want to replicate the ablation studies on the best location to insert the adapter module into an AST layer, you need to specify `--is_adapter_ablation = True`. Two additional arguments must be specified beside the standard used for adapters: `--befafter` --> whether to include the adapter before or after the selected sub-layer; `--location` --> whether to insert the adpater into the feed-forward sub-layer (FFN) or multi-head self-attention sub-layer (MHSA). For the ablation studies for LoRA, just specifcy `--is_lora_ablation = True`, and choose the matrix/matrices you want to be updated with LoRA: `'Wq', 'Wq,Wk', 'Wq,Wv', 'Wq,Wk,Wv,Wo'`. 


# Contribution to the repository
While this repository comprises three downstream tasks and 4 datasets, I'd be more than happy to integrate other downstream tasks/dataset. Feel free to make a PR if you want to add other datasets etc. 


# Contact

Please, reach out to me at: umbertocappellazzo [at] gmail [dot] com for any question. 

# Acknowledgments

We acknowledge the support of the Digital Research Alliance of Canada (alliancecan.ca).

# Citation

```latex
@misc{cappellazzo2023parameterefficient,
      title={Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers}, 
      author={Umberto Cappellazzo and Daniele Falavigna and Alessio Brutti and Mirco Ravanelli},
      year={2023},
      eprint={2312.03694},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
