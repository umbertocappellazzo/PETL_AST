# Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers
This is the repository of the paper "Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers". The paper is currently under review. It explores the use of different parameter-efficient transfer-learning methods (**PETL**) applied to the Audio Spectrogram Transformer model for various audio and speech processing tasks. As *adapter-tuning* emerges as the best approach in various settings (e.g., few-shot learning, and when we increase the # of trainable params), we conduct some specific ablation studies on them, concluding that for our scenario the best configuration entails the insertion of the adapter module ***parallel to and before the MHSA sub-layer***. We also show that residual connections can lead to drastic changes in the overall performance if overlooked.  

This repo also contains the code for additional experiments we carried out (like combining multiple PETL methods together) but we did not include in the original submission for space constraints, nonetheless you can look at them in the appendix. 




## Environment setup
The requested libraries for running the experiments are listed in the requirements.txt file. Run the command below to install them.   

```
pip install -r requirements.txt
```

I use weights and biases (https://wandb.ai/site) for tracking my experiments (I warmly recommend it). Nonetheless, you can deactivate it by setting `--use_wandb = False` in the command line.

I expect that the user has already downloaded the datasets by himself/herself.
