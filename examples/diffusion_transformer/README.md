# Diffusion transformer example

This example trains a diffusion transformer on MNIST with the linear interpolant flow matching formulation. It demonstrates how you can use ornamentalist in a full-featured research codebase.

The codebase supports multi-GPU, multi-node distributed data parallel training on SLURM clusters (although it's a bit overkill on MNIST). For didactic simplicity, I intentionally omit a few tricks, but if you swap out MNIST for imagenet, this will get you 90% of the way to a SOTA image generation setup.

## CLI (auto generated)

From the root of this repo, you can run the code simply by running `python examples/diffusion_transformer/main.py`. We use the `ornamentalist.cli()` feature, resulting in the following auto-generated CLI:

```bash
> python examples/diffusion_transformer/main.py --help

options:
  -h, --help            show this help message and exit

get_model_cls:
  Hyperparameters for examples.diffusion_transformer.dit.get_model_cls

  --get_model_cls.name  ...]
                        Type: str, choices: ('DiT-XL/2', 'DiT-XL/4', 'DiT-XL/8', 
                                             'DiT-L/2', 'DiT-L/4', 'DiT-L/8',
                                             'DiT-B/2', 'DiT-B/4', 'DiT-B/8',
                                             'DiT-S/3', 'DiT-S/4', 'DiT-S/8') (optional),
                        default=DiT-S/2

get_dataloaders:
  Hyperparameters for examples.diffusion_transformer.mnist.get_dataloaders

  --get_dataloaders.data_dir  ...]    Type: str (optional), default=/tmp/data
  --get_dataloaders.batch_size  ...]  Type: int (optional), default=256
  --get_dataloaders.num_workers  ...] Type: int (optional), default=8
  --get_dataloaders.pin_memory  ...]  Type: bool (optional), default=True

guided_prediction:
  Hyperparameters for examples.diffusion_transformer.trainer.guided_prediction

  --guided_prediction.cfg_omega  ...] Type: float (optional), default=1.0
  --guided_prediction.cfg_alpha  ...] Type: float (optional), default=0.0

generate:
  Hyperparameters for examples.diffusion_transformer.trainer.generate

  --generate.n_steps  ...] Type: int (optional), default=100

compute_loss:
  Hyperparameters for examples.diffusion_transformer.trainer.compute_loss

  --compute_loss.sigma  ...] Type: float (optional), default=0.0

train:
  Hyperparameters for examples.diffusion_transformer.trainer.train

  --train.num_steps  ...]          Type: int (optional), default=10000
  --train.log_every_n_steps  ...]  Type: int (optional), default=100
  --train.eval_every_n_steps  ...] Type: int (optional), default=1000

launcher:
  Hyperparameters for __main__.launcher

  --launcher.nodes  ...]      Type: int (optional), default=1
  --launcher.gpus  ...]       Type: int (optional), default=1
  --launcher.cpus  ...]       Type: int (optional), default=16
  --launcher.ram  ...]        Type: int (optional), default=64
  --launcher.timeout  ...]    Type: int (optional), default=20
  --launcher.partition  ...]  Type: str (optional), default=hopper
  --launcher.qos  ...]        Type: str (optional), default=normal
  --launcher.output_dir  ...] Type: str (optional), default=./outputs/
  --launcher.cluster  ...]    Type: str, choices: ('debug', 'local', 'slurm') (optional), default=debug
```

## Example usage:

Run a single-GPU job in the current process with the default parameters:
```bash
python examples/diffusion_transformer/main.py
```

Run a 2-GPU DDP job on SLURM:
```bash
python examples/diffusion_transformer/main.py --launcher.cluster slurm --launcher.gpus 2
```

Run a sweep over 4 config combinations as a SLURM array job:
```bash
python examples/diffusion_transformer/main.py --launcher.cluster slurm --compute_loss.sigma 0.0 0.1 --guided_prediction.cfg_omega 1.0 1.5
```