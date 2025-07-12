# Diffusion transformer example

This example trains a diffusion transformer on MNIST with the linear interpolant flow matching formulation. It demonstrates how you can use ornamentalist in a full-featured research codebase.

The codebase supports multi-GPU, multi-node distributed data parallel training on SLURM clusters (although it's a bit overkill on MNIST). For didactic simplicity, I intentionally omit a few tricks, but if you swap out MNIST for imagenet, this will get you 90% of the way to a SOTA image generation setup.

## CLI (auto generated)

From the root of this repo, you can run the code simply by running `python examples/diffusion_transformer/main.py`. We use the `ornamentalist.cli()` feature, resulting in the following auto-generated CLI:

```bash
> python examples/diffusion_transformer/main.py --help

options:
  -h, --help            show this help message and exit

launcher:
  --launcher.nodes int
  --launcher.gpus int   (per node)
  --launcher.cpus int   (per GPU)
  --launcher.ram int    (per GPU)
  --launcher.timeout int
                        (mins)
  --launcher.partition str
  --launcher.output_dir str
  --launcher.qos {normal,high}
  --launcher.cluster {debug,local,slurm}

guided_prediction:
  Hyperparameters for trainer.guided_prediction

  --guided_prediction.cfg_omega  ...]
                        Type: float (optional), default=1.0
  --guided_prediction.cfg_alpha  ...]
                        Type: float (optional), default=0.0

generate:
  Hyperparameters for trainer.generate

  --generate.n_steps  ...]
                        Type: int (optional), default=100

compute_loss:
  Hyperparameters for trainer.compute_loss

  --compute_loss.sigma  ...]
                        Type: float (optional), default=0.0

train:
  Hyperparameters for trainer.train

  --train.num_steps  ...]
                        Type: int (optional), default=10000
  --train.log_every_n_steps  ...]
                        Type: int (optional), default=100
  --train.eval_every_n_steps  ...]
                        Type: int (optional), default=1000

get_model_cls:
  Hyperparameters for dit.get_model_cls

  --get_model_cls.name  ...]
                        Type: str, choices: ('DiT-XL/2', 'DiT-XL/4', 'DiT-XL/8', 'DiT-L/2', 'DiT-L/4', 'DiT-L/8', 'DiT-B/2', 'DiT-B/4', 'DiT-B/8', 'DiT-S/2', 'DiT-S/4', 'DiT-S/8') (optional),
                        default=DiT-S/2

get_dataloaders:
  Hyperparameters for mnist.get_dataloaders

  --get_dataloaders.data_dir  ...]
                        Type: str (optional), default=./data
  --get_dataloaders.batch_size  ...]
                        Type: int (optional), default=256
  --get_dataloaders.num_workers  ...]
                        Type: int (optional), default=8
  --get_dataloaders.pin_memory  ...]
                        Type: bool (optional), default=True

```