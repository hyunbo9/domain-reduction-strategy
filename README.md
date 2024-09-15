# Domain Reduction Strategy for Non-Line-of-Sight Imaging

Hyunbo Shim<sup>&ast;</sup>, In Cho<sup>&ast;</sup>, Daekyu
Kwon, [Seon Joo Kim](https://sites.google.com/site/seonjookim/) (&ast; Equal contribution)

[[`arXiv`](https://arxiv.org/abs/2308.10269)] [[`BibTeX`](#Citation)]

An official implementation of the
paper [Domain Reduction Strategy for Non-Line-of-Sight Imaging](https://arxiv.org/abs/2308.10269) (in ECCV 2024).

## Preparation

### Starting with docker

We provide a prebuilt [docker image]() with the required packages installed.

```bash
docker pull join16/join16/nlos-domain-reduction:py39-cu113
```

Or you can also build a docker image by running the following command.

```bash
docker build -t nlos-domain-reduction:py39-cu113 .
```

### Installing packages with pip

Instead of using a docker image, you can also manually install the required packages using pip. We recommend using a
virtual
environment to avoid conflicts with other packages.

```bash
pip install -r requirements.txt
```

We tested our code on Python 3.9, torch 2.0.1, and CUDA 11.3.

### Building CUDA kernel

Our forward propagation model (hidden scenes to measurements) is implemented as a CUDA kernel.
To build the CUDA kernel, run the following command.

```bash
python setup.py build_ext --inplace
````

### Logging with wandb (optional)

To log experiment results using wandb, place the wandb API key in the `config/wandb.yaml` file.

```yaml
project: nlos-domain-reduction
entity: {your_wandb_username}
api_key: {your_wandb_api_key}
```

## Dataset

For the [ZNLOS](https://graphics.unizar.es/nlos_dataset) dataset, first download the original data from the official
website.
Then, modify `raw_root_dir` in the `config/data/znlos.yaml` to the root directory of the downloaded data.
For the [Stanford](https://github.com/computational-imaging/nlos-fk) real world dataset, download the original data and
modify `raw_root_dir` in the `config/data/stanford.yaml` to the path of the downloaded data.
Our evaluation script will automatically preprocess the data.

## Reconstruction

Run our model using the following command.

```bash
python main.py config/main_bunny.yaml -n {experiment_name} -g {gpu_id}
```

## Acknowledgements

We sincerely appreciate the authors for sharing their code and data, which greatly helped our research.

## <a name="citation"></a> Citation

```BibTex
@article{shim2024drs,
  author    = {Shim, Hyunbo and Cho, In and Daekyu, Kwon and Kim, Seon Joo},
  title     = {Domain Reduction Strategy for Non Line of Sight Imaging},
  journal   = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2024},
}
```