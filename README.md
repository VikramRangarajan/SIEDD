# SIEDD: Shared Implicit Encoder with Discrete Decoders

## [arXiv](https://arxiv.org/abs/2506.23382) | [Paper](https://arxiv.org/pdf/2506.23382) | [Project Page](https://vikramrangarajan.github.io/SIEDD/)

## TODO:
- [ ] Remove backblaze dependency for runpod and add `runpod_startup.sh` and `runpod_launch.sh`

## Installation and Setup
Created using [uv](https://docs.astral.sh/uv/) which you must first install. To use, clone the repository and run
```bash
uv pip install .
```

Also make sure to run wandb.login() before running experiments using wandb logging.

### Environment Variables

Some environment variables must be set using your shell, others need to be in .env.

If you are submitting SLURM jobs, you must create an .env file.

- For SLURM
    - Optional: Set the `SBATCH_QOS` environment variable
    - Optional: Set the `SBATCH_ACCOUNT` environment variable
    - Optional: Set the `SBATCH_PARTITION` environment variable
- For RunPod
    - Set your `RUNPOD_API_KEY` environment variable
    - Set `RUNPOD_VOLUME_ID` to the volume with the datasets, code, and virtual environment
    - Set the `B2_EXP_SCRATCH_PATH` environment variable to your backblaze path (ex: `b2://your-bucket/experiment_scratch`)
    - Set your `B2_APPLICATION_KEY` and `B2_APPLICATION_KEY_ID` to a backblaze s3 bucket containing the data in the structure shown below.

    This is used to transfer the config file to the pod.
- For PyTorch
    - Set `TORCHINDUCTOR_CACHE_DIR=/path/to/cache/` in a .env file in your project root to allow for warm starts when using torch.compile. **NOTE**: I have experienced errors when starting multiple jobs using the same cache at once. If this happens, delete the cache and try again.
    - Optional: For memory efficient training, use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (in .env file).
- For WandB
    - Set `WANDB_PROJECT` in the .env file
    - To group your runs together, you can use `WANDB_GROUP` (bash export)

### Dataset Setup
Make sure you first activate the virtual environment for this project

Place the folders containing the frame images according to the expected directory setup below
- UVG HD and 4k: [https://ultravideo.fi/dataset.html](https://ultravideo.fi/dataset.html)
    - Run `. download_UVG_hd.sh` and `. download_UVG_4k.sh`. These scripts will take a few hours each, so run them overnight
    - Run them from the project root directory
    - Requires: ffmpeg, curl
- DAVIS: [https://davischallenge.org/davis2016/code.html](https://davischallenge.org/davis2016/code.html)
- MarioKart: [https://www.youtube.com/watch?v=4yZlK2Ftjho](https://www.youtube.com/watch?v=4yZlK2Ftjho)
    - Run `. download_MarioKart.sh` from the project root directory
    - Requires: ffmpeg
- Bunny: From [scikit-video](https://github.com/scikit-video/scikit-video/blob/master/skvideo/datasets/data/bigbuckbunny.mp4)
    - Run `. download_BUNNY.sh`
    - Requires: ffmpeg, curl

### Additional Notes
- An `experiment_scratch` directory will be made as a "temp" directory and will contain all the SLURM .out files, launch scripts for SLURM and RunPod, and config .yaml files.
- All saved models will be located in `outputs/runs/{experiment_folder}`


## Usage

### Encoding
You can run the encoding process by doing

```bash
uv run --env-file .env train --data_path path_to_data --cfg path_to_cfg.yaml
```

from the project root. Config files can be generated using pydantic_yaml with the configs in exp.py as a good reference.

The data path is expected to be one of:
- A folder containing **3 channel RGB images** (jpeg, png) with the frames named in order (ex: f001.png, f002.png, ...)
- A yuv file (the config must show the size)
    - Note: The correct implementation converting yuv to rgb is *not* equivalent to FFMPEG's conversion, so do not use this if possible
- A video file (.mp4, .avi, .mov, .mkv)
- A folder containing folders of images (same formats as first bullet). This was used to train a shared encoder on the whole UVG-HD dataset.

To use [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/), this must be installed seperately (the pytorch bindings). If not installed correctly, any use of tiny-cuda-nn related modules will raise an error.

### Decoding
To run the decoding process, use
```bash
uv run decode_frames [PATH] [FRAMES] --output [FRAME OUTPUTS] --resolution [RESOLUTION]
```
Run the following command for more details:
```bash
uv run decode_frames --help
```
## Experiments

The experiments can be run on either SLURM or RunPod
- For RunPod:
    - NOTE: The runpod job launching will batch all the videos in the dataset sequentially as runpod has simultaneous gpu request limits
    - **Only run one experiment at a time through the CLI to ensure that it is launched properly**
- For SLURM:
    - Submits 1 SLURM job per video
    - Recommended to run on login node

Both methods request an A5000 GPU for consistency.

To run experiments, run

```bash
uv run exp x,y,z # for SLURM
```
or
```bash
uv run runpod_exp x # for RunPod
```

where x, y, z are experiment numbers (labeled in exp.py). Dataset choices for experiments are also present in exp.py.

### To Reproduce Paper Results:
- SIEDD-S: experiment 50
- SIEDD-M: experiment 53
- SIEDD-L: experiment 56
- Shared encoder weight transfer: experiment 52. Set the SHARED_ENCODER_PATH environment variable to the path of a shared encoder from another experiment.
- The 720p long video results use SIEDD-S
- UVG 4k: experiment 45
- Super Resolution: Use `uv run superres --save_path [Folder with 1080p SIEDD model in it] --data_path [Ground truth 4k data path] --name [Experiment Name]`. This will run the super resolution forward passes on each frame of the video and calculate metrics with the ground truth 4k frames.
- Any resolution decoding: TODO
- Sampling rate ablation: experiment 10. This was run on a very early iteration of SIEDD.
- Shared encoder iterations: experiment 48
- GPU Parallelization: The encoding time was calculated theoretically, but a working POC can be run using `python multigpu.py [same arguments as for uv run train]`. WandB logging is not supported here.
- Model layers and layer dimension ablation: experiments 41 and 42

Supplementary Experiments:
- Additional reconstruction metrics (VMAF/FLIP): Use `uv run extra_metrics --data_path [Data Path] --save_path [Model Save Path] --name [Experiment Name]`
- Group size ablation: experiment 54
- LoRA Ablation: experiment 55. To see LoRA experiments on early iterations of SIEDD, try experiment 7
- Quantization sweep: Use `uv run quant_exp --data_path [Ground Truth Data Path] --save_path [UNQUANTIZED Model Save Path] --name [Experiment Name]`
    - Make sure that you point the `--save_path` to a run where `run_config.quant_cfg.quantize = False`. We used experiment 40 to get the unquantized models. Be warned that they take up a lot more space.
- Video Denoising: experiment 49. To run the baselines, run `python denoising_baseline.py`.

Other interesting experiments not mentioned in the paper:
- Learning rate search: experiment 5
- SinLoRA $\omega$ search: experiment 8 (no difference found)
- 1080p patch size testing: experiment 51
- Optimizer test: experiment 12. schedulefree had much more stable training than the others, so it became the default.
- Positional encoder testing: We tested no PE, Fourier + Sine, Fourier + ReLU, Multiresolution hash grids, and NeRF. NeRF outperformed no PE slightly while everything else performed worse (experiments 14, 24, 32, 33, 34)
- Patching experiments adding offset patches to the training, and using strided patches instead of regular patches (experiments 20, 21)
- (x, y) coordinate normalization range (experiment 22)
- Image transformations (experiment 23). z-score made the most sense and performed the best, and became the default.
- SIEDD without a shared encoder: experiment 25 (doesn't work well)
- Patches with a nonlinear final activation: experiment 30
- A single shared encoder across all of UVG: experiment 38
- QAT: experiment 43
- Bottlenecked MLP: experiment 44

### To Run Your Own Configs:
- Add to exp.py, using the other experiments as a reference. See configs.py for all the options.

## Expected Directory Setup

```md
SIEDD/
├── data
│   ├── DAVIS
│   │   ├── blackswan
│   │   ├── bmx-trees
│   │   ├── boat
│   │   ├── breakdance
│   │   ├── camel
│   │   ├── car-roundabout
│   │   ├── car-shadow
│   │   ├── cows
│   │   ├── dance-twirl
│   │   └── dog
│   ├── UVG
│   │   ├── Beauty_1080p
│   │   ├── Bosphorus_1080p
│   │   ├── HoneyBee_1080p
│   │   ├── Jockey_1080p
│   │   ├── ReadySetGo_1080p
│   │   ├── ShakeNDry_1080p
│   │   └── YachtRide_1080p
│   ├── UVG_4k
│   │   ├── Beauty_4k
│   │   ├── Bosphorus_4k
│   │   ├── HoneyBee_4k
│   │   ├── Jockey_4k
│   │   ├── ReadySetGo_4k
│   │   ├── ShakeNDry_4k
│   │   └── YachtRide_4k
│   ├── UVG_Bunny
│   │   └── Bunny_720p
│   └── YOUTUBE_8M
│       └── MarioKart
├── experiment_scratch
├── outputs
│   └── runs
└── .env
```

## Citation

```
TODO
```
