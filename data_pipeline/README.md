A more efficient data pipeline:

1) (in-cluster) untar all shards into a temporary folder (see e.g. 01_extract_shards_small.sh)
2) (in-cluster) filter all samples with at least 1 face detected 
3) (remotely) filter samples with English caption and process captions with an LLM to produce t2i prompts. 
4) (in-cluster) generate images with prompts from step 3 and edit/modify captions directly in the temp folder if needed.
5) (in-cluster) tar all shards from the temp folder

-----------------------

Approx time per step/scale:

| Step | Samples/s | Time (small) | Time (medium) | Time (large) |
|------|-----------|--------------|---------------|--------------|
| 1    | 500       | 2.00 h       |               | 48h          |
| 2    | 3500      | 0.25 h       |               |              |
| 3    | 50        | 6.00 h       |               |              |
| 4    | 128       | 1.00 h       |               |              |
| 5    | 500       | 2.00 h       |               |              |
|------|-----------|--------------|---------------|--------------|
|TOTAL |           | 11.25 h      |   ~112.50 h   |  ~1125.00 h  |

-----------------------

## 1) untar all shards into a temporary folder
script `01_extract_shards_small.sh` extracts files from `SRC_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_filtered/shards"`
into `TARGET_DIR="/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/tmp"`

script `01_extract_shards_small.sh` uses xargs for parallel processing.

## 2) filter all samples with at least 1 face detected 
script `02_filter_noface.sh` reads json files from `/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/tmp` and process then with python script `filter_noface.py`. It creates a json file per shard (e.g. json file prompts/00000000.json correspond to captions in 00000000.tar shard) in `/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/captions`, each file has a dictionary with filenames as keys and captions as values:

```
{
    "000d6d31d370fcf64fcb4ce869c5ed96-0.json": "Jos Chessani medalla oro tokio 2020",
    "0041c043f693222ade350169e6a4a663-0.json": "granizosaude1",
    "0098ebd8027f13d12fb3d730f56bdafe-0.json": "HISENSE - Smart TV UHD 4K 65'' Vidaa Dolby Vision 65A6H",
...
}
```
## 3) filter samples with English caption and process captions with an LLM to produce t2i prompts.
script `03_captions_processor.py` read json files from `/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/captions` and creates files with same format in `/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/prompts`. This files (one per shard) sonatain the modified captions that will be used to generate images in next step.

the script `03_captions_processor.py' assumes you have several ollama instances serving in different ports. you can configure the ollama ports and model via arguments:

* --input "Path to folder with shard tar files."
* --output "Path to output folder where JSON files with results will be saved."
* --num_threads "Number of concurrent threads." (default=20)
* --ports "Comma-separated list of ports for ollama instances." (default="11434,11435,11436,11437")
* --model_name "Name of the ollama model." (default="person_tagger")
* --timeout type=int, "Timeout for each request in seconds." (default=10)

Calling the script:

```bash
python3 captions_processor.py --input /path/to/captions/ --output /path/to/prompts
```

generates a json file for each shard (e.g. json file prompts/00000000.json correspond to captions in 00000000.tar shard). Each json file is a dictionary with caption ID as key and the processed caption as value.

## 4) generate images with prompts from previous step
script `../slurm_job_scripts/generate_images_small.sh` launches a SLURM srun command that calls python script `04_generate_images_multigpu.py`. It generates images using prompts from `/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/prompts` and saves generated images and edited json/txt files into a clean directory `/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/edits`

## 5) rebuild shards with generated data
script `05_reshard_small.sh` calls python `reshard.py` to create tar files faster by using sequential readings from the original tar files in `/gpfs/scratch/ehpc42/datasets/datacomp/small_filtered/shards`, updating edited samples from `/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/edits`. This produces a new set of shards with the hybrid dataset in `/gpfs/scratch/ehpc42/datasets/datacomp/small_hybrid/shards`
