## extract all captions from original DataComp shards

```bash
mkdir input
mkdir output
for i in `ls /gpfs_scratch/datacomp/medium_filtered/shards/*tar`; do f=`echo $i | rev | cut -d "/" -f 1 | rev | cut -d "." -f 1`; echo $i input/$f.json; python3 get_shard_captions.py --tar $i --output input/$f.json; done
```

## process all captions with ollama model

```bash
for i in `ls input/*json`; do f=`echo $i | cut -d "/" -f 2`; python3 captions_processor.py --input $i --output output/$f; done
```

## create prompts for image generation


## rebuild shards with generated data
