## 1) Extract all captions from original DataComp shards and process all captions with ollama model

the `captions_processor.py' script assumes you have several ollama instances serving in different ports. you can configure the ollama ports and model via arguments:

* --input "Path to folder with shard tar files."
* --output "Path to output folder where JSON files with results will be saved."
* --num_threads "Number of concurrent threads." (default=20)
* --ports "Comma-separated list of ports for ollama instances." (default="11434,11435,11436,11437")
* --model_name "Name of the ollama model." (default="person_tagger")
* --timeout type=int, "Timeout for each request in seconds." (default=10)

Calling the script:

```bash
python3 captions_processor.py --input /path/to/shards/ --output output/
```

generates a json file for each shard (e.g. json file output/00000000.json correspond to captions in 00000000.tar shard). Each json file is a dictionary with caption ID as key and the processed caption as value, e.g.:

```
{
    "64946f8713ed4685b3d8f0e0627b10ff-0.txt": "Men's navy printed silk satin shirt *** Women's navy printed silk satin shirt",
    ...
}
```

## create prompts for image generation


## rebuild shards with generated data
