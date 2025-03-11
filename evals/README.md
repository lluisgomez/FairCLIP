## Implements fairness metrics from existing evaluation frameworks:

- Kong at al. (2024). Mitigating test-time bias for fair image retrieval. NeurIPS 2024.
    - `eval_So-B-IT.py` Implements normalized entropy and Contrastive Attribute-Score (C-ASC) metrics.
      
- Hamidieh et al. (2024). Identifying implicit social biases in vision-language models. AAAI/ACM Conference on AI, Ethics, and Society 2024.
    - `eval_coco_neutral.py` Implements Bias@K metric for COCO (similar to Table 2 of the paper)
    - `eval_flickr_neutral.py` Implements Bias@K metric for Flickr30K (similar to Table 2 of the paper)
      
- Geyik et al. (2019). Fairness-aware ranking in search & recommendation systems with application to linkedin talent search. ACM SIGKDD 2019.
    - TODO add NDKL implementation
 

---- 

Apart from fairness metrics, we also need standard retrieval performance metrics (and zero-shot evals) to ensure fairness improvements don't compromise overall accuracy/utility. 
Scripts `eval_coco_neutral.py` and `eval_flickr_neutral.py` also report Recall@K on COCO and Flickr30K datasets.
For zero-shot evaluation we use DataComp evaluation whch is based on CLIP_benchmark:

```
conda activate datacomp
cd datacomp
python evaluate.py --train_output_dir /gpfs_projects/jobs_output/datacomp-scale-small-filtered-seed0/ --data_dir /gpfs_projects/datasets/datacomp/eval/
```
