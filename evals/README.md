## Implements fairness metrics from existing evaluation frameworks:

- Hamidieh et al. (2024). Identifying implicit social biases in vision-language models. AAAI/ACM Conference on AI, Ethics, and Society 2024.
    - `eval_So-B-IT.py` Implements normalized entropy and Contrastive Attribute-Score (C-ASC) metrics for FairFace dataset.
      
- Kong at al. (2024). Mitigating test-time bias for fair image retrieval. NeurIPS 2024.
    - `eval_coco_neutral.py` Implements Bias@K metric for COCO (similar to Table 2 of the paper)
    - `eval_flickr_neutral.py` Implements Bias@K metric for Flickr30K (similar to Table 2 of the paper)
      
- Geyik et al. (2019). Fairness-aware ranking in search & recommendation systems with application to linkedin talent search. ACM SIGKDD 2019.
    - TODO add NDKL implementation
 

---- 

## TODO

The metrics implementations of Kong et al. and Hamidieh et al. are based on descriptions provided in their respective papers, as official implementations are not publicly available. Our code yields results closely matching those reported for the baseline CLIP model, but not exactly the same. These deviations may stem from differences in: (1) the precise FairFace dataset splits used in Hamidieh et al., and (2) variations in how text "query prompts" are constructed in both Kong et al. and Hamidieh et al.

---- 

### DataComp (CLIP benchmark) evals

Apart from fairness metrics above, we also need standard retrieval performance metrics (and zero-shot evals) to ensure fairness improvements don't compromise overall accuracy/utility. 
Scripts `eval_coco_neutral.py` and `eval_flickr_neutral.py` also report Recall@K on COCO and Flickr30K datasets.


For zero-shot evaluation we use DataComp evaluation whch is based on CLIP_benchmark:

```
conda activate datacomp
cd datacomp
ulimit -n 4096  # Increase open file limit to avoid "Too many open files" errors during evaluation
python evaluate.py --train_output_dir /gpfs_projects/jobs_output/datacomp-scale-small-filtered-seed0/ --data_dir /gpfs_projects/datasets/datacomp/eval/
```

This command evaluates CLIP on 40 datasets, including ImageNet1K, MSCOCO, and Flickr. It also performs fairness evals on FairFace and UTKFace, similar to the original OpenAI CLIP paper (section 7.1. Bias).



---------------------

Initial results on CLIP benchmark datasets for ViT-B-16 on DataComp large scale data. Our filtered data (CLIP 40%) yields slightly lower numbers than the CLIP 30% baseline (commonpool_l_clip_s1b_b8k) because initial pool is different (~20% of DataComp-large is missing).

Model         | Dataset size | Samples seen | ImageNet1K Acc. | COCO t2i R@1 | COCO i2t R@1 | Flickr30K t2i R@1 | Flickr30K i2t R@1 |
 ------------ | --- | --- | --- | --- | --- | --- | --- |
ViT-B-16 (OpenaAI)                     | 400M | 13B  | 0.64 | 0.30  | 0.51 | 0.59 | 0.78  |
ViT-B-16 (DataComp CLIP + Image-based) | 140M | 1.3B | 0.63 | 0.32  | 0.48 | 0.55 | 0.73  |
 ------------ | --- | --- | --- | --- | --- | --- | --- |
ViT-B-16 (Baseline CLIP 30%)           | 385M | 1.3B | 0.57 | 0.29  | 0.44 | 0.51 | 0.68  |
ViT-B-16 (ours CLIP 40%)               | 393M | 1.3B | 0.56 | 0.27  | 0.43 | 0.50 | 0.67  |
