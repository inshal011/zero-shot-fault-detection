### Table 1 — Zero-shot LOMO results (combined score)

| target   | sources           |    auc |   pauc_0.1 |
|:---------|:------------------|-------:|-----------:|
| fan      | pump+slider+valve | 0.454  |     0.4955 |
| pump     | fan+slider+valve  | 0.5834 |     0.5134 |
| slider   | fan+pump+valve    | 0.7661 |     0.5735 |
| valve    | fan+pump+slider   | 0.4398 |     0.4967 |


### Table 2 — Effect of domain-adversarial head (AUC)

| target   |    False |   True |
|:---------|---------:|-------:|
| ToyCar   | nan      | 0.6672 |
| fan      |   0.4629 | 0.454  |
| pump     |   0.5885 | 0.5834 |
| slider   | nan      | 0.7661 |
| valve    | nan      | 0.4502 |


### Table 3 — Scoring-function ablation (AUC)

| target   |   combined |   maha |   recon |
|:---------|-----------:|-------:|--------:|
| ToyCar   |     0.6672 | 0.6454 |  0.658  |
| fan      |     0.454  | 0.4616 |  0.4484 |
| pump     |     0.5834 | 0.5597 |  0.5976 |
| slider   |     0.7661 | 0.6788 |  0.8058 |
| valve    |     0.4502 | 0.4562 |  0.4422 |


### Table 4 — Source-domain count vs. generalization

|   n_sources | sources         |    auc |   pauc_0.1 |
|------------:|:----------------|-------:|-----------:|
|           2 | fan+pump        | 0.4606 |     0.5003 |
|           3 | fan+pump+slider | 0.4398 |     0.4967 |