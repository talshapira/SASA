# SASA: Source-Aware Self-Attention for IP Hijack Detection

We introduce here a deep learning system that examines the geography of traceroute measurements to detect malicious routes.
We use multiple geolocation services, with various levels of confidence; each also suffers from location errors. 
Moreover, identifying a hijacked route is not sufficient since an operator presented with a hijack alert needs an indication of the cause for flagging out the problematic route. Thus, we introduce a novel deep learning layer, called Source-Aware Self-Attention (SASA), which is an extension of the attention mechanism. SASA learns each data source's confidence and combines this score with the attention of each router in the route to point out the most problematic one.


<p align="center">
<img src='https://github.com/talshapira/SASA/blob/main/figures/SASA%20System%20Arch.png' width="400">
</p>

<p align="center">
<img src='https://github.com/talshapira/SASA/blob/main/figures/IL-RU-DE.PNG' width="400">
</p>

<p align="center">
<img src='https://github.com/talshapira/SASA/blob/main/figures/SASA.png' width="400">
</p>

## Data
Our dataset is [publicly available](https://drive.google.com/drive/folders/1tsVNEcDiufnkGktDkLA8YPWARxNzsZTR?usp=sharing) for researchers:
* dataset_b_noisy_7_agents - including both training and test sets for the training and evaluation of the model.
* use_cases - a few interesting deflection cases (from recent years 2017-2020) that are not part of dataset_b.

## Publications

* T. Shapira and Y. Shavitt, "SASA: Source-Aware Self-Attention for IP Hijack Detection," in IEEE/ACM Transactions on Networking, doi: 10.1109/TNET.2021.3115935. [Download paper here](https://ieeexplore.ieee.org/document/9556519)
