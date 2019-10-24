# Model Implementation of PACS and ImageNet-Sketch Experiments with comparison of JigenDG

To eliminate the effects of factors other than models such as optimization schedule, data augmentation, training-test split and improve the reproducibility, 
we implement PAR over the backbone of JigenDG repo. We refer the users to download the pretrained model and look for training details at 
https://github.com/fmcarlucci/JigenDG.

### Experiments

Before running the experiments, please update the `data_path` in `data/PARLoader.py`. 
Then follow *run_PACS_PAR.sh* or *run_ImageNet_PAR.sh* to run the experiments on PACS and ImageNet-Sketch datasets.

