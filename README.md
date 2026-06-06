# CGSF: Image Super-Resolution

## Installation

**Dependencies**

```
torch 2.4.1
basicsr 1.4.2
```

**Setup**

```bash
pip install basicsr==1.4.2
python setup.py develop
pip install numpy==1.24.4
pip install -v -e .
```

---

## Data Preparation

**Training set**: First 800 images of the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset.  
**Test sets**: Set5 / Set14 / B100 / Urban100 / Manga109

```
├── dataset
│   ├── DIV2K_decoded              # Training data 
│   │   ├── DIV2K_train_HR
│   │   └── DIV2K_train_LR_bicubic
│   └── benchmark                  # Test data 
│       ├── Set5
│       ├── Set14
│       ├── B100
│       ├── Urban100
│       └── Manga109
```

---

## Training

```bash
# x2
torchrun --nproc_per_node=$GPU_NUM$ basicsr/train.py -opt options/train/cgsfx2.yml --launcher pytorch

# x3
torchrun --nproc_per_node=$GPU_NUM$ basicsr/train.py -opt options/train/cgsfx3.yml --launcher pytorch

# x4
torchrun --nproc_per_node=$GPU_NUM$ basicsr/train.py -opt options/train/cgsfx4.yml --launcher pytorch
```

> Replace `$GPU_NUM$` with the number of GPUs to use.

---

## Testing

```bash
# x2
python test.py -opt options/test/cgsfx2.yaml

# x3
python test.py -opt options/test/cgsfx3.yaml

# x4
python test.py -opt options/test/cgsfx4.yaml
```
