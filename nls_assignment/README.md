# Nobile Neural Fields

## Installation

```bash
cd nls_assignment
conda create -n nls python=3.10 -y
conda activate nls
pip install -r requirements.txt
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Train an NLS model

```bash
export SCAN_FOLDER=processed_2025_03_06_15_45_13-temp4
python3 train.py --data_path data/$SCAN_FOLDER/frame_bundle.npz --name $SCAN_FOLDER --num_batches 50 --max_epochs 50 --point_batch_size 25600
```