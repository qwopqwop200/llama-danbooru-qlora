# llama-danbooru-qlora
## install
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Or, if you're having trouble with conda, use pip with python3.9:
# pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```
## train
```
python -m torch.distributed.launch --nproc_per_node 8 train.py
```
This was trained for 96 hours(1 epoch) with 8 RTX 3090s.
## model
https://huggingface.co/qwopqwop/danbooru-llama-qlora
