
# CS/SD Exact Integration (Paper-faithful)

## One-time setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# ensure Vicuna 7B is accessible (see ckpt/pretrained_ckpt/prepare_vicuna.md)
```

## Place encoder checkpoints (encoders only)
```
ckpt/styletts2_encoders/  text_aco_encoder.pt  text_bert_encoder.pt  reference_encoder.pt
ckpt/dreamtalk_encoders/  audio_encoder.pt     style_encoder.pt
```

## Run the four stages
```bash
bash scripts/train_cs_sd_all.sh
```
Outputs saved in `runs/`.
