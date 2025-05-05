# recommended paddle.__version__ == 2.0.0

python tools/infer_det.py \
  -c configs/det/PP-OCRv3/PP-OCRv3_mobile_det_my.yml \
  -o Global.checkpoints=./checkpoints/detv3/best_model/model \
  Global.infer_img=./imgs/