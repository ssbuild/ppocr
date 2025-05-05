# recommended paddle.__version__ == 2.0.0
python tools/infer_det.py \
  -c configs/det/PP-OCRv3/PP-OCRv3_mobile_det_my.yml \
  -o rec_model_dir=./checkpoints/best_model/ \
  Global.infer_img=./imgs/
