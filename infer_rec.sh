# recommended paddle.__version__ == 2.0.0

python tools/infer_rec.py \
  -c configs/det/PP-OCRv4/en_PP-OCRv4_rec_my.yml \
  -o Global.checkpoints=./checkpoints/recv4/best_model/model \
  Global.infer_img=./imgs/
