# recommended paddle.__version__ == 2.0.0

DET_MODEL_PATH="inference/en_det_infer/"
REC_MODE_PATH="inference/en_rec_infer/"
IMG_PATH="./imgs/"

# 参数解析
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --img) IMG_PATH="$2"; shift ;;
        --det_model) DET_MODEL_PATH="$2"; shift ;;
        --rec_model) REC_MODE_PATH="$2"; shift ;;
        -h|--help)
            echo "用法: --img <path> --model <path> --export <path> [--conf <path>]"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done


# 执行命令
python3 tools/infer/predict_system.py \
  --image_dir="$IMG_PATH" \
  --det_model_dir="${DET_MODEL_PATH}" \
  --rec_model_dir="${REC_MODE_PATH}" \
  --rec_char_dict_path ppocr/utils/en_dict.txt \
  --det_db_unclip_ratio 2.2