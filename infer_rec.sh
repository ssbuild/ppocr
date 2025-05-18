# recommended paddle.__version__ == 2.0.0


# 默认值（可选）

MODE="infer"
EXPORT_PATH="inference/en_rec_infer"
CONF_PATH="configs/rec/PP-OCRv4/en_PP-OCRv4_rec_my.yml"
MODEL_PATH="output/rec_ppocr_v4/best_model/model"
IMG_PATH="./imgs/"

# 参数解析
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode ) MODE="$2"; shift ;;
        --export) EXPORT_PATH="$2"; shift ;;
        --conf) CONF_PATH="$2"; shift ;;
        --img) IMG_PATH="$2"; shift ;;
        --model) DET_MODEL="$2"; shift ;;
        -h|--help)
            echo "用法: --img <path> --model <path> --export <path> [--conf <path>]"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done


if [ "${MODE}" = "infer" ]; then
  python tools/infer_rec.py \
    -c "${CONF_PATH}" \
    -o Global.checkpoints="${MODEL_PATH}" \
    Global.infer_img="${IMG_PATH}"

else
  python tools/export_model.py \
    -c "${CONF_PATH}" \
    -o Global.pretrained_model="${MODEL_PATH}" \
        Global.save_inference_dir="${EXPORT_PATH}"
fi



