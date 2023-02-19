# :<<!

for ((i=$1; i<=$2; i++))
do
  for f in $(ls -h /home/arthur/workspace/Datasets/MIPI2022/validation_input/)
  do
      echo "${i} => ${f}"
      python3 -u infer_emdc.py \
      --cfg_path ../config/cfg_mipi_emdc_fm_nml.py \
      --ckp_path ./../EXP_EMDC_FM_NML/epoch_${i}.pth \
      --txt_path /home/arthur/workspace/Datasets/MIPI2022/validation_input/${f}/data.list \
      --data_type ${f} \
      --vis false

  done
  python3 -u evaluate.py ${i} false
done
 !