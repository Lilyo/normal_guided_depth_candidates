# :<<!

for ((i=$1; i<=$2; i++))
do
  for f in $(ls -h /home/arthur/workspace/Datasets/MIPI2022/test_input/)
  do
      echo "${i} => ${f}"
      python3 -u infer_emdc_test.py \
      --cfg_path ../config/cfg_mipi_emdc_fm_nml.py \
      --ckp_path ./../EXP_EMDC_FM_NML_superout_0.8561/epoch_${i}.pth \
      --txt_path /home/arthur/workspace/Datasets/MIPI2022/test_input/${f}/data.list \
      --data_type ${f} \
      --vis True

  done
#  python3 -u evaluate.py ${i} false
done
 !