path='C:/Users/s/OneDrive/doc/course/ML/exp2/'
train_data='model/pkl/train/'
ckpt_dir='model/ckPoint/'
datasets=`ls $path$train_data`
ckpt=`ls $path$ckpt_dir`
# for file in $datasets
# do
#     ckpt=`ls $path$ckpt_dir`
#     for ck in $ckpt
#     do
#         python ./train.py --workbash=$path --ds_name=$file --ckpt=$ck >> "log"
#         rm -f $path$ckpt_dir$ck
#     break
#     done
    
# # python ./test.py --workbash=$path --test_ds=$file --ckpt=$ck
# done

# echo $ckpt
# if [ -f "$path$ckpt_dir$ckpt" ];then 
# echo 'success!'
# fi
# # python ./train.py
# # sh run.sh
# for i in $ckpt 
# do 
# echo $i 
# break
# done


ckpt=`ls $path$ckpt_dir`
for ck in $ckpt
do
    python ./test.py --workbash=$path --test_ds='test.pkl' --ckpt=$ck
break
done
