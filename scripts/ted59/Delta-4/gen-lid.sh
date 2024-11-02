modelfile=~/MoCE/checkpoints/MSHA/ted59/bi-direct/Delta-4-lid
output_dir=~/MoCE/output/MSHA/ted59/bi-direct/Delta-4-lid
lang_pairs="pt-en,ar-en,ms-en,th-en,lt-en,zhcn-en,da-en,hi-en,sv-en,et-en,hr-en,eo-en,he-en,ko-en,ja-en,ku-en,gl-en,es-en,mn-en,fr-en,el-en,ta-en,tr-en,sq-en,ptbr-en,ro-en,eu-en,frca-en,hy-en,ur-en,fi-en,my-en,cs-en,bg-en,mr-en,de-en,vi-en,sl-en,ka-en,sk-en,nl-en,be-en,zhtw-en,bn-en,uk-en,nb-en,az-en,bs-en,zh-en,it-en,ru-en,mk-en,sr-en,hu-en,pl-en,id-en,kk-en,fa-en,en-pt,en-ar,en-ms,en-th,en-lt,en-zhcn,en-da,en-hi,en-sv,en-et,en-hr,en-eo,en-he,en-ko,en-ja,en-ku,en-gl,en-es,en-mn,en-fr,en-el,en-ta,en-tr,en-sq,en-ptbr,en-ro,en-eu,en-frca,en-hy,en-ur,en-fi,en-my,en-cs,en-bg,en-mr,en-de,en-vi,en-sl,en-ka,en-sk,en-nl,en-be,en-zhtw,en-bn,en-uk,en-nb,en-az,en-bs,en-zh,en-it,en-ru,en-mk,en-sr,en-hu,en-pl,en-id,en-kk,en-fa"
lang_list=~/MoCE/data/ted59/langs.txt
path_2_data=~/MoCE/data/ted59/by-preprocess
python ~/MoCE/fairseq/scripts/average_checkpoints.py --inputs $modelfile/ --num-update-checkpoints 5 --output $modelfile/checkpoint_aver-update.pt &
python ~/MoCE/fairseq/scripts/average_checkpoints.py --inputs $modelfile/ --num-epoch-checkpoints 5 --output $modelfile/checkpoint_aver-epoch.pt &
wait

mkdir -p $output_dir
mkdir -p $output_dir/raw
mkdir -p $output_dir/tmp
mkdir -p $output_dir/bleus

gen(){
srclang=$2
tgtlang=$4
summary=$output_dir/tot-bleus$3.txt
model=$modelfile/checkpoint$3.pt
CUDA_VISIBLE_DEVICES=$1 fairseq-generate $path_2_data \
  --path $model \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang $srclang \
  --target-lang $tgtlang \
  --batch-size 128 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lenpen 1.5 \
  --skip-invalid-size-inputs-valid-test \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" > $output_dir/raw/pred.$srclang-$tgtlang.$3.txt

cat $output_dir/raw/pred.$srclang-$tgtlang.$3.txt | grep -P "^H" |sort -V |cut -f 3- |cat > $output_dir/tmp/$srclang-$tgtlang.$3.hyp 
cat $output_dir/raw/pred.$srclang-$tgtlang.$3.txt | grep -P "^T" |sort -V |cut -f 2- |cat > $output_dir/tmp/$srclang-$tgtlang.$3.ref


sacrebleu $output_dir/tmp/$srclang-$tgtlang.$3.ref --metrics bleu -w 3 --tokenize 13a < $output_dir/tmp/$srclang-$tgtlang.$3.hyp > $output_dir/bleus/$srclang-$tgtlang.$3.bytebleu

python ~/MoCE/tools/encode-byte.py $output_dir/tmp/$srclang-$tgtlang.$3.hyp $output_dir/tmp/$srclang-$tgtlang.$3.str.hyp
python ~/MoCE/tools/encode-byte.py $output_dir/tmp/$srclang-$tgtlang.$3.ref $output_dir/tmp/$srclang-$tgtlang.$3.str.ref

# sed -r 's/(@@ )|(@@ ?$)//g' < $output_dir/tmp/$srclang-$tgtlang.$3.str.hyp > $output_dir/tmp/$srclang-$tgtlang.$3.debpe.hyp
# sed -r 's/(@@ )|(@@ ?$)//g' < $output_dir/tmp/$srclang-$tgtlang.$3.str.ref > $output_dir/tmp/$srclang-$tgtlang.$3.debpe.ref

sacrebleu $output_dir/tmp/$srclang-$tgtlang.$3.str.ref --metrics bleu -w 3 --tokenize 13a < $output_dir/tmp/$srclang-$tgtlang.$3.str.hyp > $output_dir/bleus/$srclang-$tgtlang.$3.bleu

python ~/MoCE/tools/get-bleus.py $srclang-$tgtlang$3 $output_dir/bleus/$srclang-$tgtlang.$3.bleu >> $summary
}

for ckpt in _aver-epoch _aver-update _best _last
do
gen 0 'pt' $ckpt en &
gen 1 'ar' $ckpt en &
gen 2 'ms' $ckpt en &
gen 3 'th' $ckpt en &
wait
gen 0 'lt' $ckpt en &
gen 1 'zhcn' $ckpt en &
gen 2 'da' $ckpt en &
gen 3 'hi' $ckpt en &
wait
gen 0 'sv' $ckpt en &
gen 1 'et' $ckpt en &
gen 2 'hr' $ckpt en &
gen 3 'eo' $ckpt en &
wait
gen 0 'he' $ckpt en &
gen 1 'ko' $ckpt en &
gen 2 'ja' $ckpt en &
gen 3 'ku' $ckpt en &
wait
gen 0 'gl' $ckpt en &
gen 1 'es' $ckpt en &
gen 2 'mn' $ckpt en &
gen 3 'fr' $ckpt en &
wait
gen 0 'el' $ckpt en &
gen 1 'ta' $ckpt en &
gen 2 'tr' $ckpt en &
gen 3 'sq' $ckpt en &
wait
gen 0 'ptbr' $ckpt en &
gen 1 'ro' $ckpt en &
gen 2 'eu' $ckpt en &
gen 3 'frca' $ckpt en &
wait
gen 0 'hy' $ckpt en &
gen 1 'ur' $ckpt en &
gen 2 'fi' $ckpt en &
gen 3 'my' $ckpt en &
wait
gen 0 'cs' $ckpt en &
gen 1 'bg' $ckpt en &
gen 2 'mr' $ckpt en &
gen 3 'de' $ckpt en &
wait
gen 0 'vi' $ckpt en &
gen 1 'sl' $ckpt en &
gen 2 'ka' $ckpt en &
gen 3 'sk' $ckpt en &
wait
gen 0 'nl' $ckpt en &
gen 1 'be' $ckpt en &
gen 2 'zhtw' $ckpt en &
gen 3 'bn' $ckpt en &
wait
gen 0 'uk' $ckpt en &
gen 1 'nb' $ckpt en &
gen 2 'az' $ckpt en &
gen 3 'bs' $ckpt en &
wait
gen 0 'zh' $ckpt en &
gen 1 'it' $ckpt en &
gen 2 'ru' $ckpt en &
gen 3 'mk' $ckpt en &
wait
gen 0 'sr' $ckpt en &
gen 1 'hu' $ckpt en &
gen 2 'pl' $ckpt en &
gen 3 'id' $ckpt en &
wait
gen 0 'kk' $ckpt en &
gen 1 'fa' $ckpt en &

gen 2 en $ckpt 'kk' &
gen 3 en $ckpt 'fa' &
wait
gen 0 en $ckpt 'pt' &
gen 1 en $ckpt 'ar' &
gen 2 en $ckpt 'ms' &
gen 3 en $ckpt 'th' &
wait
gen 0 en $ckpt 'lt' &
gen 1 en $ckpt 'zhcn' &
gen 2 en $ckpt 'da' &
gen 3 en $ckpt 'hi' &
wait
gen 0 en $ckpt 'sv' &
gen 1 en $ckpt 'et' &
gen 2 en $ckpt 'hr' &
gen 3 en $ckpt 'eo' &
wait
gen 0 en $ckpt 'he' &
gen 1 en $ckpt 'ko' &
gen 2 en $ckpt 'ja' &
gen 3 en $ckpt 'ku' &
wait
gen 0 en $ckpt 'gl' &
gen 1 en $ckpt 'es' &
gen 2 en $ckpt 'mn' &
gen 3 en $ckpt 'fr' &
wait
gen 0 en $ckpt 'el' &
gen 1 en $ckpt 'ta' &
gen 2 en $ckpt 'tr' &
gen 3 en $ckpt 'sq' &
wait
gen 0 en $ckpt 'ptbr' &
gen 1 en $ckpt 'ro' &
gen 2 en $ckpt 'eu' &
gen 3 en $ckpt 'frca' &
wait
gen 0 en $ckpt 'hy' &
gen 1 en $ckpt 'ur' &
gen 2 en $ckpt 'fi' &
gen 3 en $ckpt 'my' &
wait
gen 0 en $ckpt 'cs' &
gen 1 en $ckpt 'bg' &
gen 2 en $ckpt 'mr' &
gen 3 en $ckpt 'de' &
wait
gen 0 en $ckpt 'vi' &
gen 1 en $ckpt 'sl' &
gen 2 en $ckpt 'ka' &
gen 3 en $ckpt 'sk' &
wait
gen 0 en $ckpt 'nl' &
gen 1 en $ckpt 'be' &
gen 2 en $ckpt 'zhtw' &
gen 3 en $ckpt 'bn' &
wait
gen 0 en $ckpt 'uk' &
gen 1 en $ckpt 'nb' &
gen 2 en $ckpt 'az' &
gen 3 en $ckpt 'bs' &
wait
gen 0 en $ckpt 'zh' &
gen 1 en $ckpt 'it' &
gen 2 en $ckpt 'ru' &
gen 3 en $ckpt 'mk' &
wait
gen 0 en $ckpt 'sr' &
gen 1 en $ckpt 'hu' &
gen 2 en $ckpt 'pl' &
gen 3 en $ckpt 'id' &
wait
python ~/MoCE/tools/avg-bleus.py $output_dir/tot-bleus$ckpt.txt >> $output_dir/tot-bleus$ckpt.txt
done
