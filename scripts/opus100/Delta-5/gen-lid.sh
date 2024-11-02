modelfile=~/MoCE/checkpoints/MSHA/opus100/bi-direct/Delta-5-lid
output_dir=~/MoCE/output/MSHA/opus100/bi-direct/Delta-5-lid
lang_pairs="id-en,bs-en,ka-en,es-en,si-en,gu-en,tr-en,ha-en,zh-en,hy-en,kn-en,it-en,dz-en,yo-en,ig-en,or-en,ky-en,uk-en,cy-en,ca-en,sq-en,de-en,mk-en,nl-en,no-en,hi-en,te-en,pa-en,sv-en,ms-en,nb-en,pl-en,az-en,th-en,mn-en,oc-en,ar-en,fr-en,ne-en,ko-en,pt-en,sl-en,bn-en,ja-en,my-en,tk-en,am-en,hr-en,hu-en,et-en,wa-en,da-en,ga-en,li-en,ug-en,ro-en,he-en,se-en,is-en,eo-en,mg-en,ml-en,lv-en,sr-en,el-en,ta-en,mt-en,rw-en,bg-en,ku-en,as-en,km-en,uz-en,fa-en,tg-en,nn-en,an-en,lt-en,ru-en,gd-en,vi-en,sk-en,cs-en,fy-en,af-en,xh-en,br-en,tt-en,ps-en,kk-en,eu-en,be-en,sh-en,gl-en,fi-en,ur-en,zu-en,yi-en,mr-en,en-id,en-bs,en-ka,en-es,en-si,en-gu,en-tr,en-ha,en-zh,en-hy,en-kn,en-it,en-dz,en-yo,en-ig,en-or,en-ky,en-uk,en-cy,en-ca,en-sq,en-de,en-mk,en-nl,en-no,en-hi,en-te,en-pa,en-sv,en-ms,en-nb,en-pl,en-az,en-th,en-mn,en-oc,en-ar,en-fr,en-ne,en-ko,en-pt,en-sl,en-bn,en-ja,en-my,en-tk,en-am,en-hr,en-hu,en-et,en-wa,en-da,en-ga,en-li,en-ug,en-ro,en-he,en-se,en-is,en-eo,en-mg,en-ml,en-lv,en-sr,en-el,en-ta,en-mt,en-rw,en-bg,en-ku,en-as,en-km,en-uz,en-fa,en-tg,en-nn,en-an,en-lt,en-ru,en-gd,en-vi,en-sk,en-cs,en-fy,en-af,en-xh,en-br,en-tt,en-ps,en-kk,en-eu,en-be,en-sh,en-gl,en-fi,en-ur,en-zu,en-yi,en-mr"
# "an" "dz" "yo" "mn" "hy" don't have test set
lang_list=~/MoCE/data/opus100/langs.txt
path_2_data=~/MoCE/data/opus100/by-preprocess
python ~/MoCE/fairseq/scripts/average_checkpoints.py --inputs $modelfile/ --num-epoch-checkpoints 5 --output $modelfile/checkpoint_aver-epoch.pt &
python ~/MoCE/fairseq/scripts/average_checkpoints.py --inputs $modelfile/ --num-update-checkpoints 5 --output $modelfile/checkpoint_aver-update.pt &
wait

mkdir -p $output_dir
mkdir -p $output_dir/raw
mkdir -p $output_dir/tmp
mkdir -p $output_dir/bleus

gen(){
srclang=$2
tgtlang=$4
model=$modelfile/checkpoint$3.pt
summary=$output_dir/tot-bleus$3.txt
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
  gen 0 "tt" $ckpt en &
gen 1 "zh" $ckpt en &
gen 2 "sr" $ckpt en &
gen 3 "si" $ckpt en &
wait
gen 0 "ps" $ckpt en &
gen 1 "is" $ckpt en &
gen 2 "ca" $ckpt en &
gen 3 "li" $ckpt en &
wait
gen 0 "sv" $ckpt en &
gen 1 "hu" $ckpt en &
gen 2 "ml" $ckpt en &
gen 3 "uk" $ckpt en &
wait
gen 0 "ja" $ckpt en &
gen 1 "ar" $ckpt en &
gen 2 "bn" $ckpt en &
gen 3 "pt" $ckpt en &
wait
gen 0 "ro" $ckpt en &
gen 1 "tg" $ckpt en &
gen 2 "am" $ckpt en &
gen 3 "da" $ckpt en &
wait
gen 0 "he" $ckpt en &
gen 1 "ug" $ckpt en &
gen 2 "tr" $ckpt en &
gen 3 "ms" $ckpt en &
wait
gen 0 "fa" $ckpt en &
gen 1 "de" $ckpt en &
gen 2 "rw" $ckpt en &
gen 3 "kk" $ckpt en &
wait
gen 0 "uz" $ckpt en &
gen 1 "ne" $ckpt en &
gen 2 "ur" $ckpt en &
gen 3 "ko" $ckpt en &
wait
gen 0 "eu" $ckpt en &
gen 1 "ga" $ckpt en &
gen 2 "sl" $ckpt en &
gen 3 "kn" $ckpt en &
wait
gen 0 "cy" $ckpt en &
gen 1 "ig" $ckpt en &
gen 2 "fr" $ckpt en &
gen 3 "zu" $ckpt en &
wait
gen 0 "fi" $ckpt en &
gen 1 "br" $ckpt en &
gen 2 "id" $ckpt en &
gen 3 "ha" $ckpt en &
wait
gen 0 "xh" $ckpt en &
gen 1 "fy" $ckpt en &
gen 2 "mk" $ckpt en &
gen 3 "be" $ckpt en &
wait
gen 0 "pa" $ckpt en &
gen 1 "th" $ckpt en &
gen 2 "et" $ckpt en &
gen 3 "mr" $ckpt en &
wait
gen 0 "wa" $ckpt en &
gen 1 "oc" $ckpt en &
gen 2 "gd" $ckpt en &
gen 3 "az" $ckpt en &
wait
gen 0 "or" $ckpt en &
gen 1 "as" $ckpt en &
gen 2 "bg" $ckpt en &
gen 3 "nn" $ckpt en &
wait
gen 0 "sh" $ckpt en &
gen 1 "gu" $ckpt en &
gen 2 "hr" $ckpt en &
gen 3 "lv" $ckpt en &
wait
gen 0 "pl" $ckpt en &
gen 1 "ta" $ckpt en &
gen 2 "sq" $ckpt en &
gen 3 "eo" $ckpt en &
wait
gen 0 "tk" $ckpt en &
gen 1 "nl" $ckpt en &
gen 2 "el" $ckpt en &
gen 3 "se" $ckpt en &
wait
gen 0 "mg" $ckpt en &
gen 1 "no" $ckpt en &
gen 2 "nb" $ckpt en &
gen 3 "sk" $ckpt en &
wait
gen 0 "lt" $ckpt en &
gen 1 "ru" $ckpt en &
gen 2 "ky" $ckpt en &
gen 3 "ka" $ckpt en &
wait
gen 0 "vi" $ckpt en &
gen 1 "hi" $ckpt en &
gen 2 "ku" $ckpt en &
gen 3 "my" $ckpt en &
wait
gen 0 "yi" $ckpt en &
gen 1 "cs" $ckpt en &
gen 2 "bs" $ckpt en &
gen 3 "gl" $ckpt en &
wait
gen 0 "mt" $ckpt en &
gen 1 "es" $ckpt en &
gen 2 "te" $ckpt en &
gen 3 "km" $ckpt en &
wait
gen 0 "af" $ckpt en &
gen 1 "it" $ckpt en &

gen 3 en $ckpt "af" &
gen 2 en $ckpt "it" &
wait
gen 1 en $ckpt "tt" &
gen 2 en $ckpt "zh" &
gen 3 en $ckpt "sr" &
gen 0 en $ckpt "si" &
wait
gen 3 en $ckpt "ps" &
gen 0 en $ckpt "is" &
gen 1 en $ckpt "ca" &
gen 2 en $ckpt "li" &
wait
gen 3 en $ckpt "sv" &
gen 0 en $ckpt "hu" &
gen 1 en $ckpt "ml" &
gen 2 en $ckpt "uk" &
wait
gen 1 en $ckpt "ja" &
gen 2 en $ckpt "ar" &
gen 3 en $ckpt "bn" &
gen 0 en $ckpt "pt" &
wait
gen 3 en $ckpt "ro" &
gen 0 en $ckpt "tg" &
gen 1 en $ckpt "am" &
gen 2 en $ckpt "da" &
wait
gen 3 en $ckpt "he" &
gen 0 en $ckpt "ug" &
gen 1 en $ckpt "tr" &
gen 2 en $ckpt "ms" &
wait
gen 1 en $ckpt "fa" &
gen 2 en $ckpt "de" &
gen 3 en $ckpt "rw" &
gen 0 en $ckpt "kk" &
wait
gen 0 en $ckpt "uz" &
gen 3 en $ckpt "ne" &
gen 1 en $ckpt "ur" &
gen 2 en $ckpt "ko" &
wait
gen 3 en $ckpt "eu" &
gen 0 en $ckpt "ga" &
gen 1 en $ckpt "sl" &
gen 2 en $ckpt "kn" &
wait
gen 1 en $ckpt "cy" &
gen 2 en $ckpt "ig" &
gen 3 en $ckpt "fr" &
gen 0 en $ckpt "zu" &
wait
gen 0 en $ckpt "fi" &
gen 3 en $ckpt "br" &
gen 1 en $ckpt "id" &
gen 2 en $ckpt "ha" &
wait
gen 3 en $ckpt "xh" &
gen 0 en $ckpt "fy" &
gen 2 en $ckpt "mk" &
gen 1 en $ckpt "be" &
wait
gen 1 en $ckpt "pa" &
gen 2 en $ckpt "th" &
gen 3 en $ckpt "et" &
gen 0 en $ckpt "mr" &
wait
gen 0 en $ckpt "wa" &
gen 3 en $ckpt "oc" &
gen 1 en $ckpt "gd" &
gen 2 en $ckpt "az" &
wait
gen 3 en $ckpt "or" &
gen 0 en $ckpt "as" &
gen 1 en $ckpt "bg" &
gen 2 en $ckpt "nn" &
wait
gen 1 en $ckpt "sh" &
gen 2 en $ckpt "gu" &
gen 3 en $ckpt "hr" &
gen 0 en $ckpt "lv" &
wait
gen 3 en $ckpt "pl" &
gen 0 en $ckpt "ta" &
gen 1 en $ckpt "sq" &
gen 2 en $ckpt "eo" &
wait
gen 3 en $ckpt "tk" &
gen 0 en $ckpt "nl" &
gen 1 en $ckpt "el" &
gen 2 en $ckpt "se" &
wait
gen 1 en $ckpt "mg" &
gen 2 en $ckpt "no" &
gen 3 en $ckpt "nb" &
gen 0 en $ckpt "sk" &
wait
gen 0 en $ckpt "lt" &
gen 3 en $ckpt "ru" &
gen 1 en $ckpt "ky" &
gen 2 en $ckpt "ka" &
wait
gen 3 en $ckpt "vi" &
gen 0 en $ckpt "hi" &
gen 1 en $ckpt "ku" &
gen 2 en $ckpt "my" &
wait
gen 1 en $ckpt "yi" &
gen 2 en $ckpt "cs" &
gen 3 en $ckpt "bs" &
gen 0 en $ckpt "gl" &
wait
gen 0 en $ckpt "mt" &
gen 3 en $ckpt "es" &
gen 1 en $ckpt "te" &
gen 2 en $ckpt "km" &
wait
python ~/MoCE/tools/avg-bleus.py $output_dir/tot-bleus$ckpt.txt >> $output_dir/tot-bleus$ckpt.txt
done
