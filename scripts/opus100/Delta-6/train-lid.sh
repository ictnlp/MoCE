ulimit -n 4096
DATA=~/MoCE/data/opus100/by-preprocess
lang_pairs="id-en,bs-en,ka-en,es-en,si-en,gu-en,tr-en,ha-en,zh-en,hy-en,kn-en,it-en,dz-en,yo-en,ig-en,or-en,ky-en,uk-en,cy-en,ca-en,sq-en,de-en,mk-en,nl-en,no-en,hi-en,te-en,pa-en,sv-en,ms-en,nb-en,pl-en,az-en,th-en,mn-en,oc-en,ar-en,fr-en,ne-en,ko-en,pt-en,sl-en,bn-en,ja-en,my-en,tk-en,am-en,hr-en,hu-en,et-en,wa-en,da-en,ga-en,li-en,ug-en,ro-en,he-en,se-en,is-en,eo-en,mg-en,ml-en,lv-en,sr-en,el-en,ta-en,mt-en,rw-en,bg-en,ku-en,as-en,km-en,uz-en,fa-en,tg-en,nn-en,an-en,lt-en,ru-en,gd-en,vi-en,sk-en,cs-en,fy-en,af-en,xh-en,br-en,tt-en,ps-en,kk-en,eu-en,be-en,sh-en,gl-en,fi-en,ur-en,zu-en,yi-en,mr-en,en-id,en-bs,en-ka,en-es,en-si,en-gu,en-tr,en-ha,en-zh,en-hy,en-kn,en-it,en-dz,en-yo,en-ig,en-or,en-ky,en-uk,en-cy,en-ca,en-sq,en-de,en-mk,en-nl,en-no,en-hi,en-te,en-pa,en-sv,en-ms,en-nb,en-pl,en-az,en-th,en-mn,en-oc,en-ar,en-fr,en-ne,en-ko,en-pt,en-sl,en-bn,en-ja,en-my,en-tk,en-am,en-hr,en-hu,en-et,en-wa,en-da,en-ga,en-li,en-ug,en-ro,en-he,en-se,en-is,en-eo,en-mg,en-ml,en-lv,en-sr,en-el,en-ta,en-mt,en-rw,en-bg,en-ku,en-as,en-km,en-uz,en-fa,en-tg,en-nn,en-an,en-lt,en-ru,en-gd,en-vi,en-sk,en-cs,en-fy,en-af,en-xh,en-br,en-tt,en-ps,en-kk,en-eu,en-be,en-sh,en-gl,en-fi,en-ur,en-zu,en-yi,en-mr"
lang_list=~/MoCE/data/opus100/langs.txt
savedir=~/MoCE/checkpoints/MSHA/opus100/bi-direct/Delta-6-lid

mkdir -p $savedir

TOTAL_NUM_UPDATES=1500000
WARMUP_UPDATES=4000
LR=5e-04
MAX_TOKENS=16384
UPDATE_FREQ=2

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup fairseq-train $DATA \
	--max-tokens $MAX_TOKENS \
	--task translation_multi_simple_epoch \
    --sampling-method "temperature" \
    --sampling-temperature 1.5 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" \
    --max-target-positions 4096 --max-source-positions 4096 \
	--truncate-source --share-all-embeddings \
	--ddp-backend=legacy_ddp \
	--share-decoder-input-output-embed \
	--conv-kernels "0 1 3 5 7 9 11" \
	--required-batch-size-multiple 1 \
	--arch adaptive_multiscale_head_transformer --criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 \
	--dropout 0.1 \
	--patience 10 \
	--token-level-adaptive \
	--langid-expert \
	--ms-layers 1 \
	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
	--save-interval 1 --keep-interval-updates 20 --save-interval-updates 5000 \
	--seed 222 \
	--log-format simple --log-interval 100 \
	--clip-norm 0.0 \
	--lr-scheduler inverse_sqrt --lr $LR \
	--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
	--fp16 --update-freq $UPDATE_FREQ \
	--skip-invalid-size-inputs-valid-test \
	--valid-subset valid \
	--save-dir $savedir | tee -a $savedir/log.out &
	# --eval-bleu \
	# --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
	# --eval-bleu-detok moses \
	# --eval-bleu-remove-bpe \
	# --eval-bleu-print-samples \
	# --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
