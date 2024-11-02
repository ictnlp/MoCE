dict=dict.byte
destdir=~/MoCE/data/ted59/by-preprocess
mkdir -p $destdir
for lang in "pt" "sr" "cs" "zhcn" "fr" "nl" "uk" "et" "he" "th" "ru" "de" "ro" "bs" "sv" "ku" "el" "ja" "nb" "bg" "ar" "ur" "zh" "ms" "be" "sl" "eu" "hu" "ko" "mr" "it" "es" "fa" "hy" "lt" "ka" "bn" "hi" "mk" "zhtw" "pl" "vi" "sq" "sk" "kk" "gl" "ptbr" "my" "ta" "tr" "eo" "hr" "da" "mn" "fi" "frca" "az" "id"
do
        fairseq-preprocess \
                --source-lang $lang \
            --target-lang en \
                --trainpref raw/train/train.$lang-en \
                --validpref raw/dev/dev.$lang-en \
                --testpref raw/test/test.$lang-en \
                --destdir $destdir \
                --srcdict $dict \
                --tgtdict $dict \
                --workers 32 \
                --byte-tokens
                echo "finish $lang"
done