dict=dict.byte
destdir=~/MoCE/data/opus100/by-preprocess
mkdir -p $destdir
# for lang in "an" "tt" "zh" "sr" "si" "ps" "is" "ca" "li" "sv" "hu" "ml" "uk" "ja" "ar" "bn" "pt" "ro" "tg" "am" "da" "he" "ug" "tr" "ms" "fa" "de" "rw" "kk" "uz" "ne" "ur" "ko" "eu" "ga" "sl" "kn" "cy" "ig" "fr" "zu" "fi" "br" "id" "dz" "ha" "xh" "fy" "mk" "be" "pa" "yo" "th" "et" "mr" "wa" "oc" "gd" "az" "or" "as" "bg" "nn" "sh" "gu" "hr" "lv" "pl" "ta" "sq" "eo" "tk" "nl" "el" "se" "mg" "no" "nb" "sk" "lt" "ru" "ky" "ka" "vi" "hi" "ku" "my" "mn" "yi" "cs" "bs" "gl" "mt" "es" "te" "km" "hy" "af" "it"
for lang in "tt" "zh" "sr" "si" "ps" "is" "ca" "li" "sv" "hu" "ml" "uk" "ja" "ar" "bn" "pt" "ro" "tg" "am" "da" "he" "ug" "tr" "ms" "fa" "de" "rw" "kk" "uz" "ne" "ur" "ko" "eu" "ga" "sl" "kn" "cy" "ig" "fr" "zu" "fi" "br" "id" "ha" "xh" "fy" "mk" "be" "pa" "th" "et" "mr" "wa" "oc" "gd" "az" "or" "as" "bg" "nn" "sh" "gu" "hr" "lv" "pl" "ta" "sq" "eo" "tk" "nl" "el" "se" "mg" "no" "nb" "sk" "lt" "ru" "ky" "ka" "vi" "hi" "ku" "my" "yi" "cs" "bs" "gl" "mt" "es" "te" "km" "af" "it"
do
        echo $lang
        fairseq-preprocess \
                --source-lang $lang \
            --target-lang en \
                --trainpref supervised/$lang-en/opus.$lang-en-train \
                --validpref supervised/$lang-en/opus.$lang-en-dev \
                --testpref supervised/$lang-en/opus.$lang-en-test \
                --destdir $destdir \
                --srcdict $dict \
                --tgtdict $dict \
                --workers 32 \
                --byte-tokens
                echo "finish $lang"
done

# The following 5 languages don't have valid and test set.
for lang in "an" "dz" "yo" "mn" "hy"
do
        echo $lang
        fairseq-preprocess \
                --source-lang $lang \
            --target-lang en \
                --trainpref supervised/$lang-en/opus.$lang-en-train \
                --destdir $destdir \
                --srcdict $dict \
                --tgtdict $dict \
                --workers 32 \
                --byte-tokens
                echo "finish $lang"
done