from datasets import load_dataset
from datasets.utils import disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from fairseq.models.roberta import XLMRModel as FairseqXLMRModel
from transformers import AutoTokenizer

# Points to previously converted model dir that must include sentencepiece.bpe.model
hf_tokenizer = AutoTokenizer.from_pretrained("../exported-working")

xlm_v = FairseqXLMRModel.from_pretrained("../xlmv.base")

languages = ['ace', 'af', 'als', 'am', 'an', 
             'ang', 'ar', 'arc', 'arz', 'as', 
             'ast', 'ay', 'az', 'ba', 'bar',
             'bat-smg', 'be', 'be-x-old', 'bg', 'bh',
             'bn', 'bo', 'br', 'bs', 'ca',
             'cbk-zam', 'cdo', 'ce', 'ceb', 'ckb',
             'co', 'crh', 'cs', 'csb', 'cv',
             'cy', 'da', 'de', 'diq', 'dv',
             'el', 'eml', 'en', 'eo', 'es',
             'et', 'eu', 'ext', 'fa', 'fi',
             'fiu-vro', 'fo', 'fr', 'frr',
             'fur', 'fy', 'ga', 'gan', 'gd',
             'gl', 'gn', 'gu', 'hak', 'he',
             'hi', 'hr', 'hsb', 'hu', 'hy',
             'ia', 'id', 'ig', 'ilo', 'io',
             'is', 'it', 'ja', 'jbo', 'jv',
             'ka', 'kk', 'km', 'kn', 'ko',
             'ksh', 'ku', 'ky', 'la', 'lb',
             'li', 'lij', 'lmo', 'ln', 'lt',
             'lv', 'map-bms', 'mg', 'mhr', 'mi',
             'min', 'mk', 'ml', 'mn', 'mr',
             'ms', 'mt', 'mwl', 'my', 'mzn',
             'nap', 'nds', 'ne', 'nl', 'nn',
             'no', 'nov', 'oc', 'or', 'os',
             'pa', 'pdc', 'pl', 'pms', 'pnb',
             'ps', 'pt', 'qu', 'rm', 'ro',
             'ru', 'rw', 'sa', 'sah', 'scn',
             'sco', 'sd', 'sh', 'si', 'simple',
             'sk', 'sl', 'so', 'sq', 'sr',
             'su', 'sv', 'sw', 'szl', 'ta',
             'te', 'tg', 'th', 'tk', 'tl',
             'tr', 'tt', 'ug', 'uk', 'ur',
             'uz', 'vec', 'vep', 'vi', 'vls',
             'vo', 'wa', 'war', 'wuu', 'xmf',
             'yi', 'yo', 'zea', 'zh', 'zh-classical',
             'zh-min-nan', 'zh-yue']

set_verbosity_error()
disable_progress_bar()

for language in languages:
    print(f"Tokenizing language {language}...")
    dataset = load_dataset("wikiann", language)
    
    train_sentences = dataset["train"]
    
    for train_sentence in train_sentences:
        plain_sentence = " ".join(train_sentence["tokens"])
        
        xlm_v_ids = xlm_v.encode(plain_sentence).tolist()
        hf_ids = hf_tokenizer.encode(plain_sentence)
        
        if xlm_v_ids != hf_ids:
            print("-" * 90)
            print(f"Mismatch for {language} sentence:")
            print(plain_sentence)
            print(f"XLM-V ids: {xlm_v_ids}")
            print(f"HF ids: {hf_ids}")
            print("-" * 90)
