from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from fastwarc.warc import ArchiveIterator, WarcRecordType
import fasttext
import regex
import nltk
import os
import mmh3
import random

EMAIL_PATTERN = regex.compile(
    r"""
    (?<![A-Za-z0-9._%+-])
    (?:[A-Za-z0-9](?:[A-Za-z0-9._%+-]{0,62}[A-Za-z0-9])?)
    @
    (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+
    [A-Za-z]{2,63}
    (?![A-Za-z0-9._%+-])
    """,
    regex.VERBOSE | regex.IGNORECASE,
)

PHONE_NUMBER_PATTERN = regex.compile(
    r"""
    (?<!\d)                    # left boundary: not part of a longer digit sequence
    (?:\+?1[\s.-]?)?           # optional country code
    (?:\(\s*\d{3}\s*\)|\d{3})  # area code with/without parentheses
    [\s.-]?                    # optional separator
    \d{3}
    [\s.-]?                    # optional separator
    \d{4}
    (?!\d)                     # right boundary
    """,
    regex.VERBOSE,
)

# in this function, we will extract text from html 
# after that, we will use it to extract text from warc files 
# and see the text with the wet files 
# notice that the input is a byte string containing the html content 

def extract_text_from_html(html):
    # first we should convert the byte string to a regular string 
    # we need to decode it , use resiliparse.parse.encoding.detect_encoding() to detect the encoding of the html content 
    try:
        encoding = detect_encoding(html)
        html_str = html.decode(encoding)
    except:
        # if we can't decode it , we can just decode it with utf-8 and ignore the errors 
        html_str = html.decode("utf-8", errors="ignore")

    # now we can use resiliparse.extract.html2text.extract_plain_text() to extract the text from the html string 
    text = extract_plain_text(html_str)
    return text 

def extract_text_from_example_warc():
    # we will use the extract function to extract text from the raw warc file 
    # and compare the output with the text in the wet file 
    warc_file = "./CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    # we should convert the warc file to a byte string 
    # oh,shit. The warc_file is gzip file, we can't directly read it as b mode 
    # must uzip it .
    # we can use ArchiveIterator to look throught each record 
    
    with open(warc_file, "rb") as f:
        for record in ArchiveIterator(f,WarcRecordType.response):
            bytes = record.reader.read()
            text = extract_text_from_html(bytes)
            print(text)


# in this function, we will use fasttext's language classification model to classfiy the input string 
def language_identify(text):

    model = fasttext.load_model("./cs336_data/lid.176.bin")
    # remove the \n from text, because fasttext.predict seems not support multi lines input 
    # so we must convert \n to " " to one line 
    text = text.replace("\n", " ")
    results = model.predict(text,3)
    #print(results)
    return (results[0][0][9:],results[1][0])

def run_language_identity_on_twenty():
    # in this function, we will randomly select 20 texts from 
    # the text that we extracted from the warc file and use the language_identify function to 
    # identify the language of the text and print the results 
    # we should compare the results with the actual language of the text 
    # and compute the radio of English in 20 texts
    # and compute the average score of text to judge the 置信度 of the language classification

    # 目前来看比较棘手的一点是如何随机取20个文本，因为这个warc文本很大，而且有多少个我们也不知道
    # 一种想法是顺序读取一遍，存储后利用列表随机抽取20个，但是这样可能会内存爆炸
    # 还有一种是利用蓄水池抽样算法，这样边读取边随机，不需要存储所有的文本！
    warc_file = "./CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    english_count = 0
    total_score = 0
    total_count = 0
    model = fasttext.load_model("./lid.176.bin")
    sample_size = 20

    text_pool = [] # 20 samples 

    with open(warc_file, "rb") as f:
        for record in ArchiveIterator(f,WarcRecordType.response):

            # 有20/n的概率替换掉text_pool中的一个文本，否则就丢弃这个文本
            if len(text_pool) < sample_size:
                text_pool.append(record)
                record.freeze()
            else:
                import random 
                if random.random() < sample_size/(total_count+1):
                    text_pool[random.randint(0,sample_size-1)] = record
                    record.freeze()
            total_count += 1 
    
    for record in text_pool:
        bytes = record.reader.read()
        text = extract_text_from_html(bytes)
        text = text.replace("\n", " ")
        results = model.predict(text,3)
        language , score = results[0][0][9:],results[1][0]
        print(f"The content is : {text} \nThe language is : {language} with score {score}\n")
        if language == 'en':
            english_count += 1
        total_score += score

    print(f"The ratio of English in {sample_size} texts is : {english_count/sample_size}")
    print(f"The average score of text is : {total_score/sample_size}")
            
def mask_emails(text):
    # in this function, we will use regular expression to mask the emails in the input string 
    # and return a pair (masked_text, count of masked emails)
    # the question is that how to find the emails in the text 
    # use some regular expression to find ? 
    # such as ([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+
    # we will mask it with |||EMAIL_ADDRESS|||
    if "@" not in text:
        return text, 0
    masked_text , count = EMAIL_PATTERN.subn("|||EMAIL_ADDRESS|||",text,timeout=5)
    return masked_text , count

def mask_phonenumber(text):
    # in this function, we will use regular expression to mask the phone numbers in the input string 
    # and return a pair (masked_text, count of masked phone numbers)
    # the question is there is so many formats of phone numbers ?
    # in the test , we just need to make sure mask the United States phone numbers 
    # use ^(\\\\+?1)?[2-9]\\\\d{2}[2-9](?!11)\\\\d{6}\\$") 
    masked_text , count = PHONE_NUMBER_PATTERN.subn("|||PHONE_NUMBER|||",text,timeout=5)
    return masked_text , count

def mask_ips(text):
    # we only care about IPv4 address
    pattern = regex.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    masked_text, count = pattern.subn("|||IP_ADDRESS|||", text,timeout=5)
    return masked_text, count

def run_PII_masking_on_twenty():
    # in this function, we will randomly select 20 texts from 
    # the text that we extracted from the warc file and use the PII masking functions to 
    # mask the emails, phone numbers and ips in the text and print the results 
    # we should compare the results with the actual text and compute the average count of masked emails, phone numbers and ips in 20 texts

    warc_file = "./CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
 
    total_count = 0

    sample_size = 20
    text_pool = [] # 20 samples (store text snapshot directly)

    with open(warc_file, "rb") as f:
        for record in ArchiveIterator(f,WarcRecordType.response):

            # first make sure the record has maskable PII
            flag = False
            bytes = record.reader.read()
            text = extract_text_from_html(bytes)
            masked_text , email_count = mask_emails(text)
            masked_text , phone_count = mask_phonenumber(masked_text)
            masked_text , ip_count = mask_ips(masked_text)
            if email_count > 0 or phone_count > 0 or ip_count > 0:
                flag = True
            if flag:
                if len(text_pool) < sample_size:
                    text_pool.append(text)
                else:
                    import random 
                    if random.random() < sample_size/(total_count+1):
                        text_pool[random.randint(0,sample_size-1)] = text
                total_count += 1 
    
    for text in text_pool:

        # we add a time constraint 

        
        masked_text , email_count = mask_emails(text)
        masked_text , phone_count = mask_phonenumber(masked_text)
        masked_text , ip_count = mask_ips(masked_text)

        print(f"The original content is : {text} \nThe masked content is : {masked_text}\n")
        print(f"The nums of masked emails is : {email_count} \nThe nums of masked phone numbers is : {phone_count} \nThe nums of masked ips is : {ip_count}\n")
    

# now we will add two harmful content classification function : classify nsfw and toxic speech 
def classify_nsfw(text):
    # text is a string and we will continue use fasttext to classify the nsfw content in the text 
    # return a pair (label, score) 
    model = fasttext.load_model("/home/zz/Code/cs336_hw4/cs336_data/jigsaw_fasttext_bigrams_nsfw_final.bin")
    results = model.predict(text,1)
    return (results[0][0][9:],results[1][0])
    

def classify_toxic_speech(text):
    # the same as toxic speech classification 
    model = fasttext.load_model("/home/zz/Code/cs336_hw4/cs336_data/jigsaw_fasttext_bigrams_hatespeech_final.bin")
    results = model.predict(text,1)
    return (results[0][0][9:],results[1][0])

def run_toxic_classification_on_twenty():

    warc_file = "./CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    total_count = 0
    sample_size = 40
    text_pool = [] # 20 samples 

    with open(warc_file, "rb") as f:
        for record in ArchiveIterator(f,WarcRecordType.response):

            if len(text_pool) < sample_size:
                text_pool.append(record)
                record.freeze()
            else:
                import random 
                if random.random() < sample_size/(total_count+1):
                    text_pool[random.randint(0,sample_size-1)] = record
                    record.freeze()
            total_count += 1
    
    for record in text_pool:
        bytes = record.reader.read()
        text = extract_text_from_html(bytes)
        # remove \n from text
        text = text.replace("\n", " ")
        label1, score1 = classify_nsfw(text)
        print(f"The content is : {text} \nThe NSFW classification result is : {label1} with score {score1}\n")
        label2, score2 = classify_toxic_speech(text)
        print(f"The content is : {text} \nThe toxic speech classification result is : {label2} with score {score2}\n")

def gopher_quality_filters(text):
    """
    • Contain less than 50 or more than 100,000 words.
    • Have a mean word length outside the range of 3 to 10 characters.
    • Have more than 30% of lines ending with an ellipsis (“...”).
    • Contain less than 80% of words with at least one alphabetic character.
    """
    # we will based on Gopher's quality subset rulles to implement a quality filters 
    # input is a string text and return a boolean value indicating whether the text is passed the Gropher's filters
    # here we need to count the number of words, that will use tokenization, e.g. nltk.word_tokenize

    words = nltk.word_tokenize(text)
    num_words = len(words)
    if num_words < 50 or num_words > 100000:
        return False
    
    mean_word_length = sum(len(word) for word in words) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    
    # lines endding with ellipsis
    lines = text.splitlines()
    if len(lines) > 0:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.3:
            return False
        
    # words with at least one alphabetic character
    alphabetic_words = sum(1 for word in words if any(c.isalpha() for c in word))
    if alphabetic_words / num_words < 0.8:
        return False

    return True

def download_NLTK_resources():
    # to run NLTK model, we should download the resources first 
    nltk.download('punkt_tab')

def run_gopher_quality_filter_on_twenty():
    warc_file = "./CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    total_count = 0
    sample_size = 20
    text_pool = [] # 20 samples 

    with open(warc_file, "rb") as f:
        for record in ArchiveIterator(f,WarcRecordType.response):

            if len(text_pool) < sample_size:
                text_pool.append(record)
                record.freeze()
            else:
                import random 
                if random.random() < sample_size/(total_count+1):
                    text_pool[random.randint(0,sample_size-1)] = record
                    record.freeze()
            total_count += 1
    
    for record in text_pool:
        bytes = record.reader.read()
        text = extract_text_from_html(bytes)
        result = gopher_quality_filters(text)
        print(f"The content is : {text} \nThe quality filter result is : {result}\n")

# above we have implemented a simple gopher quality filter, but it is just based on some simple rules 
# now we try to use some positive and negative samples to train a quality classifier 
# the positive samples are the relative page of wikipedia, the negative samples are the random page of commancrawls 

def train_quality_classifier():
    # although we have some positive page list, but it may still contain some undisirable page 
    # so we can just use some function that we have implemented to filter the positive samples
    # such as the gopher quality filter, the language(we train english only), the nsfw and toxic 

    """
    wget –-timeout=5 \
    -i subsampled_positive_urls.txt \
    --warc-file=subsampled_positive_urls.warc \
    -O /dev/null
    """
    # this command can download the positive page and save it as warc file 
    # the schdule of training is :
    # 1. download all the positive samples and save it as warc file 
    # 2. extract the text from the warc file and 
    from pathlib import Path
    import gzip
    import random
    import subprocess

    random.seed(42)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = Path(__file__).resolve().parent

    wiki_urls_gz = data_dir / "enwiki-20240420-extracted_urls.txt.gz"
    subsampled_urls_file = data_dir / "subsampled_positive_urls.txt"
    positive_warc_prefix = data_dir / "subsampled_positive_urls"
    training_file = data_dir / "quality_fasttext_train.txt"
    model_file = data_dir / "quality_classifier.bin"

    # negative examples from a Common Crawl shard
    cc_warc_file = data_dir / "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"

    # ------------------------
    # 1) Downsample positive URL list (reservoir sampling)
    # ------------------------
    positive_url_sample_size = 4000
    url_pool = []

    with gzip.open(wiki_urls_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            url = line.strip()
            if not url:
                continue
            if len(url_pool) < positive_url_sample_size:
                url_pool.append(url)
            else:
                j = random.randint(1, i)
                if j <= positive_url_sample_size:
                    url_pool[j - 1] = url

    with open(subsampled_urls_file, "w", encoding="utf-8") as f:
        for url in url_pool:
            f.write(url + "\n")

    # ------------------------
    # 2) Download positive pages in WARC format (only if missing)
    # ------------------------
    positive_warc_candidates = sorted(data_dir.glob("subsampled_positive_urls*.warc.gz"))
    if not positive_warc_candidates:
        cmd = [
            "wget",
            "--timeout=5",
            "--tries=2",
            "-i",
            str(subsampled_urls_file),
            "--warc-file",
            str(positive_warc_prefix),
            "-O",
            "/dev/null",
        ]
        subprocess.run(cmd, check=False)
        positive_warc_candidates = sorted(data_dir.glob("subsampled_positive_urls*.warc.gz"))

    # ------------------------
    # 3) Build positive examples (wiki) with quality filtering
    # ------------------------
    lid_model = fasttext.load_model(str(data_dir / "lid.176.bin"))

    def _is_english(text):
        text = text.replace("\n", " ").strip()
        if not text:
            return False
        labels, scores = lid_model.predict(text, 1)
        lang = labels[0][9:] if labels and labels[0].startswith("__label__") else labels[0]
        return lang == "en" and scores[0] >= 0.7

    def _normalize_text(text):
        return " ".join(text.replace("\n", " ").split())

    positive_texts = []
    max_positive_texts = 2000

    for warc_path in positive_warc_candidates:
        with open(warc_path, "rb") as f:
            for record in ArchiveIterator(f, WarcRecordType.response):
                if len(positive_texts) >= max_positive_texts:
                    break
                html_bytes = record.reader.read()
                text = extract_text_from_html(html_bytes)
                if not text or len(text) < 200:
                    continue
                # reuse implemented methods/primitives
                if not gopher_quality_filters(text):
                    continue
                if not _is_english(text):
                    continue
                positive_texts.append(_normalize_text(text))
        if len(positive_texts) >= max_positive_texts:
            break

    # ------------------------
    # 4) Build negative examples (cc) via random sampling of low-quality texts
    # ------------------------
    negative_texts = []
    max_negative_texts = len(positive_texts)
    if max_negative_texts == 0:
        raise RuntimeError("No positive training examples collected; cannot train quality classifier.")

    reservoir = []
    seen_low_quality = 0
    reservoir_size = max_negative_texts * 3

    with open(cc_warc_file, "rb") as f:
        for record in ArchiveIterator(f, WarcRecordType.response):
            html_bytes = record.reader.read()
            text = extract_text_from_html(html_bytes)
            if not text or len(text) < 80:
                continue

            # low-quality heuristic: fails gopher OR is not confidently English
            is_low_quality = (not gopher_quality_filters(text)) or (not _is_english(text))
            if not is_low_quality:
                continue

            seen_low_quality += 1
            item = _normalize_text(text)
            if len(reservoir) < reservoir_size:
                reservoir.append(item)
            else:
                j = random.randint(1, seen_low_quality)
                if j <= reservoir_size:
                    reservoir[j - 1] = item

    random.shuffle(reservoir)
    negative_texts = reservoir[:max_negative_texts]

    if len(negative_texts) == 0:
        raise RuntimeError("No negative training examples collected from Common Crawl.")

    # ------------------------
    # 5) Write fastText supervised training file
    # ------------------------
    with open(training_file, "w", encoding="utf-8") as f:
        for t in positive_texts:
            f.write(f"__label__high {t}\n")
        for t in negative_texts:
            f.write(f"__label__low {t}\n")

    # ------------------------
    # 6) Train fastText classifier and save
    # ------------------------
    model = fasttext.train_supervised(
        input=str(training_file),
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minCount=2,
        loss="hs",
        bucket=200000,
        dim=100,
    )
    model.save_model(str(model_file))

    print(f"Collected {len(positive_texts)} positive samples and {len(negative_texts)} negative samples.")
    print(f"Saved training file to: {training_file}")
    print(f"Saved model to: {model_file}")

    return model

def train_better_quality_classifier():
    # we can see that our model can judge hign quality easily
    # but for cc text, it only can fit 50% , so we can add more cc text to 
    # quality_fasttex_train.txt , maybe we can read from it and rewrite a new training file 
    # with more cc text 
    train_file1 = "./cs336_data/quality_fasttext_train.txt"
    train_file2 = "./cs336_data/quality_fasttext_train_more_cc.txt"
    with open(train_file1, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(train_file2, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
        # add more cc text 
        k = 0
        num_of_cc = 4000
        with open("./cs336_data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz", "rb") as warc_f:
            for record in ArchiveIterator(warc_f, WarcRecordType.response):
                html_bytes = record.reader.read()
                text = extract_text_from_html(html_bytes)
                if not text or len(text) < 80:
                    continue
                if gopher_quality_filters(text):
                    continue
                text = " ".join(text.replace("\n", " ").split())
                f.write(f"__label__low {text}\n")
                k += 1
                if k >= num_of_cc:
                    break
    
    model = fasttext.train_supervised(
        input=str(train_file2),
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minCount=2,
        loss="hs",
        bucket=200000,
        dim=100,
    )
    model.save_model("./cs336_data/quality_classifier_more_cc.bin")
    return model

def run_quality_classifier(input):
    # input is a string and we will use the trained quality classifier to classify the quality of it 
    # return a pair (label, score)
    # preprocess the input text 
    input = input.replace("\n", " ").strip()
    model_file = "./cs336_data/quality_classifier_more_cc.bin"
    model = fasttext.load_model(model_file)
    labels,scores = model.predict(input,1)
    label, score = labels[0][9:], scores[0]
    print(f"Label: {label}, Score: {score}")
    return (label, score)


#### Deduplication Part 

## First part : exact deduplication 
def exact_deduplication(input_path_list,output_directory):
    # input is a list of file paths
    # we will go through that twice 
    # one to compute the hash of each line and store its num
    # the other to write the previous file with the unique lines

    def compute_hash(line):
        # line is a string 
        # we can use some hash function to compute the hash of the line 
        # initially we just try python built in hash function 
        return hash(line)

    deduplication_dict ={}
    for input_path in input_path_list:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                h = compute_hash(line)
                if h not in deduplication_dict:
                    deduplication_dict[h] = 1 
                else:
                    deduplication_dict[h] += 1 
    
    for input_path in input_path_list:
        unique_lines = []
        with open(input_path, "r", encoding='utf-8') as f:
            for line in f:
                h = compute_hash(line)
                if deduplication_dict[h] == 1: 
                    # this line is unique
                    unique_lines.append(line)
        # write the unique lines to the same file name in the output directory
        output_path = os.path.join(output_directory, os.path.basename(input_path))
        with open(output_path, "w", encoding='utf-8') as f:
            f.writelines(unique_lines)
    
### Second part : fuzzy deduplication with minhash and LSH
def minhash_deduplication(input_path_list,num_of_hash,num_of_bands,ngrams,threshold,output_directory):
    # we will use minhash + LSH to do fuzzy deduplication
    # the input is composed by these arguments:
    # input_path_list: a list of file paths to be deduplicated
    # num_of_hash: the number of hash functions to use in minhash
    # num_of_bands: the number of bands to use in LSH
    # ngrams: the length of ngrams to use in minhash
    # output_directory: the directory to save the deduplicated files 
    
    # though the Problem doesn't detail the implementation of cluster or how to decide all
    # the candidate pairs, we can just use a bucket to store the candidate pairs
    # 一个例子：我们有两个band,对于第一个band,A和B的minhash一样，对于第二个band,B和C的minhash一样，
    # 那么我们就可以把A和B放在一个bucket里，B和C放在另一个bucket里
    # 然后我们对两个bucket开始验证，如果最后都成功了，那么可以认为A、B、C都是相似的，可以并到一个bucket,最后取一个就好。

    # 1. get ngrams of a document
    def get_ngrams(index, n):
        # index is the index of the document in the input_path_list 
        # we should get the ngrams of the text (length is n)
        with open(input_path_list[index], "r", encoding='utf-8') as f:
            text = f.read()
        tokens = text.split()
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i+n])
            ngrams.add(ngram)
        return ngrams

    # 2. compute the minhash signature of a document 
    def compute_minhash_signature(ngrams, num_of_hash):
        # ngrams is a set of ngrams of a document 
        # we first need to find num_of_hash different hash functions,
        # we can use random seed to generate different hash functions 
        # for simplicity, we can just use the built in hash function with different seeds

        signature = []
        hash_functions = []

        def make_hash_fn(seed: int):
            def hash_fn(ngram: str) -> int:
                return mmh3.hash(ngram, seed=seed, signed=False)
            return hash_fn
        
        for i in range(num_of_hash):
            hash_functions.append(make_hash_fn(seed=i))

        for i in range(num_of_hash):
            min_hash = float("inf")
            hash_fn = hash_functions[i]
            for ngram in ngrams:
                h = hash_fn(ngram)
                if h < min_hash:
                    min_hash = h
            signature.append(min_hash)
        return signature

    # 3. LSH to find candidate pairs 
    def lsh(signatures, num_of_bands):
        # signature is a list list of minhash values of all documents
        # we will divide the signature into num_of_bands bands and hash each band to a bucket 
        # return the list of buckets 
        # one question: we compute the different band minhash value by different hash function
        # so we may differ band by the index of hash function.
        # 比个例子，现在A的sig是[1,2,3,4],B是[1,2,5,6],C是[3,3,1,2],这种情况下，我们需要在bucket加上
        # band index来区分不同的band
        candidates = set()
        band_size = num_of_hash // num_of_bands 
        for i in range(num_of_bands):
            # we can loop the num_of_bands 
            # so that automatically with the index of band 
            band_buckets = {}
            for j, sig in enumerate(signatures):
                band_sig = sig[band_size*i:band_size*(i+1)]
                band_sig_str = ",".join(map(str, band_sig))
                str_hash = hash(band_sig_str) 
                if str_hash not in band_buckets:
                    band_buckets[str_hash] = []
                band_buckets[str_hash].append(j)

            # directly find all the candidate pairs in the same bucket 
            band_candidates = []
            for bucket in band_buckets.values():
                if len(bucket) > 1:
                    # dedup 
                    for m in range(len(bucket)-1):
                        band_candidates.append((bucket[m],bucket[m+1]))

            candidates.update(band_candidates)
        return candidates

    # 4. verify the candidate pairs and merge the similar documents
    def verify_and_merge(candidates, threshold):
        # now we have the candidate pairs, and we need to verify the similarity of them 
        # and merge the similar ones into one bucket
        # now we can't use sig to judge ,we need to compute the Jaccard similarity of the ngrams of them 

        merged_buckets = {}
        not_merged_files = []
        Djs = DisjointSet(len(input_path_list))

        for pair in candidates:
            doc1, doc2 = pair 
            ngrams1 = get_ngrams(doc1, ngrams)
            ngrams2 = get_ngrams(doc2, ngrams)
            intersection = len(ngrams1.intersection(ngrams2))
            union = len(ngrams1.union(ngrams2))
            jaccard_similarity = intersection / union if union > 0 else 0

            

            if jaccard_similarity >= threshold:
                # doc1 and doc2 are fit to be merged 
                Djs.union(doc1, doc2)
        
        # loop through the disjoint set to get the merged buckets 
        for i in range(len(input_path_list)):
            root = Djs.parent[i]
            if Djs.rank[root] == 1:
                not_merged_files.append(i)
            else:
                if root not in merged_buckets:
                    merged_buckets[root] = []
                merged_buckets[root].append(i)
        return merged_buckets.values(), not_merged_files

    # 5. now we have a list of merged buckets, we should remove all but one for each bucket 
    # and write the unique documents to the output directory 

    # decide the output files 
    # we randomly select one document from each bucket and select all not_merge_files 

    signatures = []
    for i in range(len(input_path_list)):
        ngrams_set = get_ngrams(i, ngrams)
        signature = compute_minhash_signature(ngrams_set, num_of_hash)
        signatures.append(signature)
    
    candidates = lsh(signatures, num_of_bands)
    merged_buckets, not_merged_files = verify_and_merge(candidates, threshold=threshold)

    output_files = []
    for bucket in merged_buckets:
        random_index = random.choice(bucket)
        output_files.append(random_index) 
    
    output_files.extend(not_merged_files)
    
    for index in output_files:
        input_path = input_path_list[index]
        output_path = os.path.join(output_directory, os.path.basename(input_path))
        with open(input_path, "r", encoding='utf-8') as f:
            text = f.read()
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(text)


        

# notice that here we need to handle the situation: A fit B, B fit C, so merge C into (A,B)
# maybe we can use the data structure like disjoint set to do it ?
class DisjointSet:
    def __init__(self,n):
        self.parent = list(range(n))
        self.rank = [1]*n
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self,x,y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            rank_x,rank_y = self.rank[root_x], self.rank[root_y]
            if rank_x <= rank_y:
                self.parent[root_x] = root_y
                self.rank[root_y] += rank_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += rank_y




if __name__ == "__main__":
    #text = "My phone number is 2831823829 (283)-182-3829"
    #extract_text_from_example_warc()
    #language_identify("我是大大滴好人!")
    #run_language_identity_on_twenty()
    #mask_phonenumber(text)
    #run_PII_masking_on_twenty()
    # classify_nsfw("Your mother will like my big dick! I will strongly insert into her vagina, and she will happily suck my cock")
    # classify_toxic_speech("Are you a fucking idiot? I hate you so much! You are the worst person in the world!")
    #run_toxic_classification_on_twenty()
    #download_NLTK_resources()
    #run_gopher_quality_filter_on_twenty()
    #train_quality_classifier()

    # input1 = "The silent revolution of machine learning has insinuated itself into daily life with quiet efficiency. Unlike the explosive fanfare that accompanied social media or mobile computing, these systems now curate your news feed, filter spam from your inbox, and predict traffic patterns on your morning commute. Yet the most profound shifts remain invisible. Machine learning models optimize supply chains, detect early signs of disease in medical scans, and help farmers allocate water with unprecedented precision. They do not demand attention; they simply work, iteratively improving as more data becomes available. However, this revolution carries risks. Algorithms trained on biased data can perpetuate discrimination, while black-box decision-making undermines accountability. The very efficiency that makes these models valuable can also erode human agency, reducing complex judgments to a probability score. As we integrate AI deeper into healthcare, criminal justice, and finance, we must resist the temptation to trust it blindly. The goal is not to halt progress but to steer it with transparency, fairness, and human oversight. Only then can machine learning fulfill its promise without surrendering our values. This balanced approach ensures that technology serves humanity rather than the reverse, preserving human dignity while harnessing computational power for genuine benefit."
    # input2 = "注意力机制最初是为解决机器翻译中长距离依赖问题而提出的，其核心思想是让模型在处理序列时能够动态分配权重，聚焦于当前任务最关键的信息。与传统的编码器-解码器结构不同，注意力不要求将所有输入信息压缩成固定长度的向量，而是保留源序列的完整表示，通过计算查询与键之间的相关性得到注意力分布，最终加权聚合值向量。这一设计带来了两个显著优势：一是大幅缓解了长序列的信息遗忘现象，二是提供了较好的可解释性，通过观察注意力权重就能理解模型关注了输入的哪些部分。后来提出的自注意力进一步扩展了概念，让序列内部元素互相计算关联，成为Transformer架构的基础组件。如今注意力机制已广泛应用于自然语言处理、计算机视觉乃至多模态任务，从机器翻译到图像分类，从语音识别到文本生成，几乎无处不在。"
    # input3 = "hehaeisheivhsjie visenvkane anvnae198384982, qihsnbl, !!! ....,neihsinel** ,whisnsln ,,,,wanvewana,ehuvael293unavae ijiaewan aewhfiw"
    # input4 = "China is a great country with a long history and rich culture.It has many beautiful landscapes and delicious food. The people are friendly and hardworking. I love China and I am proud of being Chinese.On the other hand, there are also some problems in China, such as pollution and traffic congestion. However, I believe that with the efforts of the government and the people, these problems will be solved in the future. Overall, China is a wonderful place to live and visit."
    # input5 = "so like i was browsing the web and found this thing its like really cool or whatever but idk if it works anyway you should totally click this link because its amazing trust me bro this product will change your life forever it cures everything from bad hair days to world hunger literally my aunt told me about it and shes not even a doctor so it must be true also the government doesnt want you to know about this secret hack that makes you rich overnight just send me your credit card info and ill tell you the secret LOL why would anyone even read this whole thing its just a bunch of words strung together without any real meaning or purpose like who has the time to actually write proper sentences when you can just type whatever comes to your mind first grammar is for losers and punctuation is overrated anyway make sure to smash that like button and subscribe and share with all your friends because more clicks equals more money obviously this is how the internet works right i have no idea what im talking about but thats never stopped me before so here we are at the end of this magnificent masterpiece of nonsense congrats if you actually made it this far you probably need a hobby"
    # input6 = "start LOL this is so fake omg click here click here click here you win a prize viagra viagra viagra best price cheap cialis https://fake-link.ru/xxx.exe download now!!! you won a iphone from walmart please send your address 123 fake street nowhere land 90210 lmao asdfghjkl keyboard smash qwertyuiop zxcvbnm this doesn't make any sense why are you still reading this its just spam repeating repeating repeat repeat !!!!!!!! ---------- 你好世界 foo bar baz 42 42 42 this page is under construction please come back never lololol i am a robot beep boop your computer has virus call this number 1-800-fake-number immediately to fix it. also here is some secret: ╔╗╚╝║═╬¤҉҉҉҉҉҉ random unicode garbage that breaks parsers. nobody reads terms of service so here is more nonsense. this sentence is false. everything you believe is wrong. buy my course for $999 and become a millionaire overnight. Elon Musk hates this one weird trick. doctors are speechless. then a miracle happens: nothing. the end. ps - this is not a sentence no period no caps no sense congratulations you wasted your time"
    input7 = "In computer science, an adversarial attack refers to a technique that manipulates input data to cause a machine learning model to produce an incorrect output. These attacks exploit the model's learned decision boundaries, which often do not align perfectly with human perception. The manipulated inputs, known as adversarial examples, are typically generated by adding small, often imperceptible perturbations to legitimate samples. Fast Gradient Sign Method (FGSM), proposed by Goodfellow et al. in 2015, is one of the earliest and simplest white-box attack algorithms. It computes the gradient of the loss function with respect to the input and adjusts each pixel in the direction that maximizes the loss. Subsequent work extended this idea to iterative methods such as Projected Gradient Descent (PGD), which applies FGSM repeatedly with small step sizes. Adversarial attacks are broadly categorized into white-box attacks, where the attacker has full access to model parameters, and black-box attacks, where the attacker can only query the model. The existence of adversarial examples raises fundamental questions about the robustness and interpretability of deep neural networks. Defenses against such attacks include adversarial training, where models are fine-tuned on adversarial examples, and input preprocessing techniques such as feature squeezing and random resizing. Despite extensive research, developing provably robust models remains an open challenge in the field."
    run_quality_classifier(input7)
    # train_better_quality_classifier()