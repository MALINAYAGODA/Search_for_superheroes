import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer, pos_tag

dler = nltk.downloader.Downloader()
sw_eng = set(stopwords.words('english'))
stemmer = SnowballStemmer(language='english')
dler.download('omw-1.4')
dler.download('averaged_perceptron_tagger')



def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = re.sub("(.)\\1{2,}", "\\1", x)
    x = re.sub(r'[^\w\s]', '', x)
    return x


def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.NOUN


def my_lemmatizer(sent):
    lemmatizer = WordNetLemmatizer()
    tokenized_sent = sent.split()
    pos_tagged = [(word, get_wordnet_pos(tag))
                  for word, tag in pos_tag(tokenized_sent)]
    return ' '.join([lemmatizer.lemmatize(word, tag)
                     for word, tag in pos_tagged])


def preprocessing(s: str):
    s = get_clean(s)
    slave = []
    for word in s.split():
        if not word[0].isdigit():
            if not word in sw_eng:
                slave.append(word)
    s = ' '.join(slave)
    s = my_lemmatizer(s)
    return s
