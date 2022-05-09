from nltk.tokenize import regexp_tokenize
import numpy as np

# Here is a default pattern for tokenization, you can substitue it with yours
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = text.lower()
    return regexp_tokenize(text, pattern)


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1
        
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
            

class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        self.bigram = {}

    def getBigram(self, sentence, index):
        # get the two words for the bigram
        if (index < len(sentence)):
            word = sentence[index].lower()
        else:
            word = 'STOP' # non-stopwords are lower case so this won't interfere
        if (index == 0):
            prevWord = 'START' # non-stopwords are lower case so this won't interfere
        else:
            prevWord = sentence[index - 1].lower()
        # return the bigram
        return (prevWord, word)

    def fit(self, text_set):
        index = 0
        for sentence in text_set:
            for i in range(len(sentence) + 1):
                # get the bigram at index i
                b = self.getBigram(sentence, i)
                # put the (bigram, index) into the hash map
                if b not in self.bigram:
                    self.bigram[b] = index
                    index += 1
                else:
                    continue

    def transform(self, text):
        feature = np.zeros(len(self.bigram))
        for i in range(len(text) + 1):
            b = self.getBigram(text, i)
            if b in self.bigram:
                feature[self.bigram[b]] += 1
        return feature

    def transform_list(self, text_set):
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)

class CustomFeature(FeatureExtractor):
    """customized feature extractor, such as TF-IDF
    uses a combination of unigram and bigram
    """
    def __init__(self):
        self.uf = UnigramFeature()
        self.bf = BigramFeature()

    def fit(self, text_set):
        self.uf.fit(text_set)
        self.bf.fit(text_set)

    def transform(self, text):
        uft = self.uf.transform(text)
        bft = self.bf.transform(text)
        return np.concatenate((uft, bft), axis=0)

    def transform_list(self, text_set):
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        return np.array(features)


        
