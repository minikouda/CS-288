from collections import ChainMap
from typing import Callable, Dict, Set

import pandas as pd
import re


class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        words = set(text.lower().split()) - self.STOP_WORDS
        ret = {word: 1.0 for word in words}
        return self.prefix_with_name(ret)


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        if len(text.split()) < 10:
            k = "short"
            v = 1.0
        else:
            k = "long"
            v = 5.0
        ret = {k: v}
        return self.prefix_with_name(ret)

class NegativeWords(FeatureMap):
    name = "neg"
    NEGATIVE = {"not", "no", "n't", 
                "never", "neither", "nothing", 
                "none", "doesn't", "don't", 
                "didn't", "won't", "wouldn't", 
                "shouldn't", "cannot", "can't", "isn't", "aren't"}
    
    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        toks = text.lower().split()
        feats = {"neg_count": 0.0}

        negating = False
        for t in toks:
            if t in cls.NEGATIVE:
                feats["neg_count"] += 1.0
                negating = True
                continue
            if any(p in t for p in [".", "!", "?", ";"]):
                negating = False
            if negating and t.isalpha():
                feats[f"NOT_{t}"] = 1.0

        return cls.prefix_with_name(feats)

class AverageWordLength(FeatureMap):
    name = "avgwordlen"
    
    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = text.split()
        ret = {"avgwordlen": sum(map(len,words)) / len(words)}
        return cls.prefix_with_name(ret)

class PunctuationCounts(FeatureMap):
    name = "punct"
    
    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        exclamation_count = text.count('!')
        question_count = text.count('?')
        ellipsis_count = text.count('...')
        
        multi_exclaim = 1.0 if '!!' in text or '!!!' in text else 0.0
        multi_question = 1.0 if '??' in text or '???' in text else 0.0
        
        ret = {
            "exclamation": float(exclamation_count),
            "question": float(question_count),
            "ellipsis": float(ellipsis_count),
            "multi_exclaim": multi_exclaim,
            "multi_question": multi_question
        }
        return cls.prefix_with_name(ret)
class CapsFeatures(FeatureMap):
    name = "caps"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = re.findall(r"[A-Za-z]+", text)
        total = max(1, len(words))
        caps = sum(w.isupper() and len(w) >= 2 for w in words)
        ret = {"caps_cnt": float(caps), "caps_ratio": caps / total}
        return cls.prefix_with_name(ret)

class TopicBuckets(FeatureMap):
    name = "topic"
    BUCKETS = {
        "money": {"price", "cost", "expensive", "value", "worth"},
        "time": {"late", "delay", "slow", "postpone"},
        "quality": {"great", "perfect", "excellent", "durable"},
    }

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        toks = set(text.lower().split())
        ret = {k: float(len(toks & v)) for k, v in cls.BUCKETS.items()}
        return cls.prefix_with_name(ret)

FEATURE_CLASSES_MAP = {c.name: c for c in [BagOfWords, 
                                           SentenceLength, 
                                           NegativeWords, 
                                           AverageWordLength, 
                                           PunctuationCounts,
                                           CapsFeatures,
                                           TopicBuckets
                                           ]}
def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len"})
    print(featurize(text))
