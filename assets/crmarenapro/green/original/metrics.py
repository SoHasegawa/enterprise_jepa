"""
Fuzzy Match Metrics for Original CRMArena-Pro Mode

Implements evaluation metrics used by the original benchmark:
- Exact Match (EM)
- F1 Score (token-level)
- BLEU Score
- ROUGE Score

Reference: refrence/CRMArena-main/crm_sandbox/agents/utils.py (lines 175-215)
"""

import re
import string
from typing import Dict, List, Union
from collections import Counter


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for comparison.
    
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """Tokenize normalized string."""
    if not s:
        return []
    return normalize_answer(s).split()


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """
    Exact match after normalization.
    
    Returns: 1 if exact match, 0 otherwise
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Returns: Float between 0.0 and 1.0
    """
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def bleu_score(prediction: str, ground_truth: str, max_n: int = 4) -> float:
    """
    Simplified BLEU score (without brevity penalty for single reference).
    
    Computes n-gram precision for n=1 to max_n and returns geometric mean.
    
    Returns: Float between 0.0 and 1.0
    """
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    
    if not pred_tokens or not gold_tokens:
        return 0.0 if pred_tokens or gold_tokens else 1.0
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    precisions = []
    for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
        pred_ngrams = get_ngrams(pred_tokens, n)
        gold_ngrams = get_ngrams(gold_tokens, n)
        
        if not pred_ngrams:
            break
            
        matches = sum((pred_ngrams & gold_ngrams).values())
        total = sum(pred_ngrams.values())
        
        if total > 0:
            precisions.append(matches / total)
        else:
            precisions.append(0.0)
    
    if not precisions or all(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions (add small epsilon to avoid log(0))
    import math
    log_precisions = [math.log(p + 1e-10) for p in precisions]
    avg_log = sum(log_precisions) / len(log_precisions)
    
    return math.exp(avg_log)


def rouge_score(prediction: str, ground_truth: str) -> float:
    """
    ROUGE-L score (longest common subsequence based).
    
    Returns: Float between 0.0 and 1.0
    """
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(ground_truth)
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # Compute LCS length using dynamic programming
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(gold_tokens)
    
    # F1 of precision and recall
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def get_all_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Compute all fuzzy match metrics.
    
    Returns dict with: em, f1, bleu, rouge
    """
    return {
        "em": exact_match_score(prediction, ground_truth),
        "f1": f1_score(prediction, ground_truth),
        "bleu": bleu_score(prediction, ground_truth),
        "rouge": rouge_score(prediction, ground_truth),
    }
