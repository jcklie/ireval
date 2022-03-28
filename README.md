# irmetrics

This Python package provides an implementation of the most common information retrieval (IR) metrics.
Our goal is to return the same scores as [trec_eval](https://github.com/usnistgov/trec_eval).
We achieve this by extensively comparing our implementations across many different datasets with their results.
`irmetrics` can be installed via

    pip install irmetrics

## Implemented metrics

The following metrics are currently implemented:

| Name              | Function                 | Description                                                                                                                                              |
|-------------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Precision@k       | `precision_at_k`         | Precision is the fraction of retrieved documents that are relevant to the query. Precision@k considers only the documents with the highest `k` scores.   |
| Precision@k%      | `precision_at_k_percent` | Precision is the fraction of retrieved documents that are relevant to the query. Precision@k% considers only the documents with the highest `k`% scores. |
| Recall@k          | `recall_at_k`            | Recall is the fraction of the relevant documents that are successfully retrieved. Recall@k considers only the documents with the highest `k` scores.     |
| Recall@k%         | `recall_at_k_percent`    | Recall is the fraction of the relevant documents that are successfully retrieved. Recall@k% considers only the documents with the highest `k`% scores.   |
| Average precision | `average_precision`      | Average precision is the area under the precision-recall curve.                                                                                          |
| R-precision       | `r_precision`            | R-Precision is the precision after R documents have been retrieved, where R is the number of relevant documents for the topic.                           | |

## Usage

```python
import irmetrics

relevancies = [1, 0, 1, 1, 0]
scores = [0.1, 0.4, 0.35, 0.8]

p5 = irmetrics.precision_at_k(relevancies, scores, 5)
p5pct = irmetrics.precision_at_k_percent(relevancies, scores, 5)

r5 = irmetrics.recall_at_k(relevancies, scores, 5)
r5pct = irmetrics.recall_at_k_percent(relevancies, scores, 5)

ap = irmetrics.average_precision(relevancies, scores)
rprec = irmetrics.r_precision(relevancies, scores)
```
