# irmetrics

This Python package provides an implementation of the most common information retrieval (IR) metrics.
Our goal is to return the same scores as [trec_eval](https://github.com/usnistgov/trec_eval).
We achieve this by extensively comparing our implementations across many different datasets with their results.
`irmetrics` can be installed via

    pip install irmetrics

## Implemented metrics

The following metrics are currently implemented:

| Name         | Function                 | Description                                                                                                                                              |
|--------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Precision@k  | `precision_at_k`         | Precision is the fraction of retrieved documents that are relevant to the query. Precision@k considers only the documents with the highest `k` scores.   |
| Precision@k% | `precision_at_k_percent` | Precision is the fraction of retrieved documents that are relevant to the query. Precision@k% considers only the documents with the highest `k`% scores. |

## Usage

```python
from irmetrics import *

relevancies = [0, 0, 1, 1]
scores = [0.1, 0.4, 0.35, 0.8]

p5 = precision_at_k(relevancies, scores, 5)
p5 = precision_at_k_percent(relevancies, scores, 5)

```
