# Fare Snapshot Audit

The lowest-error model in this experiment cannot quote a fare. It uses a field recorded after booking. On 600 later test quotes it scores 0.041 mean absolute error, which makes it a useful diagnostic for leakage. The serving path rejects that model. [Refusal test](tests/test_contracts.py#L42) · [Fitted coefficients](results/model.json)

A second linear model uses only inputs observed by quote time. It fits feature scaling on the training period and selects regularization on a later validation period. Its test MAE is **3.325 fictional fare units**, against **24.067** for a prediction equal to the training mean. [Per-quote predictions](results/decisions.json) · [Split sizes and errors](results/metrics.json)

A good fare estimate can still be too old to serve. The request fixture exercises the validity window separately from model error. It admits 300 requests and refuses 300 expired ones. The [quote implementation](core.py) checks the feature timestamp and snapshot lifetime before returning a prediction.

## Fit the two models

Run with Python 3.11 and the pinned NumPy dependency in an active virtual environment.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output runs/local
```

[Data card](docs/data-card.md) · [Run manifest](results/manifest.json) · [Source and contribution notes](docs/provenance.md)
