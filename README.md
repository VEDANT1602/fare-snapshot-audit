# Fare Snapshot Audit

**The best offline fare model may be using a fact that does not exist when the quote is requested.**

I separated two questions: can a model predict an eventual paid fare from booking-time inputs, and is the quote snapshot still valid when someone requests it? The usual shortcut adds post-booking fields and reports an impressive error. My framework is **declare feature availability → fit on earlier quotes → select on validation → freeze test → enforce snapshot expiry**.

The real fitted model is ridge regression solved by NumPy least squares with training-only normalization. A mean-price control and a deliberately ineligible post-booking model reveal the difference between useful prediction and leakage. The [recorded experiment](results/report.md) contains all test predictions and the actual coefficients in `results/model.json`.


## Recorded result

Across **600 chronological test quotes**, decision-time MAE is **3.3245** versus **24.0671** for the historical mean. The unsafe post-booking diagnostic scores **0.04095**, which is leakage rather than a release candidate. The request fixture serves **300** valid quotes and refuses **300** expired ones. Errors are fictional fare units.

## Reproduce and inspect

Python 3.11, CPU only. No API keys, paid services, provider calls, customer records or external data download.

```bash
python -m venv .venv
# Activate the environment for your shell.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --output runs/local
```

`core.py` contains the generator and actual comparison. `data/dataset.json` is the committed original input. `results/decisions.json` preserves the evaluated rows/traces. `results/metrics.json` and `results/report.md` state the measured outcomes. `results/manifest.json` binds core source, input, and output bytes with SHA-256. A fresh run writes to the requested directory and leaves committed results available for comparison.

This is a new independent portfolio experiment. Its results do not establish historical employer/client outcomes, production adoption, or service scale. Read the [data card](docs/data-card.md) and [provenance](docs/provenance.md) before interpreting the numbers.
