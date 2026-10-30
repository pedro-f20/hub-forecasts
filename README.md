# hub-forecasts

Zero-shot [Chronos-2](https://huggingface.co/amazon/chronos-2) forecasts submitted weekly to the [AI4Casting Hub](https://github.com/ai4castinghub) as `pedro-Chronos2`:

- [hospitalization-forecast](https://github.com/ai4castinghub/hospitalization-forecast): Ontario hospital bed occupancy (COVID, influenza, RSV)
- [rvdss-forecast](https://github.com/ai4castinghub/rvdss-forecast): Canadian lab test positivity

No fitting, tuning or covariates: the model runs on each hub's own target data. Same model as [respicast-chronos2](https://github.com/pedro-f20/respicast-chronos2).

```
uv venv --python 3.12 && uv pip install -r requirements.txt
.venv/bin/python forecast.py ai4c-hosp <hub checkout> 2026-10-31 out
```

A GitHub Action runs `submit.sh` on Fridays and Saturdays and opens the PR once the round's data is out.
