# PredictiveMaintenanceAI

Predictive maintenance pipeline that scores industrial machine telemetry for failure risk in real time and pushes alerts (Discord + Google Calendar) via n8n.

Built at DataJam. Trains a Gradient Boosting classifier on the AI4I 2020 Predictive Maintenance dataset, simulates live machine sensor data, and routes failure predictions through an automated notification/scheduling workflow.

## How it works

```
ai4i2020.csv --> train_model.ipynb --> machine_failure_model.pkl
                                              |
                                              v
                      machine_simulator.py (synthetic telemetry, every ~10s)
                                              |
                                              v
                         POST /webhook/machine-status  (n8n)
                                              |
                                              v
                n8n.json workflow: routes by risk level
                  - Healthy       -> Discord "all good" message
                  - Needs service -> Discord alert + Google Calendar maintenance event
                  - Failure       -> Discord alert + Google Calendar emergency repair event
```

`main.py` is a one-shot version of the same flow: it loads the trained model, scores a single hardcoded sample, and posts the result to the webhook — useful for testing the n8n workflow without running the full simulator.

## Repo contents

| File | Description |
|---|---|
| `ai4i2020.csv` | Raw AI4I 2020 Predictive Maintenance dataset (source data) |
| `training_data.csv` | Processed/engineered dataset used for training |
| `train_model.ipynb` | Notebook: loads data, engineers features, trains and evaluates the model, saves it to disk |
| `machine_failure_model.pkl` | Trained scikit-learn pipeline (preprocessing + Gradient Boosting classifier) |
| `machine_simulator.py` | Generates synthetic machine telemetry on a loop, scores it with the model, and POSTs results to the n8n webhook |
| `main.py` | Single-prediction script for quickly testing the model + webhook integration |
| `n8n.json` | Importable n8n workflow: receives predictions, routes by status, sends Discord notifications, and schedules Google Calendar events for maintenance/repairs |

## Model

- **Task:** Binary classification — will the machine fail? (`Machine failure`: 0/1)
- **Features:** Air temperature, process temperature, rotational speed, torque, tool wear, machine type (L/M/H), and an engineered temperature-difference feature
- **Pipeline:** `StandardScaler` (numeric features) + `OneHotEncoder` (machine type) → `GradientBoostingClassifier`, via a scikit-learn `Pipeline`
- **Evaluation:** classification report, ROC AUC, confusion matrix, and correlation heatmap (see `train_model.ipynb`)

## Setup

```bash
pip install pandas numpy scikit-learn joblib requests matplotlib seaborn
```

### Retrain the model

Open and run `train_model.ipynb` end to end. It reads `ai4i2020.csv` and writes a new `machine_failure_model.pkl`.

### Run a single test prediction

```bash
python main.py
```

### Run the live simulator

```bash
python machine_simulator.py
```

Generates a random machine reading every 10 seconds, scores it, and POSTs the result to the n8n webhook endpoint.

### Set up notifications (n8n)

1. Import `n8n.json` into your n8n instance.
2. Update the webhook URL in `main.py` / `machine_simulator.py` to point at your own n8n webhook.
3. Connect your own Discord and Google Calendar credentials in the imported workflow.
4. Activate the workflow.

## Notes

- The webhook URLs in `main.py` and `machine_simulator.py` point to a demo n8n instance and will need to be replaced with your own.
- `training_data.csv` and `machine_failure_model.pkl` are included so the notebook doesn't need to be rerun just to try the simulator/webhook flow.

## Suggested repo name

Renamed from `DataJam_MachineFailure` to **`PredictiveMaintenanceAI`** — it's more descriptive of what the project actually does (predictive maintenance + automated alerting) and reads better outside the hackathon context. Other solid options: `MachineFailurePredictor`, `SmartMaintenanceAlert`, `FailSafe-ML`.
