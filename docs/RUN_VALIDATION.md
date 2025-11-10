# Model Validation Commands - Run These in Order

## Step 1: Investigate 100% Accuracy (5 minutes)

This will analyze your training progression and check for data leakage:

```cmd
cd /d J:\DF
.venv\Scripts\python.exe investigate_accuracy.py
```

**What it does:**
- Analyzes training curves (epochs 1-22)
- Checks for data leakage in train/val split
- Compares to published Celeb-DF benchmarks
- Diagnoses if 100% is legitimate or concerning

**Expected output:** Detailed analysis report showing whether your model is truly strong or overfitting

---

## Step 2: Run Test Set Inference (10-15 minutes)

This is the CRITICAL test - evaluates model on held-out data:

```cmd
cd /d J:\DF
.venv\Scripts\python.exe train\inference.py --checkpoint checkpoints\best_model.pth --test-pairs training_pairs.csv --output-dir results
```

**What it does:**
- Loads best model (epoch 12, 100% val acc)
- Uses last 15% of data as test set (unseen during training)
- Generates per-video predictions
- Calculates final metrics (accuracy, AUC, precision, recall)

**Output files:**
- `results/predictions.csv` - Per-video predictions with confidence scores
- `results/metrics.json` - Final performance metrics

**Expected test accuracy:**
- **>97%:** Model is legitimately strong ✅ (publishable!)
- **90-95%:** Some overfitting but acceptable (normal)
- **<85%:** Significant overfitting ⚠️ (needs investigation)

---

## Step 3: Review Results

After both commands complete, we'll:
1. Compare validation (100%) vs test accuracy
2. Determine if model is publication-ready
3. Generate plots and figures for paper
4. Create architecture diagrams

---

## Quick Copy-Paste (Both Commands)

```cmd
cd /d J:\DF
.venv\Scripts\python.exe investigate_accuracy.py
.venv\Scripts\python.exe train\inference.py --checkpoint checkpoints\best_model.pth --test-pairs training_pairs.csv --output-dir results
```

---

## What to Look For

### Investigation Script Output:
- Training progression (smooth or suspicious jumps?)
- Data leakage warnings
- Comparison to SOTA results
- Diagnosis and recommendations

### Test Inference Output:
```
Test Results:
  Accuracy: 0.XXXX
  AUC-ROC: 0.XXXX
  Precision: 0.XXXX
  Recall: 0.XXXX
  F1-Score: 0.XXXX
```

### Success Criteria:
✅ Test accuracy within 2-3% of validation accuracy (99.39%)
✅ AUC-ROC > 0.97
✅ Investigation shows no data leakage
✅ Training curve shows smooth progression

---

**Ready to start? Copy the commands above and paste into your terminal!**
