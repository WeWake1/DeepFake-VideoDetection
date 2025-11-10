# Workspace Cleanup Commands

**Copy and paste these commands into your cmd terminal one section at a time.**

## ✅ Verification Results

**Paper folder structure (PERFECT):**
- ✓ `paper/methodology/` - 3 files (architecture.md, INSTALLATION.md, README.md)
- ✓ `paper/results/` - 2 files (best_model.pth, training_log.csv)
- ✓ `paper/code_reference/` - 5 files (dataset.py, defaults.yaml, inference.py, models.py, train.py)
- ✓ `paper/data_description/` - 3 files (enhanced_mapping.csv, frame_mapping.csv, training_pairs.csv)
- ✓ `paper/figures/` - created (empty, ready for plots)

---

## Step 1: Rename folder (9 → co)

```cmd
cd /d J:\DF
ren 9 co
```

**Why:** All documentation references `co/` but the folder is currently named `9/`.

---

## Step 2: Remove empty folders

```cmd
cd /d J:\DF
rmdir _to_delete
rmdir archive\experiments
rmdir archive\preprocessing
rmdir archive\temporary_fixes
rmdir archive
```

**Why:** These folders are empty (no files were staged). Keeping them creates clutter.

---

## Step 3: Archive old experimental code (OPTIONAL)

Since you don't have old experimental scripts in your workspace (they were likely deleted earlier), you have two options:

**Option A - Keep archive for future use:**
```cmd
REM Do nothing - keep empty archive/ folder structure for future experiments
```

**Option B - Remove archive completely:**
```cmd
rmdir /s /q J:\DF\archive
```

**Recommendation:** Remove it (Option B) since you don't have old code to archive.

---

## Step 4: What to put in archive/ (if keeping it)

If you want to preserve the archive structure for documentation purposes, you could move these files into it:

**Files that could go in `archive/preprocessing/`:**
- `co/framer_cpu(final)` - Working reference implementation (CPU version)
- `co/create_mappings.py` - Mapping creation script (used once, not needed for training)
- `co/verify_face_extraction.py` - Verification script (used once)

**Commands (if you want to archive these):**
```cmd
cd /d J:\DF
mkdir archive\preprocessing
copy co\framer_cpu(final) archive\preprocessing\
copy co\create_mappings.py archive\preprocessing\
copy co\verify_face_extraction.py archive\preprocessing\
```

**Note:** Keep these in `co/` if you might need them again. Only archive if you want to declutter.

---

## Step 5: Files that can be safely deleted

```cmd
cd /d J:\DF
del check_system.py
del organize_workspace.py
del INSTALL.md
```

**Why:**
- `check_system.py` - diagnostic script, purpose served (GPU verified working)
- `organize_workspace.py` - one-time organization script, no longer needed
- `INSTALL.md` - duplicate of `INSTALLATION.md` (less comprehensive version)

---

## Final Directory Structure

After cleanup, your workspace will look like:

```
J:\DF/
├── train/                      # Production training code ✓
├── config/                     # Configuration files ✓
├── docs/                       # Architecture documentation ✓
├── checkpoints/                # Trained models (best + last) ✓
├── logs/                       # Training history ✓
├── co/                         # Preprocessing scripts (final versions) ✓
│   ├── face_detect_mtcnn_gpu(final).py
│   ├── framer_cpu(final)
│   ├── create_mappings.py
│   ├── verify_face_extraction.py
│   └── FACE_DETECTION_README.md
├── DS/                         # Dataset metadata ✓
├── paper/                      # Research paper materials ✓
│   ├── methodology/            # 3 docs
│   ├── results/                # 2 files (model + log)
│   ├── code_reference/         # 5 key scripts
│   ├── data_description/       # 3 CSV files
│   └── figures/                # Empty (ready for plots)
├── Celeb-synthesis FAKE FRAMES-1/  # Frame data
├── *.csv                       # Dataset mappings ✓
├── *.json                      # Metadata files ✓
├── README.md                   # Main documentation ✓
├── INSTALLATION.md             # Setup guide ✓
├── GPU_USAGE_GUIDE.md          # GPU usage docs ✓
├── TRAINING_MONITOR.md         # Training monitoring ✓
├── requirements.txt            # Dependencies ✓
└── .venv/                      # Virtual environment ✓
```

---

## Quick Cleanup (All Steps Combined)

**If you want to do everything at once:**

```cmd
cd /d J:\DF
ren 9 co
rmdir _to_delete
rmdir /s /q archive
del check_system.py
del organize_workspace.py
del INSTALL.md
```

---

## Next Steps After Cleanup

1. **Run inference on test set:**
```cmd
.venv\Scripts\python.exe train\inference.py --checkpoint checkpoints\best_model.pth --test-pairs training_pairs.csv --output-dir results
```

2. **Generate training curve plots:**
   - I can create a Python script to plot your training metrics from `logs/training_log.csv`

3. **Create architecture diagram:**
   - I can generate a visual diagram of your dual-stream model

4. **Draft paper sections:**
   - Methodology (preprocessing + architecture + training)
   - Results (performance metrics + comparisons)

---

**Which would you like to do first?**
