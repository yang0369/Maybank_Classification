from pathlib import Path

ROOT = Path(__file__).parents[2]

SEED = 1

# processing variables
counts = [
    "NUM_PRD",
    "CASATD_CNT",
    "N_FUNDS",
    "ANN_N_TRX",
    ]

values = [
    "MTHCASA",
    "MAXCASA",
    "MINCASA",
    "pur_price_avg",
    "UT_AVE",
    "MAXUT",
    "CC_AVE",
    "MAX_MTH_TRN_AMT",
    "MIN_MTH_TRN_AMT",
    "MTHTD",
    "MAXTD",
    "Asset value",
    "AVG_TRN_AMT",
    "ANN_TRN_AMT",
    "CC_LMT",
    ]

# drop useless features based on feature importance graph
useless = [
    'C_HSE_OFFICE',
    'ANN_TRN_AMT / AVG_TRN_AMT',
    'C_HSE_COMMERICAL BUILDING',
    'AVG_TRN_AMT / ANN_TRN_AMT',
    'C_HSE_INDUSTRIAL BUILDING',
    'C_HSE_HOTEL/ SERVICE APARTMENT'
    ]


# feature engineering
trans_primitives = [
    "add_numeric",
    "subtract_numeric",
    "divide_numeric",
    'multiply_numeric'
    ]

# xgboost specific
input_dir = ROOT / "data/Assessment.xlsx"
model_dir = ROOT / "model"

target_name = "C_seg"
index_col = "C_ID"
nominal = ["C_EDU", "C_HSE", "gn_occ", "HL_tag", "AL_tag", "C_seg"]
ordinal = ["INCM_TYP"]

colsample_bytree = 0.6
n_estimators = 69
tree_method = "auto"
# booster = "dart"  # update booster to dart
booster = "gbtree"

# tuning parameters
eta = 0.3
max_depth = 10
max_leaves = 2**max_depth

# prevent overfitting
min_child_weight = 1.5
gamma = 0.45095519663195377
subsample = 0.9

# parameters
objective = 'binary:logistic'
booster = "gbtree"
eval_metric = "logloss"

best_threshold = 0.281
model_dir = ROOT / "model"

# for neural network
drop_cols = ["HL_tag", "AL_tag", "pur_price_avg", "UT_AVE", "MAXUT", "N_FUNDS"]
