from pathlib import Path

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import f1_score

from src.config import config
from src.E2EPipeline import E2EPipeline
from src.util.logger import CustomLogger

ROOT = Path(__file__).parents[2]
logger = CustomLogger()


# run performance test
def test_run_e2e_per_img_obj():
    df_raw = pd.read_excel(
        ROOT / "data/Assessment.xlsx",
        engine='openpyxl',
        sheet_name=1).sample(1000, random_state=config.SEED)

    X_test = df_raw.loc[:, [col for col in df_raw.columns if col != "C_seg"]]
    y_test = df_raw.loc[:, "C_seg"]
    le = preprocessing.LabelEncoder()
    y_test = pd.DataFrame(le.fit_transform(y_test))

    pipe = E2EPipeline()
    preds = pipe.inference(X_test)
    f1 = f1_score(y_test, preds)
    logger.info(f"f1 score is: {f1}")
    assert f1 >= 0.9
