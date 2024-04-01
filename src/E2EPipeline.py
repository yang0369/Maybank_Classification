import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import featuretools as ft
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.config import config
from src.util.logger import CustomLogger

if config.if_GPU:
    import cupy as cp

# show all columns/rows
pd.options.display.max_rows = 10
pd.options.display.max_columns = 20

ROOT = Path(__file__).parents[1]

logger = CustomLogger()


class Pipeline(ABC):
    """Base class

    Args:
        ABC (_type_): base class
    """

    @abstractmethod
    def preprocess(self,
                   data: pd.DataFrame,
                   drop_missing: bool = False) -> pd.DataFrame:
        pass

    @abstractmethod
    def train(self, data) -> float:
        pass

    @abstractmethod
    def inference(self, test_data: pd.DataFrame) -> List[int]:
        pass


@dataclass
class E2EPipeline(Pipeline):
    """create a model pipeline for training and inference

    Returns:
        _type_: f1 score
    """
    index_col: str = config.index_col
    target_name: str = config.target_name
    nominal_cols: list = field(default_factory=lambda: config.nominal)
    ordinal_cols: list = field(default_factory=lambda: config.ordinal)
    values: list = field(default_factory=lambda: config.values)
    counts: list = field(default_factory=lambda: config.counts)

    def preprocess(self,
                   data: pd.DataFrame,
                   drop_missing: bool = False) -> pd.DataFrame:

        # drop original C_ID and replace it by row index
        data.drop("C_ID", axis=1, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.reset_index(inplace=True)
        data.rename(columns={"index": self.index_col}, inplace=True)

        # drop postal code
        data.drop("PC", axis=1, inplace=True)

        if drop_missing:
            # drop columns with too many missing value
            data.drop(config.drop_cols, axis=1, inplace=True)

            from sklearn.impute import SimpleImputer

            # impute missing values
            imp = SimpleImputer(
                missing_values=np.nan,
                strategy='most_frequent')
            data = pd.DataFrame(
                imp.fit_transform(data),
                columns=data.columns)
        else:
            # impute the missing value with 0
            data["HL_tag"].fillna(value=0, inplace=True)
            data["AL_tag"].fillna(value=0, inplace=True)

        for col in (self.nominal_cols + self.ordinal_cols):
            if col in data.columns:
                data[col] = data[col].astype(str)

        # map string back to np.nan
        data.replace({"nan": np.nan, "NaN": np.nan, "NAN": np.nan},
                     inplace=True)

        # prepare df for standardization
        stand_cols = self.counts + self.values + ['C_AGE', 'DRvCR']
        stand_cols = [col for col in stand_cols if col in data.columns]
        df_stand = data.loc[:, stand_cols]
        data.drop(stand_cols, axis=1, inplace=True)

        # standardize features
        scaler = StandardScaler()
        feature_arr = scaler.fit_transform(df_stand.values)
        df_stand = pd.DataFrame(feature_arr,
                                index=df_stand.index,
                                columns=df_stand.columns)
        data = pd.concat([data, df_stand], axis=1)
        logger.info(f"1. before fe: {data.shape}")

        # feature engineering 1
        es = ft.EntitySet(id='ft')
        fe_col = [col for col in self.values + [self.index_col]
                  if col in data.columns]
        es = es.add_dataframe(
            dataframe_name="ft",
            dataframe=data.loc[:, fe_col],
            index=self.index_col)
        features_matrix, _ = ft.dfs(
            entityset=es,
            target_dataframe_name='ft',
            trans_primitives=config.trans_primitives,
            max_depth=1,
            verbose=True)
        features_matrix.reset_index(inplace=True)
        features_matrix = features_matrix.drop(
            [col for col in self.values
             if col in features_matrix.columns],
            axis=1)
        data = data.merge(features_matrix, how='left', on=self.index_col)
        logger.info(f"2. after fe1: {data.shape}")

        # feature engineering 2
        es = ft.EntitySet(id='ft')
        fe_col = [col for col in self.counts + [self.index_col]
                  if col in data.columns]
        es = es.add_dataframe(
            dataframe_name="ft",
            dataframe=data.loc[:, fe_col],
            index=self.index_col)
        features_matrix, _ = ft.dfs(
            entityset=es,
            target_dataframe_name='ft',
            trans_primitives=[
                "add_numeric",
                "subtract_numeric",
                "divide_numeric",
                'multiply_numeric'
                ],
            max_depth=1,
            verbose=True)
        features_matrix.reset_index(inplace=True)
        features_matrix = features_matrix.drop(
            [col for col in self.counts if
             col in features_matrix.columns],
            axis=1)
        data = data.merge(features_matrix, how='left', on=self.index_col)
        logger.info(f"3. after fe2: {data.shape}")

        # impute extreme values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # data[config.target_name].replace(
        #     {"AFFLUENT": 1, "NORMAL": 0},
        #     inplace=True)
        return data

    def train(self,
              data: pd.DataFrame,
              model_dir: Union[str, Path] = config.model_dir,
              if_GPU: bool = config.if_GPU) -> float:
        """train a model and save it

        Args:
            data (pd.DataFrame): dataset
            model_dir (Union[str, Path], optional): model to be saved at.
            Defaults to config.model_dir.
            if_GPU (bool): if use GPU to train

        Returns:
            float: f1 score
        """

        # drop the customer_id
        df_X = data.drop([self.index_col, self.target_name], axis=1)
        df_y = data.loc[:, [self.target_name]]

        # train, val, test split
        X, X_test, y, y_test = train_test_split(
            df_X,
            df_y,
            test_size=0.1,
            random_state=config.SEED,
            shuffle=True,
            stratify=df_y)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.1,
            random_state=config.SEED,
            shuffle=True,
            stratify=y)

        logger.info(f"train set size: {X_train.shape}, val set size: \
                    {X_val.shape}, test set size: {X_test.shape}")

        # get cols requiring onehotencoding
        categorical = [col for col in (self.ordinal_cols + self.nominal_cols)
                       if col != self.target_name]
        logger.info(f"the categorical variables that need one hot \
                    encoding are: {categorical}")

        # apply onehotencoding for categorical variables
        enc = OneHotEncoder(
            handle_unknown='error',
            sparse_output=False,
            drop=None)
        enc.fit_transform(df_X.loc[:, categorical])
        feature_labels = enc.get_feature_names_out()

        # save trained model
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # save onehotencoder
        with open(model_dir / "encoder", "wb") as f:
            pickle.dump(enc, f)

        feature_arr = enc.transform(X[categorical])
        cat_X = pd.DataFrame(
            feature_arr, columns=feature_labels).reset_index(drop=True)
        X = pd.concat(
            [X.drop(categorical, axis=1).reset_index(drop=True), cat_X],
            axis=1)

        # encode training data
        feature_arr = enc.transform(X_train[categorical])
        cat_train = pd.DataFrame(
            feature_arr, columns=feature_labels).reset_index(drop=True)
        X_train = pd.concat(
            [X_train.drop(categorical, axis=1).reset_index(drop=True),
             cat_train], axis=1)

        # encode validation data
        feature_arr = enc.transform(X_val[categorical])
        cat_val = pd.DataFrame(
            feature_arr, columns=feature_labels).reset_index(drop=True)
        X_val = pd.concat(
            [X_val.drop(categorical, axis=1).reset_index(drop=True), cat_val],
            axis=1)

        # encode test data
        feature_arr = enc.transform(X_test[categorical])
        cat_test = pd.DataFrame(
            feature_arr, columns=feature_labels).reset_index(drop=True)
        X_test = pd.concat(
            [X_test.drop(categorical, axis=1).reset_index(drop=True),
             cat_test],
            axis=1)

        # impute missing feature values with np.nan for XGBoost
        X = X.replace(np.nan)
        X_train = X_train.replace(np.nan)
        X_val = X_val.replace(np.nan)
        X_test = X_test.replace(np.nan)

        nan_cols = [col for col in X.columns if "_nan" in col]

        # impute the nan columns
        X = self.impute_encoded_col(X, nan_cols)
        X_train = self.impute_encoded_col(X_train, nan_cols)
        X_val = self.impute_encoded_col(X_val, nan_cols)
        X_test = self.impute_encoded_col(X_test, nan_cols)

        # drop useless columns by feature selection
        X.drop(config.useless, axis=1, inplace=True)
        X_train.drop(config.useless, axis=1, inplace=True)
        X_val.drop(config.useless, axis=1, inplace=True)
        X_test.drop(config.useless, axis=1, inplace=True)

        # label encoding for target variable
        le = preprocessing.LabelEncoder()
        y = pd.DataFrame(le.fit_transform(y[self.target_name]), columns=["Y"])

        y_train = pd.DataFrame(
            le.transform(y_train[self.target_name]),
            columns=["Y"])

        y_val = pd.DataFrame(
            le.transform(y_val[self.target_name]),
            columns=["Y"])

        y_test = pd.DataFrame(
            le.transform(y_test[self.target_name]),
            columns=["Y"])

        # finally train with all traing + val, and change to dart booster
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y
        )

        # Use "hist" for constructing the trees, with early stopping enabled.
        model = xgb.XGBClassifier(
            device="cuda" if if_GPU else "cpu",
            tree_method=config.tree_method,
            objective=config.objective,
            n_estimators=config.n_estimators,
            missing=np.nan,
            eval_metric=config.eval_metric,
            booster=config.booster,
            eta=config.eta,
            max_depth=config.max_depth,
            max_leaves=config.max_leaves,
            min_child_weight=config.min_child_weight,
            gamma=config.gamma,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree)

        if if_GPU:
            X = cp.array(X)
            y = cp.array(y)
            X_test = cp.array(X_test)

        model.fit(X,
                  y,
                  verbose=True,
                  sample_weight=sample_weights)
        model.save_model(model_dir / "xgboost.json")

        # predict on test set
        prob = model.predict_proba(X_test)[:, 1]
        preds = (prob > config.best_threshold).astype("int")

        return f1_score(y_test.values.ravel(), preds)

    # as onehot add extra nan column, so we need to remove it
    @staticmethod
    def impute_encoded_col(df: pd.DataFrame, col_lst: list) -> pd.DataFrame:
        """remove the nan columns created by onehotencoding,
        and impute the values correspondingly

        Args:
            df (pd.DataFrame): data
            col_lst (list): nan columns

        Returns:
            pd.DataFrame: processed data
        """
        for col_n in col_lst:
            key = col_n[:-4]
            index = (df.loc[df[col_n] == 1]).index

            # drop column
            df.drop(col_n, axis=1, inplace=True)

            # impute with np.nan
            df.loc[index, [col for col in df.columns if key in col]] = np.nan

        return df

    def inference(self,
                  test_data: pd.DataFrame,
                  model_dir: Path = config.model_dir) -> List[int]:
        """
        inference pipeline
        """
        data = self.preprocess(test_data)
        X = data.drop([self.index_col], axis=1)

        try:
            path = config.model_dir / "encoder"
            with open(path, "rb") as f:
                enc = pickle.load(f)
            logger.info("successfully loaded the onehot encoder")
        except FileNotFoundError:
            logger.info(f"Please ensure there is encoder in the this path: \
                        {config.model_dir}")

        # get cols requiring onehotencoding
        categorical = [col for col in (self.ordinal_cols + self.nominal_cols)
                       if col != self.target_name]

        # for training data
        feature_arr = enc.transform(X[categorical])
        cat_train = pd.DataFrame(
            feature_arr,
            columns=enc.get_feature_names_out()).reset_index(drop=True)

        X = pd.concat(
            [X.drop(categorical, axis=1).reset_index(drop=True),
             cat_train], axis=1)
        logger.info(f"4. after ohe: {X.shape}")

        X = X.replace(np.nan)

        nan_cols = [col for col in X.columns if "_nan" in col]

        X = self.impute_encoded_col(X, nan_cols)
        X.drop(config.useless, axis=1, inplace=True)
        logger.info(f"5. before training: {X.shape}")

        # make inference
        # Use "hist" for constructing the trees, with early stopping enabled.
        model = xgb.XGBClassifier(
            tree_method=config.tree_method,
            objective=config.objective,
            n_estimators=config.n_estimators,
            missing=np.nan,
            eval_metric=config.eval_metric,
            booster=config.booster,
            eta=config.eta,
            max_depth=config.max_depth,
            max_leaves=config.max_leaves,
            min_child_weight=config.min_child_weight,
            gamma=config.gamma,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree)

        model.load_model(model_dir / "xgboost.json")
        logger.info("successfully loaded the trained model")

        prob = model.predict_proba(X)[:, 1]
        preds = (prob > config.best_threshold).astype("int")

        return preds

    def load_data(data_dir: Union[str, Path]) -> pd.DataFrame:
        """load raw dataset

        Args:
            data_dir (Union[str, Path]): raw data diretory

        Returns:
            pd.DataFrame: raw data
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        df_raw = pd.read_excel(data_dir, engine='openpyxl', sheet_name=1)
        return df_raw


if __name__ == "__main__":

    df_raw = pd.read_excel(
        ROOT / "data/Assessment.xlsx",
        engine='openpyxl',
        sheet_name=1)

    pipe = E2EPipeline()
    data = pipe.preprocess(df_raw)
    f1 = pipe.train(data)
    logger.info(f1)
