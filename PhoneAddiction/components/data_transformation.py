import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
import re
from PhoneAddiction.constant.training_pipeline import TARGET_COLUMN
from PhoneAddiction.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from PhoneAddiction.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from PhoneAddiction.entity.config_entity import DataTransformationConfig
from PhoneAddiction.exception.exception import NetworkSecurityException 
from PhoneAddiction.logging.logger import logging
from PhoneAddiction.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
 
    def clean_school_grade(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove 'th', 'st', 'nd', 'rd' from School_Grade column and convert to int"""
        df["School_Grade"] = df["School_Grade"].apply(
            lambda x: int(re.sub(r"(st|nd|rd|th)", "", str(x))) if pd.notnull(x) else np.nan
        )
        return df
    def categorize_addiction_level(self, df: pd.Series) -> pd.Series:
        """
        Convert continuous Addiction_Level into categories:
        0 = Low (<= 3)
        1 = Medium (3 < x <= 7)
        2 = High (> 7)
        """
        return pd.cut(
            df,
            bins=[-np.inf, 3, 7, np.inf],
            labels=[0, 1, 2]
        ).astype(int)


    def get_data_transformer_object(self) -> Pipeline:
        """
        Build a preprocessing pipeline:
        - Drop ID, Name, Location
        - OneHotEncode Gender
        - LabelEncode Phone_Usage_Purpose
        - Clean School_Grade
        - Impute missing values with KNN
        """
        logging.info("Entered get_data_transformer_object method")

        try:
            categorical_onehot = ["Gender"]
            categorical_label = ["Phone_Usage_Purpose"]
            numeric_features = [
                "Age","School_Grade","Daily_Usage_Hours","Sleep_Hours",
                "Academic_Performance","Social_Interactions","Exercise_Hours",
                "Anxiety_Level","Depression_Level","Self_Esteem","Parental_Control",
                "Screen_Time_Before_Bed","Phone_Checks_Per_Day","Apps_Used_Daily",
                "Time_on_Social_Media","Time_on_Gaming","Time_on_Education",
                "Family_Communication","Weekend_Usage_Hours"
            ]

            # OneHotEncoder
            onehot_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            # LabelEncoder wrapped in FunctionTransformer
            label_pipeline = Pipeline([
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
            # Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("onehot", onehot_pipeline, categorical_onehot),
                    ("label", label_pipeline, categorical_label),
                    ("num", "passthrough", numeric_features)
                ],
                remainder="drop"  # Drop ID, Name, Location
            )

            # Full pipeline with imputer
            processor = Pipeline([
                ("school_grade_cleaner", FunctionTransformer(self.clean_school_grade, validate=False)),
                ("preprocessor", preprocessor),
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS))
            ])

            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys)


        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = self.categorize_addiction_level(train_df[TARGET_COLUMN])

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = self.categorize_addiction_level(test_df[TARGET_COLUMN])
            preprocessor=self.get_data_transformer_object()

            preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)
             

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object,)


            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
