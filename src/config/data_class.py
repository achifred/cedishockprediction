from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MacroEconomicParameters:
    name:str
    file_path:str
    column_name:str
    value_name:str
    output_file:str
    lag_days:int

@dataclass
class NonMacroEconomicParameters:
    file_path:str
    date_column:str
    output_file:str



@dataclass
class FileParameters:
    file_path:str
    columns_to_keep:list[str]
    columns_to_rename:dict[str,str]


@dataclass
class LstmEvaluationParameters:
   target_test_final:Any
   target_probabilities :Any
   target_predictions :Any
   target_predictions:Any
   dates_final:Any


@dataclass
class XGBoostEvaluationParameters:
  target_probability:Any
  target_test:Any
  feature_importance:Any
  columns:Any


@dataclass
class EnsemblePlotParameters:
  precision:Any
  recall:Any
  pr_auc_score:float


@dataclass
class LSTMTrainingParameters:
  feature_train_3d:Any
  target_train:Any
  feature_validation_3d:Any
  validation_dates: Any
  target_validation:Any
  model_output_path: Path
  feature_names:Any


@dataclass
class DebtToGdpParameters:
    file_path:str
    year:str