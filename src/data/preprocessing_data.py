from src.config.config import RAW_DEBT_TO_GDP_DATA_FILE
from src.config.file_config import (FORMATTED_MACRO_DATA_FILES_CONFIG , FORMATTED_NON_MACRO_DATA_FILES_CONFIG,
                                    DEBT_TO_GDP_DATA_FILES_CONFIG,MACRO_ECONOMIC_DATA_FILES_CONFIG, NON_MACRO_ECONOMIC_DATA_FILES_CONFIG)
from src.utils.utils import combine_file , compile_raw_debt_to_gdp_data , transform_macro_economic_data_files , \
  transform_non_macro_economic_data,lag_debt_to_gdp_data

from src.config.looger_config import get_logger
from src.config.data_class import DebtToGdpParameters , FileParameters , MacroEconomicParameters , \
  NonMacroEconomicParameters


logger = get_logger(name=__name__, log_file="preprocessing_data.log")
RAW_DEBT_TO_GDP_DATA_FILE = 'data/raw/debt_to_gdp_data.csv'
FORMATTED_DEBT_TO_GDP_DATA_FILES_CONFIG = 'data/interim/formatted_debt_to_gdp.csv'

def extract_debt_to_gdp_data():
    gdp_data_files: list[DebtToGdpParameters] = []
    try:
        logger.info("extracting debt to gdp data")
        for data in DEBT_TO_GDP_DATA_FILES_CONFIG:
            gdp_data_files.append(DebtToGdpParameters(file_path=data["file_path"], year=data["year"]))
        compile_raw_debt_to_gdp_data(input_files = gdp_data_files, output_file = RAW_DEBT_TO_GDP_DATA_FILE, logger = logger)



        logger.info("extracting debt to gdp data done")
    except Exception as e:
        logger.exception(f"sorry something went wrong: {e}")

def format_macro_economic_historical_data():
    try:
        logger.info("formatting macro economic data")

        for data in MACRO_ECONOMIC_DATA_FILES_CONFIG:
            param = MacroEconomicParameters(
              name=data["name"],
              file_path=data["file_path"],
              column_name=data["column_name"],
              value_name=data["value_name"],
              output_file=data["output_file"],
              lag_days = data["lag_days"]
              )
            transform_macro_economic_data_files(param=param, logger=logger)

        logger.info("formating macro economic data done")
    except Exception:
        logger.exception("sorry something went wrong")



def format_non_macro_economic_historical_data():
    try:
        logger.info("formatting non macro economic data")

        for data in NON_MACRO_ECONOMIC_DATA_FILES_CONFIG:
            param = NonMacroEconomicParameters(file_path=data["file_path"], 
                                            date_column=data["date_column"],
                                            output_file=data["output_file"]
                                            )
            
            transform_non_macro_economic_data(param=param, logger=logger)

        logger.info("formatting non macro economic data done")
    except Exception as e:
        logger.error(f"sorry something went wrong: {e}")


def combine_file_to_one_dataset():

  try:
        logger.info("combining files to one dataset")

        macro_file_params: list[FileParameters] = []
        non_macro_file_params: list[FileParameters] = []

        for data in FORMATTED_MACRO_DATA_FILES_CONFIG:
          macro_file_params.append(FileParameters(file_path=data["file_path"],
                                       columns_to_keep = data["columns_to_keep"],
                                       columns_to_rename = data["columns_to_rename"])
                        )

        for data in FORMATTED_NON_MACRO_DATA_FILES_CONFIG:
          non_macro_file_params.append(FileParameters(file_path=data["file_path"],
                                       columns_to_keep = data["columns_to_keep"],
                                       columns_to_rename = data["columns_to_rename"])
                        )

        combine_file(non_macro_files=non_macro_file_params, macro_files =macro_file_params , logger=logger)


  except Exception as e:
      logger.exception(f"sorry something went wrong: {e}")



if __name__ == "__main__":
   extract_debt_to_gdp_data()
   lag_debt_to_gdp_data(input_file = RAW_DEBT_TO_GDP_DATA_FILE, output_file = FORMATTED_DEBT_TO_GDP_DATA_FILES_CONFIG,
                        logger = logger)
   format_macro_economic_historical_data()
   format_non_macro_economic_historical_data()
   combine_file_to_one_dataset()