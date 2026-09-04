from pathlib import Path
from logging import Logger
import yfinance as yf
import pandas as pd
import numpy as np

from src.config.config import COMBINED_DATASET_FILE , FORWARD_FILL_END , FORWARD_FILL_START , END_DATE , START_DATE
from src.config.data_class import MacroEconomicParameters, NonMacroEconomicParameters, FileParameters


def get_root_path():
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent/".gitignore").exists():
            return parent
    
    return current_path.parents[2]

PROJECT_ROOT = get_root_path()

def get_path(*path_name:str):

    target_path = PROJECT_ROOT.joinpath(*path_name)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path

def save_to_csv(data:pd.DataFrame|pd.Series, output_file:str|Path, logger:Logger, set_index:bool=False):
    try:
        logger.info("checking file path")
        file_path = get_path(output_file)
        # if file_path.exists():
        #     logger.info("data already exists, skipping")
        #     return None
        
        logger.info("saving data")
        data.to_csv(file_path, index=set_index)
        logger.info(f"data saved successfully to {output_file} file")
    except Exception as e:
        raise e

def download_data(ticker:str, start_date:str, end_date:str, output_file:str, logger: Logger, interval:str="1d"):

    try:
        logger.info("done checking file path")
        logger.info(f"starting to download {output_file} data")
        data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=True
        )

        logger.info(f"downloading {output_file} data successful")
        data = data.reset_index()

        save_to_csv(data=data, output_file=output_file, logger = logger)
        logger.info(f"data saved successfully to {output_file} file")

    except Exception as e:
        logger.error(f"sorry something went wrong: {e}")
        


def format_macro_economic_data(file_path:str|Path, column_name:str, value_name:str):
    source_file = get_path(file_path)
    df = pd.read_csv(source_file)
    pattern = r'^[A-Z][a-z]{2}-\d{2}$'
    df = df.melt(var_name=column_name, value_name=value_name)

    df = df[
      df[column_name]
      .str.contains(pattern, regex=True, na=False)
    ]

    df[column_name] = pd.to_datetime(df[column_name], format="%b-%y")
    df.columns = (df.columns.str.strip().str.lower().str.replace( " " , "-" ))

    return df

def slice_data(start_date:str, end_date:str, data:pd.DataFrame):
    return data.loc[start_date:end_date]

def forward_fill_data(raw_data:pd.DataFrame, start:str, end:str, freq:str, index_column:str):
     daily_calender = pd.date_range(start=start, end=end, freq=freq)
     temp_df = raw_data.copy()

     df = temp_df.set_index( index_column ).reindex( daily_calender ).ffill()
     return df.reset_index().rename(columns={"index":index_column})

 

def lag_macro_economic_data_by_30_days(macro_data:pd.DataFrame, output_file:str, logger:Logger):

    df = macro_data.copy()
    df = df.set_index("date")

    lagged_macro_data = df.shift(30)


    file_path = get_path(output_file)

    sliced_data = slice_data( start_date=START_DATE , end_date=END_DATE , data=lagged_macro_data )
    sliced_data = sliced_data.reset_index().rename( columns = { "index" : "date" } )

    
    save_to_csv(data=sliced_data, output_file=file_path, logger=logger)
    return None


def transform_macro_economic_data_files(param:MacroEconomicParameters, logger:Logger):

    try:

        logger.info(f"formatting {param.file_path} data")

        formatted_data = format_macro_economic_data(file_path=param.file_path,column_name=param.column_name, value_name=param.value_name)
        
        logger.info(f"formatting {param.file_path} data done")

        logger.info(f"forward filling {param.file_path} data")
        align_point_in_time(macro_df = formatted_data,date_col = "date",availability_lag_days =param.lag_days,output_file = param.output_file, logger = logger)
        logger.info("calculating point in time done")

        # forward_filled_data = forward_fill_data(raw_data= formatted_data, start=FORWARD_FILL_START, end=FORWARD_FILL_END, freq="D", index_column="date")
        #
        # lag_macro_economic_data_by_30_days(macro_data=forward_filled_data,
        #                                    output_file=param.output_file, logger=logger)

        logger.info(f"forward filling {param.file_path} data done")
    except Exception as e:
        raise e


def transform_non_macro_economic_data(param:NonMacroEconomicParameters, logger:Logger):

    source_file = get_path(param.file_path)
    df = pd.read_csv(source_file, parse_dates=[param.date_column])

    df.columns = (df.columns.str.strip().str.lower().str.replace(" ","-"))
    df = df.rename(columns={param.date_column:"date"})
    df.set_index("date")
    value_cols = [ c for c in df.columns if c != "date" ]
    for c in value_cols :
      if c == 'mid-rate':
        df.loc[df[c].astype(str).str.contains(','), c] = np.nan

      df.loc[ (df[ c ].isin([0,'0.0000','',np.nan,None])) , c ] = np.nan
    df = forward_fill_data(df, start=START_DATE, end=END_DATE,freq="D",index_column = "date").bfill()


    save_to_csv(data=df, output_file=param.output_file, logger=logger)


def combine_files(params:list[FileParameters], logger:Logger):
  data_calender =  pd.date_range(start=START_DATE, end=END_DATE,freq="D")
  combined_data = pd.DataFrame(index = data_calender)
  combined_data.index.name ="date"

  try:
    logger.info(f"combining {len(params)} files")

    for param in params :
      source_file = get_path(param.file_path)
      df = pd.read_csv( source_file )
      df = df[ param.columns_to_keep ]

      if param.columns_to_rename is not None :
        df = df.rename( columns = param.columns_to_rename)

      df[ 'date' ] = pd.to_datetime( df[ 'date' ], format = "%Y-%m-%d" )
      df = df.set_index("date")
      combined_data = combined_data.join( df , how = "left" )

    combined_data = combined_data.reset_index().rename( columns = { "index" : "date" } )
    logger.info( f"combined data: {combined_data.head()}" )
    save_to_csv(data = combined_data, output_file = COMBINED_DATASET_FILE, logger = logger)
    logger.info(f"combining {len(params)} files done")
  except Exception as e:
    raise e

def convert_volume ( value ) :
  # Convert to string and strip any accidental whitespace
  val_str = str( value ).strip().upper()

  if 'K' in val_str :
    return float( val_str.replace( 'K' , '' ) ) * 1_000
  elif 'M' in val_str :
    return float( val_str.replace( 'M' , '' ) ) * 1_000_000
  elif 'B' in val_str :
    return float( val_str.replace( 'B' , '' ) ) * 1_000_000_000

  # If there is no letter, just convert directly to a float
  try :
    return float( val_str )
  except ValueError :
    return None




def align_point_in_time (
  macro_df:pd.DataFrame ,
  date_col:str,
  availability_lag_days:int,
  output_file:str,
  logger:Logger
) :

  macro = macro_df.copy()

  macro[ date_col ] = pd.to_datetime( macro[ date_col ] )

  # --------------------------------------------------------
  # Estimated information availability date
  # --------------------------------------------------------

  macro[ "available_date" ] = (
    macro[ date_col ]
    + pd.Timedelta( days = availability_lag_days )
  )


  macro = macro.sort_values( "available_date" )
  macro = macro.set_index( "date" )
  sliced_data = slice_data( start_date = START_DATE , end_date = END_DATE , data = macro )
  sliced_data = sliced_data.reset_index().rename( columns = { "index" : "date" } )
  file_path = get_path( output_file )
  save_to_csv( data = sliced_data , output_file = file_path , logger = logger )



def combine_file(non_macro_files:list[FileParameters], macro_files:list[FileParameters], logger:Logger):
  data_calender =  pd.date_range(start=START_DATE, end=END_DATE,freq="D")
  combined_data = pd.DataFrame(index = data_calender)
  combined_data.index.name ="date"

  try:
    logger.info(f"combining {len(non_macro_files) + len(macro_files)} files")

    for param in non_macro_files:
      source_file = get_path(param.file_path)
      df = pd.read_csv( source_file )
      df = df[ param.columns_to_keep ]

      if param.columns_to_rename is not None :
        df = df.rename( columns = param.columns_to_rename)

      df[ 'date' ] = pd.to_datetime( df[ 'date' ], format = "%Y-%m-%d" )
      df = df.set_index("date")
      combined_data = combined_data.join( df , how = "left" )

    combined_data = combined_data.reset_index()
    combined_data[ 'date' ] = pd.to_datetime( combined_data[ 'date' ] )
    combined_data = combined_data.sort_values( "date" )
    print(f"combined data: {combined_data.head()}" )

    for val in macro_files:
      source_file = get_path(val.file_path)
      macro = pd.read_csv( source_file )
      macro = macro[ val.columns_to_keep ]

      if val.columns_to_rename is not None :
        macro = macro.rename( columns = val.columns_to_rename)

      macro[ 'available_date' ] = pd.to_datetime( macro[ 'available_date' ], format = "%Y-%m-%d" )
      macro = macro.sort_values( "available_date" )

      macro = macro.reset_index( drop = True )
      # if "date" not in combined_data.columns :
      #   combined_data = combined_data.reset_index()

      combined_data = pd.merge_asof(
        combined_data ,
        macro,
        left_on = "date" ,
        right_on = "available_date" ,
        direction = "backward"
      )
      combined_data = combined_data.drop( columns = [ "available_date" ] )
      # combined_data = combined_data.reset_index( drop = True )

    logger.info( f"combined data: {combined_data.head()}" )
    file_path = get_path(COMBINED_DATASET_FILE)
    save_to_csv(data = combined_data, output_file = file_path, logger = logger)
    logger.info(f"combining {len(non_macro_files) + len(macro_files)} files done")
  except Exception as e:
    raise e