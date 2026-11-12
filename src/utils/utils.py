import os
from pathlib import Path
from logging import Logger
import yfinance as yf
import pandas as pd
import numpy as np
import csv
import openpyxl as pyxl
import re
from sklearn.metrics import precision_recall_curve , roc_curve

from src.config.config import COMBINED_DATASET_FILE , FORWARD_FILL_END , FORWARD_FILL_START , END_DATE , START_DATE
from src.config.data_class import DebtToGdpParameters , MacroEconomicParameters , NonMacroEconomicParameters , \
  FileParameters

FAILED_DEBT_FILES = []

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
      if param.columns_to_keep is not None :
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
      new_cols = [ c for c in macro.columns if c != 'available_date' and c != 'date' ]
      combined_data[ new_cols ] = combined_data[ new_cols ].ffill()
      # combined_data = combined_data.reset_index( drop = True )

    logger.info( f"combined data: {combined_data.head()}" )
    file_path = get_path(COMBINED_DATASET_FILE)
    save_to_csv(data = combined_data, output_file = file_path, logger = logger)
    logger.info(f"combining {len(non_macro_files) + len(macro_files)} files done")
  except Exception as e:
    raise e



def get_best_threshold(target_val, val_probability):
  target_val_array = np.asarray( target_val ).astype( int )
  val_probability_array = np.asarray( val_probability, dtype = float )
  if target_val_array.shape != val_probability_array.shape :
    raise ValueError( "target validation and validation probability must have the same shape" )

  precision , recall , thresholds = precision_recall_curve( target_val , val_probability )
  f1_scores = np.divide( 2 * (precision * recall) , (precision + recall) ,
                         out = np.zeros_like( precision * recall ) , where = (precision + recall) != 0 )

  index = np.argmin( f1_scores[:-1] )
  trh = thresholds[index]

  # target_recall = 0.7
  # valid = recall[ :-1 ] >= target_recall
  # if valid.any() :
  #   best_index = np.where( valid )[ 0 ][ np.argmax( precision[ :-1 ][ valid ] ) ]
  #   best_threshold = thresholds[ best_index ]
  # else :
  #   # no threshold reaches the target recall -- fall back to F1-best
  #   best_index = np.argmax( f1_scores[ :-1 ] )
  #   best_threshold = thresholds[ best_index ]

  # false_positive_rates, true_positive_rates, thresholds = roc_curve(target_val_array, val_probability_array,
  #                                                                   drop_intermediate = False)
  # false_positive_rates = false_positive_rates[1:]
  # true_positive_rates = true_positive_rates[1:]
  # thresholds = thresholds[1:]
  # youden_j_values = false_positive_rates  - true_positive_rates
  # best_index  = int(np.argmax(youden_j_values))
  # best_threshold = thresholds[best_index]

  if trh > 0.5 :
    trh = 0.5
  return trh


def parse_month_string ( cell_str ) :
  """
  Standardizes month strings into a consistent 'YYYY-Mon' format (e.g., '2017-Jan').
  Handles variants like 'JAN-2017', 'JANUARY-2017', 'JAN-17', or 'JAN'.
  """
  cell_str = str(cell_str).upper().strip()
  print(f"Processing cell_str: {cell_str}")

  month_map = {
    'JAN' : 'Jan' , 'FEB' : 'Feb' , 'MAR' : 'Mar' , 'APR' : 'Apr' , 'MAY' : 'May' , 'JUN' : 'Jun' ,
    'JUL' : 'Jul' , 'AUG' : 'Aug' , 'SEP' : 'Sep' , 'OCT' : 'Oct' , 'NOV' : 'Nov' , 'DEC' : 'Dec' ,
    'JANUARY' : 'Jan' , 'FEBRUARY' : 'Feb' , 'MARCH' : 'Mar' , 'APRIL' : 'Apr' , 'JUNE' : 'Jun' ,
    'JULY' : 'Jul' , 'AUGUST' : 'Aug' , 'SEPTEMBER' : 'Sep' , 'OCTOBER' : 'Oct' , 'NOVEMBER' : 'Nov' ,
    'DECEMBER' : 'Dec', 'SEPT' : 'Sep', '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
        '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec',
        '1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun',
        '7': 'Jul', '8': 'Aug', '9': 'Sep'
  }

  try:
    iso_match = re.match( r'^(\d{4})-(\d{2})-(\d{2})' , cell_str )
    if iso_match :
      year , month_num = iso_match.group( 1 ) , iso_match.group( 2 )
      return f"{year}-{month_map.get( month_num , 'Jan' )}"

    # Try parsing 'MMM-YYYY' or 'MMM-YY'
    parts = re.split( r'[-–—_ ]' , cell_str )

    if len( parts ) == 2 :
      part1 , part2 = parts[ 0 ] , parts[ 1 ]
      if (part1 in month_map) and part2.isdigit() :
        year = f"20{part2[ -2 : ]}" if len( part2 ) <= 3 else part2
        return f"{year}-{month_map[ part1 ]}"
      elif (part2 in month_map) and part1.isdigit() :
        year = f"20{part1[ -2 : ]}" if len( part1 ) <= 3 else part1
        return f"{year}-{month_map[ part2 ]}"

    # Fallback to pure string if it's a lone name (e.g., 'JAN')
    if cell_str in month_map :
      return month_map[ cell_str ]

    return cell_str

  except Exception as e:
    print(f"Error parsing month string '{cell_str}': {e}")
    raise e

def extract_from_file ( input_xlsx_path , default_year = None ) :
  """
  Parses and extracts monthly rows from a specific year's spreadsheet matrix.
  """
  if not os.path.exists( input_xlsx_path ) :
    print( f"Warning: File '{input_xlsx_path}' not found. Skipping..." )
    return [ ]

  try :
    workbook = pyxl.load_workbook( input_xlsx_path , data_only = True )
    sheet = workbook.active

  except Exception as e :
    FAILED_DEBT_FILES.append(input_xlsx_path)
    print( f"Error opening Excel file '{input_xlsx_path}': {e}" )
    return [ ]

  raw_months = [ ]
  external_debt_gdp = [ ]
  domestic_debt_gdp = [ ]
  total_debt_gdp = [ ]



  for row in sheet.iter_rows( values_only = True ) :
    if not row or all( cell is None for cell in row ) :
      continue

    row_str_list = [ str( c ).upper().strip() for c in row if c is not None ]
    row_str_raw = "".join( row_str_list )

    # 1. Dynamically target the header calendar tracking row
    if not raw_months:


      has_iso = any( re.match( r'^\d{4}-\d{2}-\d{2}' , cell ) for cell in row_str_list )
      has_text_month = any(
        any( m in cell for m in [ 'JAN-' , 'FEB-' , 'MAR-' , 'JANUARY' ] ) for cell in row_str_list )

      if has_iso or has_text_month :
        # Grab standard data boundaries (Columns 2 to 13)
        raw_months = [ cell for cell in row[ 1 :13 ] if cell is not None ]
        continue
    # 2. Extract relative target rows tracking key metrics (ignoring white spaces)
    if 'GROSSEXTERNALDEBT/GDP*' in row_str_raw.replace( " " , "" ) or 'GROSSCGEXTERNALDEBT/GDP' in row_str_raw.replace( " " , "" ) :
      external_debt_gdp = [  normalize_percentage_value(cell) for cell in row[ 1 :13 ] ]


    elif 'GROSSDOMESTICDEBT/GDP*' in row_str_raw.replace( " " , "" ) or 'GROSSCGDOMESTICDEBT/GDP' in row_str_raw.replace( " " , "" ) :
      domestic_debt_gdp = [  normalize_percentage_value(cell) for cell in row[ 1 :13 ] ]

    elif 'GROSSPUBLICDEBT/GDP*' in row_str_raw.replace( " " , "" ) or 'GROSSTOTALCGDEBT/GDP' in row_str_raw.replace( " " , "" ):
      total_debt_gdp = [  normalize_percentage_value(cell) for cell in row[ 1 :13 ] ]

  # Collate parallel metrics into unified rows
  records = [ ]

  limit = min( len( raw_months ) , len( external_debt_gdp ) , len( domestic_debt_gdp ) , len( total_debt_gdp ) )

  for i in range( limit ) :
    raw_date = raw_months[ i ]
    formatted_date = parse_month_string( raw_date )

    # Stitch context year parameter if cells are missing specific year suffixes
    if '-' not in formatted_date and default_year :
      formatted_date = f"{default_year}-{formatted_date}"

    records.append( [
      formatted_date ,
      external_debt_gdp[ i ] ,
      domestic_debt_gdp[ i ] ,
      total_debt_gdp[ i ]
    ] )

  return records


def compile_raw_debt_to_gdp_data ( input_files: list[DebtToGdpParameters], output_file: str, logger: Logger ):
  """
  Loops through multiple sheet names, runs data consolidation routines,
  and builds one centralized historical output file.
  """
  master_records = [ ]
  output_csv_path = get_path( output_file )

  for file_entry in input_files :
    file_path = get_path(file_entry.file_path)
    fall_back_year = file_entry.year

    print( f"Reading: {file_path}..." )
    file_data = extract_from_file( file_path , default_year = fall_back_year )
    if file_data is not None :
      master_records.extend( file_data )
    else:
      FAILED_DEBT_FILES.append( file_path )

    print(f"failed files: {FAILED_DEBT_FILES}")

  try :
    if master_records is not None :
      master_records.sort( key = lambda x : x[ 0 ] )

      columns = [ 'date' , 'external-debt-to-gdp' , 'domestic-debt-to-gdp' , 'total-public-debt-to-gdp' ]
      df = pd.DataFrame(master_records, columns=columns)
      save_to_csv(data = df, output_file = output_csv_path, logger = logger )


    if FAILED_DEBT_FILES is not None:
      for file_path in FAILED_DEBT_FILES :
        process_bog_and_append_debt_data( input_file = file_path , output_file = output_csv_path , logger = logger )

  except Exception as e :
    raise e

def process_bog_and_append_debt_data ( input_file:str|Path, output_file:str|Path, logger:Logger ) :
  """
  Transforms horizontal monthly debt variables into a clean vertical format
  using pandas, calculates total public debt, and appends to an output file.
  """
  input_csv_path = get_path( input_file )

  # 1. Load the data into a DataFrame
  df = pd.read_csv( input_csv_path )

  # 2. Reshape columns (Jan-Dec) into vertical rows
  # This turns columns into two fields: "variable" (the month name) and "value"
  id_vars = [ "Year" , "Variables" ]
  month_cols = [ "Jan" , "Feb" , "Mar" , "Apr" , "May" , "Jun" , "Jul" , "Aug" , "Sep" , "Oct" , "Nov" , "Dec" ]

  df_melted = df.melt( id_vars = id_vars , value_vars = month_cols , var_name = "Month" , value_name = "Ratio" )

  # 3. Clean strings and construct the required "Date" column (e.g., '2017-Jan')
  df_melted[ "date" ] = df_melted[ "Year" ].astype( str ) + "-" + df_melted[ "Month" ]

  # 4. Standardize row variables names to pivot them back cleanly into distinct columns
  df_melted[ "Variables" ] = df_melted[ "Variables" ].str.lower()

  # Filter or map variables into clean categories
  def categorise_variable ( text ) :
    if "external" in text : return "external-debt-to-gdp"
    if "domestic" in text : return "domestic-debt-to-gdp"
    return "other"

  df_melted[ "Metric" ] = df_melted[ "Variables" ].apply( categorise_variable )
  df_melted = df_melted[ df_melted[ "Metric" ] != "other" ]  # Drop unrecognized rows if any

  # 5. Pivot 'external' and 'domestic' indicators from rows into side-by-side columns
  df_pivoted = df_melted.pivot( index = "date" , columns = "Metric" , values = "Ratio" ).reset_index()

  # Fill any empty missing fields with a zero baseline default value
  df_pivoted = df_pivoted.fillna( 0.0 )

  # 6. Calculate the dynamic total public debt column vector
  df_pivoted[ "total-public-debt-to-gdp" ] = df_pivoted[ "external-debt-to-gdp" ] + df_pivoted[
    "domestic-debt-to-gdp" ]


  # Re-order the columns to match your exact requested structure
  final_cols = [ "date" , "external-debt-to-gdp" , "domestic-debt-to-gdp" , "total-public-debt-to-gdp" ]
  df = df_pivoted[ final_cols ]

  # 8. Check file system state and save data securely
  output_csv_path = get_path( output_file )
  file_exists = output_csv_path.exists()

  if file_exists :
    try :
      df_existing = pd.read_csv( output_csv_path )
      # Gather a list of dates that have already been saved
      existing_dates = df_existing[ "date" ].astype( str ).unique()

      # Filter incoming rows: keep only records where 'Date' is NOT in existing_dates
      df = df[ ~df[ "date" ].isin( existing_dates ) ]

      # 7. Write remaining rows to disk if they pass the filter
      if df.empty :
        print( "⏭️ No new records found. All processed rows already exist in the database." )

      df.to_csv( output_csv_path , mode = 'a' , index = False , header = not file_exists )

    except Exception as e :
      raise e

def normalize_percentage_value ( val ) :
  """
   ensures whole percentages are logged correctly.
  Converts decimal representation (e.g., 0.453) to whole percentages (e.g., 45.3),
  but leaves raw whole values (e.g., 53.0) unmanipulated.
  """
  if val is None :
    return 0.0
  try :
    float_val = float( val )
    # If value is written as a fractional decimal ratio (less than 1.0), scale up to %
    if 0.0 < float_val < 1.0 :
      return round( float_val * 100 , 2 )
    return round( float_val , 2 )
  except ValueError :
    return 0.0



def lag_debt_to_gdp_data(input_file:str, output_file:str, logger:Logger):
  try:

    df = pd.read_csv( get_path( input_file ) )
    df['date'] = pd.to_datetime( df['date'], format = "%Y-%b"   )
    align_point_in_time(macro_df = df, date_col = 'date',availability_lag_days = 90,output_file = output_file, logger = logger )
    logger.info('lagging debt-to-gdp data done')

  except Exception as e :
    raise e

def get_correlation_matrix ( df: pd.DataFrame ) :
  try :
    feature_cols = list(df.drop( columns = [ "target_shock_window" ] ).columns)
    correlation_matrix = df.corr()
    strong_correlation = correlation_matrix.unstack()
    strong_pairs = strong_correlation[ (abs( strong_correlation ) > 0.8) & (strong_correlation != 1) ]
    print( f"strongly correlated features: {strong_pairs}" )

    # correlations = df[
    #   feature_cols + [ 'target_shock_window' ]
    #   ].corr()[ 'target_shock_window' ].sort_values(
    #   ascending = False
    # )
    # print(f"correlation matrix: {correlations}")

  except Exception as e :
    print( e )