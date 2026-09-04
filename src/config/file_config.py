DATA_TICKERS = [
    {
        "name":"Brent Crude Oil daily data",
        "ticker":"BZ=F",
        "start_date":"2009-01-31",
        "end_date":"2026-01-31",
        "interval":"1d",
        "output_file":"data/raw/brent_crude_daily.csv"
    },
    {
        "name":"comex gold continuous futures data",
        "ticker":"GC=F",
        "start_date":"2009-01-01",
        "end_date":"2026-01-31",
        "interval":"1d",
        "output_file":"data/raw/gold_daily.csv"
    }
]



MACRO_ECONOMIC_DATA_FILES_CONFIG = [
    {
        "name":"Food Inflation Rate",
        "file_path":"data/raw/food_inflation.csv",
        "column_name":"date",
        "value_name":"food_inf",
        "output_file":"data/interim/formatted_food_inflation_rate.csv",
        "lag_days":15
    },
    {
        "name":"InterBank Interest Rate",
        "file_path":"data/raw/interbank_interest_rate.csv",
        "column_name":"date",
        "value_name":"interbank_interest_rate",
        "output_file":"data/interim/formatted_interbank_interest_rate.csv",
        "lag_days":25
    },
    {
        "name":"GH Monetary Policy Rate",
        "file_path":"data/raw/gh_monetary_policy_rate.csv",
        "column_name":"date",
        "value_name":"monetary_policy_rate",
        "output_file":"data/interim/formatted_monetary_policy_rate.csv",
        "lag_days":3
    },
    {
        "name":"GH Net International Reserve",
        "file_path":"data/raw/net_international_reserve.csv",
        "column_name":"date",
        "value_name":"net_international_reserve",
        "output_file":"data/interim/formatted_net_international_reserve.csv",
        "lag_days":25
    },
    {
        "name":"Non Food Inflation Rate",
        "file_path":"data/raw/non_food_inflation.csv",
        "column_name":"date",
        "value_name":"non_food_inf",
        "output_file":"data/interim/formatted_non_food_inflation_rate.csv",
        "lag_days":15
    },
    {
        "name":"Treasury Bill Rate",
        "file_path":"data/raw/treasury_bill_91_days_rate.csv",
        "column_name":"date",
        "value_name":"tbill",
        "output_file":"data/interim/formatted_treasury_bill_rate.csv",
        "lag_days":3
    }
    
]


NON_MACRO_ECONOMIC_DATA_FILES_CONFIG = [
    {
        "name":"Daily USDGH Exchange Rate",
        "file_path":"data/raw/usdgh_historical_fx_rates.csv",
        "date_column":"Date",
        "output_file":"data/interim/formatted_daily_usdgh_exchange_rate.csv"
    },
    {
        "name":"Daily Gold Price",
        "file_path":"data/raw/gold_daily.csv",
        "date_column":"Date",
        "output_file":"data/interim/formatted_daily_comex_gold_price.csv"
    },
    {
        "name":"Daily Brent Crude Price",
        "file_path":"data/raw/brent_crude_daily.csv",
        "date_column":"Date",
        "output_file":"data/interim/formatted_daily_brent_crude_price.csv"
    },
    {
        "name":"US federal Fund Rate",
        "file_path":"data/raw/us_federal_fund_rate.csv",
        "date_column":"observation_date",
        "output_file":"data/interim/formatted_daily_us_federal_fund_rate.csv"
    }
    
]

COMBINE_DATA_FILES_CONFIG = [
  {
    "name" : "Food Inflation Rate" ,
    "file_path" : "data/interim/formatted_food_inflation_rate.csv" ,
    "columns_to_keep" : ["date", "food_inf"] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "InterBank Interest Rate" ,
    "file_path" :"data/interim/formatted_interbank_interest_rate.csv",
    "columns_to_keep" : ["date", "interbank_interest_rate"] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "GH Monetary Policy Rate" ,
    "file_path" :"data/interim/formatted_monetary_policy_rate.csv" ,
    "columns_to_keep" : [ "date", "monetary_policy_rate" ] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "GH Net International Reserve" ,
    "file_path" : "data/interim/formatted_net_international_reserve.csv",
    "columns_to_keep" : [ "date", "net_international_reserve" ] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "Non Food Inflation Rate" ,
    "file_path" : "data/interim/formatted_non_food_inflation_rate.csv",
    "columns_to_keep" : ["date",  "non_food_inf" ] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "Treasury Bill Rate" ,
    "file_path" : "data/interim/formatted_treasury_bill_rate.csv",
    "columns_to_keep" : [ "date", "tbill" ] ,
    "columns_to_rename" : None
  },
  {
      "name":"US federal Fund Rate",
      "file_path":"data/interim/formatted_daily_us_federal_fund_rate.csv",
      "columns_to_keep" : [ "date", "dff" ] ,
      "columns_to_rename" : {"dff":"us_federal_fund_rate"}
    },
  {
      "name":"Daily Brent Crude Price",
      "file_path":"data/interim/formatted_daily_brent_crude_price.csv",
      "columns_to_keep" : [ "date", "price","vol." ] ,
      "columns_to_rename" : {"price":"crude_close_price", "vol.":"crude_volume"}
    },
  {
      "name":"Daily Gold Price",
      "file_path":"data/interim/formatted_daily_comex_gold_price.csv",
      "columns_to_keep" : ["date", "price","vol." ] ,
      "columns_to_rename" : {"price":"gold_close_price", "vol.":"gold_volume"}
    },
  {
      "name":"Daily USDGH Exchange Rate",
      "file_path":"data/interim/formatted_daily_usdgh_exchange_rate.csv",
      "columns_to_keep" : [ "date", "mid-rate" ] ,
      "columns_to_rename" : {"mid-rate":"usdgh_exchange_rate"}
  }

]



FORMATTED_MACRO_DATA_FILES_CONFIG = [
  {
    "name" : "Food Inflation Rate" ,
    "file_path" : "data/interim/formatted_food_inflation_rate.csv" ,
    "columns_to_keep" : [ "food_inf","available_date" ] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "InterBank Interest Rate" ,
    "file_path" :"data/interim/formatted_interbank_interest_rate.csv",
    "columns_to_keep" : [ "interbank_interest_rate","available_date" ] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "GH Monetary Policy Rate" ,
    "file_path" :"data/interim/formatted_monetary_policy_rate.csv" ,
    "columns_to_keep" : [  "monetary_policy_rate", "available_date" ] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "GH Net International Reserve" ,
    "file_path" : "data/interim/formatted_net_international_reserve.csv",
    "columns_to_keep" : [  "net_international_reserve", "available_date" ] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "Non Food Inflation Rate" ,
    "file_path" : "data/interim/formatted_non_food_inflation_rate.csv",
    "columns_to_keep" : [  "non_food_inf","available_date" ] ,
    "columns_to_rename" : None
  } ,
  {
    "name" : "Treasury Bill Rate" ,
    "file_path" : "data/interim/formatted_treasury_bill_rate.csv",
    "columns_to_keep" : [  "tbill","available_date" ] ,
    "columns_to_rename" : None
  }
]


FORMATTED_NON_MACRO_DATA_FILES_CONFIG = [
{
      "name":"US federal Fund Rate",
      "file_path":"data/interim/formatted_daily_us_federal_fund_rate.csv",
      "columns_to_keep" : [ "date", "dff" ] ,
      "columns_to_rename" : {"dff":"us_federal_fund_rate"}
    },
  {
      "name":"Daily Brent Crude Price",
      "file_path":"data/interim/formatted_daily_brent_crude_price.csv",
      "columns_to_keep" : [ "date", "price","vol." ] ,
      "columns_to_rename" : {"price":"crude_close_price", "vol.":"crude_volume"}
    },
  {
      "name":"Daily Gold Price",
      "file_path":"data/interim/formatted_daily_comex_gold_price.csv",
      "columns_to_keep" : ["date", "price","vol." ] ,
      "columns_to_rename" : {"price":"gold_close_price", "vol.":"gold_volume"}
    },
  {
      "name":"Daily USDGH Exchange Rate",
      "file_path":"data/interim/formatted_daily_usdgh_exchange_rate.csv",
      "columns_to_keep" : [ "date", "mid-rate" ] ,
      "columns_to_rename" : {"mid-rate":"usdgh_exchange_rate"}
  }

]