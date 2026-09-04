from src.utils.utils import download_data
from src.config.looger_config import get_logger
from src.config.file_config import DATA_TICKERS

logger = get_logger(name=__name__, log_file="historical_data.log")


def get_historical_data():

    for data in DATA_TICKERS:

        name = data["name"]
        ticker = data["ticker"]
        start_date =data["start_date"]
        end_date = data["end_date"]
        interval= data["interval"]
        output_file = data["output_file"]

        download_data(ticker=ticker, start_date=start_date, end_date=end_date,output_file=output_file,logger=logger, interval=interval)





