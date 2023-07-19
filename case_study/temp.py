# -*- coding: utf-8 -*-
"""
Readme file for case study Algorithmic trading 
isa

"""
import os

def create_readme(directory):
    content = """
# Project Name

This is a brief description of the project.
#Scripts 
1. casestudy_script.py #  Importing required packages
                       #  Getting public historical data and storing in dataframe
                       #  Making Connection to Deribit
                       #  Creating windows for strategy signals
                       #  Getting current market price
                       #  Define sell function
                       #  Define buy function
                       #  Function for live trading
                       #  Closing Open positions
                       #  Backtesting, (I misunderstood the idea of backtesting, 
                          thus, instead of choosing strategy through backtesting, 
                          I performed it to show how the strategy could work on 
                          historical data which represent more time
                       #  Equite curve
                       #  Calculation of cumulative returns
                       #  Calculating Risk parameters
                       #  Calculating log returns
                       #  Plotting buy and sell signal results over narket price 
                          based on live trading data
                       
2. casestudy_2ndbenchmark_sma.py  #For second benchmark strategy I created separate 
                        script, so variable explorer will not be corrupted
                       #  Importing required packages
                       #  Obtain historical market data from Test Deribit 
                       #  Calculate the moving average
                       #  Generate buy and sell signals
                       #  Plot the buy and sell signals
                       #  Backtesting

# In the Import part I placed directory setting code snippet, you can modify based on your directory,
  Each script also follows the content list that I provided and they are almost 
  same so its easy to read
# Data section
  As I mentioned on summary paper, my live trading interrupted several times, 
  as it was automated
  I wasnt following the running process, and every time when setted looop ended
  or I checked I modified 
  the code tried to find solutions, thats why first round of live trading datas 
  that I called to store
  were overwritten, although I later adjusted code to not overwrite the csv file,
  but continue adding,
  while checking file I saw some data loses as well. Data Handler class didnt 
  work for me, I couldnt store
  data on SQL, I used code from lecture and tutorials, even tried to modified 
  but no matter what somehow my devide didnt work with it. As a better solution, 
  I downloaded Trade history data manually, and took uninterrupted 8.5 hours data,
  in folder you can also find original trade history data.
  
3. trading_history.csv is manual download of trade history from test deribit
4. livetradedata.csv is last uninterrupted 8.5 hours data that I sort from the trade history

# Presentation
5. casestudy_presentation.pdf is my presentation
# Summary
6. summary_Algorithmic_Trading_Strategy_Development_Performance_Analysis_CaseStudy
    is my one pager that summarize the strategy and outcomes.   

#Requirements
7. requirements_sma_2ndStrategy.yml is requiremts file for Single Moving Average script
8. requirements.yml is requirements file for casestudy_script


## License

This project is licensed under the [MIT License](LICENSE).
"""
    readme_path = os.path.join(directory, "README.md")
    with open(readme_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    create_readme('/Users/isa/Downloads/pythondataMLU220213600/220213600/case_study')
