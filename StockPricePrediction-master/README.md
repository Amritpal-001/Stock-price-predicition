# Stock Market Price Predictor using Supervised Learning

### Aim
To examine a number of different forecasting techniques to predict future stock returns based on past returns and numerical news indicators to construct a portfolio of multiple stocks in order to diversify the risk. We do this by applying supervised learning methods for stock price forecasting by interpreting the seemingly chaotic market data.

## Setup Instructions
```
    $ workon myvirtualenv                                  [Optional]
    $ pip install -r requirements.txt
    $ python scripts/Algorithms/regression_models.py <input-dir> <output-dir>
```

Download the Dataset needed for running the code from [here](https://drive.google.com/open?id=0B2lCmt16L_r3SUtrTjBlRHk3d1E).

## Project Concept Video
[![Project Concept Video](screenshots/presentation.gif)](https://www.youtube.com/watch?v=z6U0OKGrhy0)

### Methodology 
1. Preprocessing and Cleaning
2. Feature Extraction
3. Twitter Sentiment Analysis and Score
4. Data Normalization
5. Analysis of various supervised learning methods
6. Conclusions

### Research Paper
- [Machine Learning in Stock Price Trend Forecasting. Yuqing Dai, Yuning Zhang](http://cs229.stanford.edu/proj2013/DaiZhang-MachineLearningInStockPriceTrendForecasting.pdf)
- [Stock Market Forecasting Using Machine Learning Algorithms. Shunrong Shen, Haomiao Jiang. Department of Electrical Engineering. Stanford University](http://cs229.stanford.edu/proj2012/ShenJiangZhang-StockMarketForecastingusingMachineLearningAlgorithms.pdf)
- [How can machine learning help stock investment?, Xin Guo](http://cs229.stanford.edu/proj2015/009_report.pdf)


### Datasets used
1. http://www.nasdaq.com/
2. https://in.finance.yahoo.com
3. https://www.google.com/finance


### Useful Links 
- **Slides**: http://www.slideshare.net/SharvilKatariya/stock-price-trend-forecasting-using-supervised-learning
- **Video**: https://www.youtube.com/watch?v=z6U0OKGrhy0
- **Report**: https://github.com/scorpionhiccup/StockPricePrediction/blob/master/Report.pdf

### References
- [Recurrent Neural Networks - LSTM Models](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [ARIMA Models](http://people.duke.edu/~rnau/411arim.htm)
- https://github.com/dv-lebedev/google-quote-downloader
- [Book Value](http://www.investopedia.com/terms/b/bookvalue.asp)
- http://www.investopedia.com/articles/basics/09/simplified-measuring-interpreting-volatility.asp
- [Volatility](http://www.stock-options-made-easy.com/volatility-index.html)
- https://github.com/dzitkowskik/StockPredictionRNN
- [Scikit-Learn](http://scikit-learn.org/stable/)
- [Theano](http://deeplearning.net/software/theano/)
