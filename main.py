import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
import matplotlib.pyplot as plt
# App title
st.markdown('''
# Stock Price App
Shown are the stock price data for query companies!

**Credits**
- App built by Rudra,Omkar,Rohit,Cheenmay
- Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas` and `datetime`
''')
st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

# Ticker information
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

# Ticker data
st.header('**Ticker data**')
st.write(tickerDf)

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

####
#st.write('---')
#st.write(tickerData.info)

import twint
import asyncio


def twitter_scrape(ticker, tweet_cnt=200):
    """
    Scrapes the most recent tweets concerning the selected stock
    """
    # Prevents error: no current event loop in thread
    asyncio.set_event_loop(asyncio.new_event_loop())

    # Configuring Twint to search for the subject in the first specified city
    c = twint.Config()

    # Hiding the print output of tweets scraped
    c.Hide_output = True

    # The amount of tweets to return sorted by most recent
    c.Limit = tweet_cnt

    # Input parameters
    c.Search = '$' + str(string_name)

    # Removing retweets
    c.Filter_retweets = True

    # No pictures or video
    c.Media = False

    # English only
    c.Lang = 'en'

    # Excluding tweets with links
    c.Links = 'exclude'

    # Making the results pandas friendly
    c.Pandas = True

    twint.run.Search(c)

    # Assigning the DF
    df = twint.storage.panda.Tweets_df

    return df


def sentiment_class(score):
    """
    Labels each tweet based on its sentiment score
    """
    if score > 0:
        score = "POS+"
    elif score < 0:
        score = 'NEG-'
    else:
        score = 'NEU'

    return score

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download()

def vader_scores(df):
    # Instantiating the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Grabbing the sentiment scores and assigning them to a new column
    df['sentiment'] = [sid.polarity_scores(df.tweet.iloc[i])['compound'] for i in range(len(df))]

    # Labeling the tweets in a new column
    df['feel'] = df.sentiment.apply(sentiment_class)

    return df


def tweet_donut(df, tickerSymbol):
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 15
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(list(df.feel.value_counts()),
           labels=df.feel.value_counts().index,
           autopct='%1.1f%%',
           wedgeprops={'linewidth': 7, 'edgecolor': 'whitesmoke'})

    circle = plt.Circle((0, 0), 0.3, color='whitesmoke')
    fig = plt.gcf()
    fig.gca().add_artist(circle)

    ax.axis('equal')
    st.pyplot(fig)


def tweet_hist(df, tickerSymbol):
    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotting the sentiment scores
    ax.hist(df['sentiment'], bins=5)

    plt.title(f"Sentiment for {tickerSymbol}")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['negative', 'neutral', 'positive'])
    plt.xlabel("Sentiment")
    plt.ylabel("# of Tweets")
    st.pyplot(fig)


def create_sentiment(ticker, tweet_cnt=200):
    """
    Runs all the required twitter scraping functions
    """
    # Creates a DF with tweets and sentiment scores and labels
    df = vader_scores(twitter_scrape(ticker, tweet_cnt))

    # Creates a donut chart of the tweet count and labels
    tweet_donut(df, ticker)

    st.subheader("Distribution of the Sentiment scores")

    # Creates a histogram of the sentiment scores
    tweet_hist(df, ticker)


# Sentiment Analysis
if st.checkbox("Sentiment Analysis - NLP on Twitter: (Observing General Opinion)"):
    "- Determining the stock's future based on people's thoughts and opinions."



    with st.spinner(f"Getting tweets about {tickerSymbol} take awhile..."):
        st.subheader(f"200 Most Recent Tweets Regarding {tickerSymbol}")

        # Graphs the donut chart and histogram of the sentiment values
        create_sentiment(string_name)

        st.write("_(Using SentimentIntensityAnalyzer from NLTK.VADER)_")


# Forecast for longer period

from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
if st.checkbox("Forecast for longer period"):
    st.subheader("Select number of years to forecast")
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365


    @st.cache
    def load_data(tickerSymbol):
        data = yf.download(tickerSymbol, START, TODAY)
        data.reset_index(inplace=True)
        return data


    data_load_state = st.text('Loading data...')
    data = load_data(tickerSymbol)
    data_load_state.text('Loading data... done!')

    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

