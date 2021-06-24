import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import nltk_utils
import matplotlib.pyplot as plt
import webbrowser
import datetime
import subprocess
cmd = ['pip3','install','--user','--upgrade','e','git+https://github.com/twintproject/twint.git@origin/master#egg=twint']
subprocess.run(cmd)
st.set_page_config(layout="wide")
# App title
st.markdown(
    """
# Stock Price App
Shown are the stock price data for query companies!
"""
)
st.write("---")
# Sidebar

m = st.markdown(
    """
<style>
div.stButton > button:first-child {
    background-color: #fff;
}

</style>""",
    unsafe_allow_html=True,
)

link = "http://127.0.0.1:5500/index.html"

if st.sidebar.button("Home"):
    webbrowser.open(link, new=0)

st.sidebar.subheader("Query parameters")
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

# Retrieving tickers data
ticker_list = pd.read_csv("Indian-Ticker.txt", sep=",", header=None)

tickerOpt = st.sidebar.selectbox("Stock ticker", ticker_list[1])  # Select ticker symbol
tickerStr = ticker_list[ticker_list[1] == tickerOpt]

a = tickerStr[0]
b = tickerStr[1]
c = tickerStr[2]
tickerSymbol = a[0]
tickerName = b[0]
tweeterid = c[0]
tickerData = yf.Ticker(tickerSymbol)  # Get ticker data
tickerDf = tickerData.history(
    period="1d", start=start_date, end=end_date
)  # get the historical prices for this ticker
# Ticker information
string_logo = "<img src=%s>" % tickerData.info["logo_url"]
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info["longName"]
st.header("**%s**" % string_name)

string_summary = tickerData.info["longBusinessSummary"]
st.info(string_summary)

# Ticker data
st.header("**Ticker data**")
st.write(tickerDf)

# Bollinger bands
st.header("**Bollinger Bands**")
qf = cf.QuantFig(tickerDf, title="First Quant Figure", legend="top", name="GS")
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

####
# st.write('---')
# st.write(tickerData.info)

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
    c.Search = "$" + str(tweeterid)

    # Removing retweets
    c.Filter_retweets = True

    # No pictures or video
    c.Media = False

    # English only
    c.Lang = "en"

    # Excluding tweets with links
    c.Links = "exclude"

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
        score = "NEG-"
    else:
        score = "NEU"

    return score


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def vader_scores(df):
    # Instantiating the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Grabbing the sentiment scores and assigning them to a new column
    df["sentiment"] = [
        sid.polarity_scores(df.tweet.iloc[i])["compound"] for i in range(len(df))
    ]

    # Labeling the tweets in a new column
    df["feel"] = df.sentiment.apply(sentiment_class)

    return df


def tweet_donut(df, tickerSymbol):
    plt.style.use("fivethirtyeight")
    plt.rcParams["font.size"] = 3.5
    plt.rcParams["figure.subplot.left"] = 0
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.pie(
        list(df.feel.value_counts()),
        labels=df.feel.value_counts().index,
        autopct="%.1f%%",
        startangle=0,
        wedgeprops={"linewidth": 0.9, "edgecolor": "whitesmoke"},
    )

    circle = plt.Circle((0, 0), 0, color="whitesmoke")
    fig = plt.gcf()

    fig.gca().add_artist(circle)
    plt.legend()
    ax.axis("equal")
    do.write(fig)


def tweet_hist(df, tickerSymbol):
    plt.style.use("fivethirtyeight")

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plotting the sentiment scores
    ax.hist(df["sentiment"], bins=5)

    plt.title(f"Sentiment for {tickerSymbol}")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["negative", "neutral", "positive"])
    plt.xlabel("Sentiment")
    plt.ylabel("# of Tweets")
    his.write(fig)


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

    with st.spinner(f"Getting tweets about {tickerName} take awhile..."):
        st.subheader(f"200 Most Recent Tweets Regarding {tickerName}")
        do, his = st.beta_columns(2)
        # Graphs the donut chart and histogram of the sentiment values
        create_sentiment(tweeterid)

        st.write("_(Using SentimentIntensityAnalyzer from NLTK.VADER)_")

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

# finviz_url = "https://finviz.com/quote.ashx?t="
# news_tables = {}

# url = finviz_url + tickerSymbol
# req = Request(url=url, headers={"user-agent": "app"})
# response = urlopen(req)
# html = BeautifulSoup(response, features="html.parser")
# news_table = html.find(id="news-table")
# news_tables[tickerSymbol] = news_table
# parsed_data = []

# for tickerSymbol, news_table in news_tables.items():

#     for row in news_table.findAll("tr"):

#         title = row.a.text
#         date_data = row.td.text.split(" ")

#         if len(date_data) == 1:
#             time = date_data[0]
#         else:
#             date = date_data[0]
#             time = date_data[1]

#         parsed_data.append([tickerSymbol, date, time, title])

# df = pd.DataFrame(parsed_data, columns=["tickerSymbol", "date", "time", "title"])

# vader = SentimentIntensityAnalyzer()

# f = lambda title: vader.polarity_scores(title)["compound"]
# df["compound"] = df["title"].apply(f)
# df["date"] = pd.to_datetime(df.date).dt.date
# st.write(f"Sentiment analysis of news Headlines on {tickerSymbol}")
# plt.figure(figsize=(2, 4))
# mean_df = df.groupby(["date", "tickerSymbol"]).mean().unstack()
# mean_df = mean_df.xs("compound", axis="columns")
# mean_df.plot(kind="bar")
# plt.legend()
# plt.show()
# st.pyplot(plt)

# Forecast for longer period

from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
if st.checkbox("Forecast for next one month"):
    n_month = 1
    period = n_month * 31

    @st.cache
    def load_data(tickerSymbol):
        data = yf.download(tickerSymbol, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("Loading data...")
    data = load_data(tickerSymbol)
    data_load_state.text("Loading data... done!")

    # Predict forecast with Prophet.
    df_train = data[["Date", "Close"]]

    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader("Forecast data")
    st.write(forecast.tail())
    name = str(datetime.date.today())
    fore = forecast[-50:]
 #  fore.to_csv(
 #      "C:/Users/RUDRA/Desktop/pythonProject3/data/" + name
  # )
    st.write(f"Forecast plot for {n_month} months")
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
