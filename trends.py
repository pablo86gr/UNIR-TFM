# Basado en https://www.honchosearch.com/blog/seo/how-to-use-python-pytrends-to-automate-google-trends-data/

from pytrends.request import TrendReq
import pandas as pd

pytrend = TrendReq(hl='en-GB', tz=360)

colnames = ["keywords"]
df = pd.read_csv("keyword_list.csv", names=colnames)
df2 = df["keywords"].values.tolist()
# df2.remove("keywords")

dataset = []

for x in range(0,len(df2)):
     keywords = [df2[x]]
     pytrend.build_payload(
          kw_list=keywords,
          cat=0,
          timeframe='2009-01-01 2021-05-01',
          geo='SN')
     data = pytrend.interest_over_time()
     if not data.empty:
          data = data.drop(labels=['isPartial'],axis='columns')
          dataset.append(data)

result = pd.concat(dataset, axis=1)
result.to_csv('search_trends_SN.csv')