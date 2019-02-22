# import spark
import pyspark.sql.functions as f
from dateutil import parser
from pyspark.sql.types import DateType,IntegerType
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StopWordsRemover
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def getSparkSession():
	return SparkSession \
    	.builder \
    	.appName("Python Spark SQL") \
    	.getOrCreate()

def readJsonFile(spark,path):
 	return spark.read.json(path)

def readSentimentFile(spark,path):
	return spark.read.csv(path)

def getTrumpTwitt(df):
	parseDate = f.udf (lambda x: parser.parse(x),DateType())
	return df.select(parseDate(df.created_at).alias('created_at'),f.lower(df.text).alias('text_lower'),'user.id_str','user.screen_name','user.followers_count',df.retweet_count,df.id,f.col('entities.user_mentions.screen_name').alias('mentioned'),f.col('entities.hashtags.text').alias('hashtags')).where( (f.col('text_lower').like('%trump%')) & (f.col('place.country_code') == 'US') )

def clearAndSplit(df):
	df = df.withColumn('text_lower', f.regexp_replace('text_lower', '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' '))
	return df.withColumn("text_lower", f.split("text_lower", "\s+"))

def buildWordColumn(df):
	remover = StopWordsRemover(inputCol="text_lower", outputCol="text_clean")
	df = remover.transform(df)
	return df.withColumn("word", f.explode(df.text_clean))

def joinDfWithSentiment(df,sentiment):
	return df.join(sentiment, df.word == sentiment._c0, 'left_outer')

def groupByIdAndSumSentiment(df):
	return df.na.fill({'_c1':0}).groupBy('id')\
		.agg(f.sum('_c1').alias('sentiment'),f.first('id_str').alias('user_id'), \
		f.first('screen_name').alias('user_name'), f.first('followers_count').alias('followers_count'), \
		f.first('retweet_count').alias('retweet_count'), f.first('created_at').alias('created_at'), \
		f.first('text_lower').alias('text_lower'),\
 		f.first('mentioned').alias('mentioned'), \
 		f.first("hashtags").alias('hashtags'),)

def filterForDate(dfTrump,startDate,endDate):
	return dfTrump.select("*").where( f.col("created_at").between(startDate,endDate)  )

def groupByDateAndCalculateAverage(df):
	return df.select('sentiment', 'created_at', 'id').groupBy('created_at').agg(f.avg('sentiment').alias('sentiment'), f.count('id').alias('tweet_daily')).withColumn('sentiment',f.col('sentiment') * 100.0)

def groupByDateAndCount(df):
	return df.select('sentiment', 'created_at').groupBy('created_at').agg(f.count('sentiment').alias('tweet_daily'))

def toPandas(df):
	return pd.DataFrame.from_records(df.collect(), columns=df.columns)

def plotGraph(df, y_):
    maxValue = max(df[y_].values) + 2
    minValue = min(df[y_].values) - 2
    if maxValue > 100:
    	maxValue = 100
    if minValue < -100:
    	minValue = -100
    tlen = pd.Series(data = df[y_].values,index=df['created_at'])
    tlen.plot(title='trump_'+y_,ylim=(minValue,maxValue),figsize=(15,4), color='r')

def saveGraphAsPng(graphName):
	graphName = "graph_"+graphName+".png"
	plt.savefig(graphName, format="PNG")
	plt.figure()

def groupByUserId(df):
	return df.groupBy('user_id')\
		.agg(f.count('text_lower').alias('tweet_count'),f.avg('sentiment').alias('sentiment'), \
		f.first('user_name').alias('user_name'), f.avg('followers_count').cast(IntegerType()).alias('followers_count'))

def groupByMention(df):
	return df.withColumn('mentioned', f.explode(df.mentioned)).groupBy('mentioned')\
		.agg(f.count('id').alias('count'),f.avg('sentiment').alias('sentiment'))

def buildForMentionAndPlotBar(df):
	df = groupByMention(df)
	df = buildSentimentType(df)
	topOfMentioned = getTopOfMentioned(df)
	plotBarAndSaveIt(topOfMentioned,'mentioned','count','mentioned')

def buildForFollowersAndTweetCountAndPlotBar(df):
	df = groupByUserId(df)
	df = buildSentimentType(df)
	topOfFollowerCount = getTopOf(df,'followers_count')
	topOfTweetCount = getTopOf(df,'tweet_count')
	plotBarAndSaveIt(topOfFollowerCount,'user_name','followers_count','followers count')
	plotBarAndSaveIt(topOfTweetCount,'user_name','tweet_count','tweet count')


def buildOpinionLeader(df):
	buildForMentionAndPlotBar(df)
	buildForFollowersAndTweetCountAndPlotBar(df)


def getTopOfMentioned(df):
	return df.select(f.col('mentioned').alias('mentioned'),'count',f.col('sentiment').alias('sentiment_type')).orderBy(f.desc('count')).limit(10)

def getTopOf(df,target):
	return df.select("user_name",target,f.col("sentiment").alias("sentiment_type")).orderBy(f.desc(target)).limit(10)

def buildSentimentType(dfOpenionLeader):
	return dfOpenionLeader.withColumn("sentiment",
		f.when(dfOpenionLeader.sentiment > 0 , 'green')\
		.when(dfOpenionLeader.sentiment == 0 , 'blue')\
		.otherwise('red'))

def plotBarAndSaveIt(df,x_,y_,target):
	topOfPd = toPandas(df)
	plotBar(topOfPd,x_,y_,target)
	saveGraphAsPng(y_+"_")

def plotBar(df,x_,y_,target):
	import numpy as np
	fig = plt.figure(figsize=(13,13))
	ax = fig.add_subplot(111)
	y = df[y_]
	x = np.arange(len(y))
	xl = df[x_]
	ax.bar(x ,y, color = df['sentiment_type'], alpha = 0.8)
	plt.xticks(rotation=45)
	plt.title('Top 10 users with highest number of '+target)
	ax.set_xticks(x)
	ax.set_xticklabels(xl)
	red_patch = mpatches.Patch(color='red', label='Negative')
	blue_patch = mpatches.Patch(color='blue', label='Positive')
	green_patch = mpatches.Patch(color='green', label='Neutral')
	ax.legend(handles=[red_patch, blue_patch, green_patch])

def buildStatistic(df):
	df = groupByDateAndCalculateAverage(df)
	dfpandas = toPandas(df)
	plotGraph(dfpandas, 'sentiment')
	saveGraphAsPng('sentiment')
	plotGraph(dfpandas, 'tweet_daily')
	saveGraphAsPng('tweet_daily')

def buildStatisticAndOpenionLeader(df,startDate,endDate):
	df = filterForDate(df,startDate,endDate)
	buildStatistic(df)
	buildOpinionLeader(df)

def main():
	pathToJsonFile = "/data/twitter/twitter2016-2017_trump_metoo_lgbt.json"
	#"/data/twitter/old/twitter2017_1st_week.json"
	#"/data/twitter/twitter2016-2017_trump_metoo_lgbt.json"
	pathToSentimentFile = "/ngantran29/twitter/sentiment.csv"
	startDate = parser.parse("06-01-2016")
	endDate = parser.parse("05-01-2017")
	spark = getSparkSession()
	sqldf = readJsonFile(spark,pathToJsonFile)
	bing = readSentimentFile(spark,pathToSentimentFile)
	dfTrump = getTrumpTwitt(sqldf)
	dfTrump = clearAndSplit(dfTrump)
	dfTrump = buildWordColumn(dfTrump)
	dfTrump = joinDfWithSentiment(dfTrump,bing)
	dfTrump = groupByIdAndSumSentiment(dfTrump)
	buildStatisticAndOpenionLeader(dfTrump,startDate,endDate)

main()

'''
	dfTrump = extractMentionedAndHashtags(dfTrump)
def extractMentionedAndHashtags(df):
	df = df.withColumn('mentioned',f.regexp_extract('text_lower', '^(\@)(\w+)', 2))
	return df.withColumn('hashtags',f.regexp_extract('text_lower', '^(\#)(\w+)', 2))


def saveDfAsCsv(df,fileName):
	fileName = fileName + ".csv"
	dfTime.repartition(1).write.csv.save(fileName)


dfm = df.select('mentioned','id')
dfm = dfm.withColumn("mentioned", f.explode(df.mentioned))
dfm = dfm.groupBy("mentioned").agg(f.count('id').alias('count'))
dfm.orderBy('count').show()

'''
