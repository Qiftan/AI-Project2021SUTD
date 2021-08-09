def get_columns():
    featuresCol = ["tweet_id", 
            "username", 
            "timestamp",
            "#followers",
            "#friends",
            "#retweets", 
            "#favorites", 
            "entities", 
            "sentiment",
            "mentions", 
            "hashtags", 
            "urls"]
#     print(featuresCol)
    return featuresCol
