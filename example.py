"""
Simple example using aggregateMessages.

The goal is to find the minimum rating of a player (node) by sending the neighbors
the currently known minimum rating by each node.

Run with: ~/spark-2.3.4-bin-hadoop2.7/bin/spark-submit --packages graphframes:graphframes:0.7.0-spark2.3-s_2.11  /vagrant/example.py

"""



from graphframes import GraphFrame
from graphframes.lib import AggregateMessages as AM
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from graphframes.lib import AggregateMessages as AM
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, functions as sqlfunctions, types
from pyspark.sql import functions as F

spark = SparkSession.builder.appName('nba2k-ratings').getOrCreate()

# Some NBA players that have played for the Chicago Bulls and their NBA2K19 ratings.
vertices = spark.createDataFrame([('23', 'Michael', 'Jordan', 99), 
                                  ('33', 'Scottie', 'Pippen', 97),
                                  ('1', 'Derrick', 'Rose', 94),
                                  ('53', 'Artis', 'Gilmore', 94),
                                  ('91', 'Dennis', 'Rodman', 93)],
                                  ['id', 'firstname', 'lastname', 'rating'])


# Some random associations between the players that form a connected graph.
edges = spark.createDataFrame([('23', '33'), 
                               ('33', '23'),
                              ('23', '91'),
                              ('91', '23'),
                               ('1', '33'),
                               ('33', '1'),
                               ('53', '23'),
                               ('23', '53')],
                              ['src', 'dst'])


# UDF for creating a rating data type.
def new_rating(rating, id):
    return {"id": id, "rating": rating}
player_rating_type = types.StructType([types.StructField("id", types.StringType()), types.StructField("rating", types.IntegerType())])
new_rating_udf = F.udf(new_rating, player_rating_type)

# Add as the currently known mimimum rating of each node, the rating of the node itself.
vertices = vertices.withColumn("minRating", new_rating_udf(vertices["rating"], vertices["id"]))
cached_vertices = AM.getCachedDataFrame(vertices)

# Create and print information on the respective GraphFrame
g = GraphFrame(cached_vertices, edges)
g.vertices.show()
g.edges.show()
g.degrees.show()

# UDF for preserving the minimum rating between those received by all neighbors.
def min_rating(ratings):
    min_rating = -1
    min_rating_id = -1
    for rating in ratings:
        if min_rating == -1 or rating.rating < min_rating:
            min_rating = rating.rating
            min_rating_id = rating.id
    return {"id": min_rating_id, "rating": min_rating}
min_rating_udf = F.udf(min_rating, player_rating_type)

# UDF for finding the minimum rating between the old one and the new one.
def compare_rating(old_rating, new_rating):
    return old_rating if old_rating.rating < new_rating.rating else new_rating
compare_rating_udf = F.udf(compare_rating, player_rating_type)

# Iterative graph computations
max_iterations = 5
for _ in range(max_iterations):
  aggregates = g.aggregateMessages(F.collect_set(AM.msg).alias("agg"),
              sendToDst=AM.src["minRating"])
  res = aggregates.withColumn("newMinRating", min_rating_udf("agg")).drop("agg")
  new_vertices = g.vertices.join(res, on="id", how="left_outer").withColumnRenamed("minRating", "oldMinRating").withColumn("minRating", compare_rating_udf(F.col("oldMinRating"), F.col("newMinRating"))).drop("oldMinRating").drop("newMinRating")
  cached_new_vertices = AM.getCachedDataFrame(new_vertices)
  g = GraphFrame(cached_new_vertices, g.edges)
  g.vertices.show()
