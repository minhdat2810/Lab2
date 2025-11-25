from pyspark import SparkContext, SparkConf

def main():
    conf = SparkConf().setAppName("Lab2_Bai2").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    ratings1 = sc.textFile("ratings_1.txt")
    ratings2 = sc.textFile("ratings_2.txt")
    movies = sc.textFile("movies.txt")

    all_ratings = ratings1.union(ratings2)
    movie_ratings = all_ratings.map(lambda line: line.split(",")) \
                               .map(lambda x: (x[1], float(x[2])))

    movies_genres = movies.map(lambda line: line.split(",")) \
                          .map(lambda x: (x[0], x[2]))

    joined_data = movie_ratings.join(movies_genres)

    genre_ratings = joined_data.flatMap(lambda x: [(g, (x[1][0], 1)) for g in x[1][1].split("|")])

    genre_stats = genre_ratings.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    genre_avg = genre_stats.mapValues(lambda x: (x[0] / x[1], x[1]))

    results = genre_avg.collect()
    for genre, (avg, count) in results:
        print(f"{genre} - AverageRating: {avg:.2f} (TotalRatings: {count})")

    sc.stop()

if __name__ == "__main__":
    main()