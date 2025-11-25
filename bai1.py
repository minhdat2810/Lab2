from pyspark import SparkContext, SparkConf

def main():
    conf = SparkConf().setAppName("Lab2_Bai1").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    ratings1 = sc.textFile("ratings_1.txt")
    ratings2 = sc.textFile("ratings_2.txt")
    movies = sc.textFile("movies.txt")

    all_ratings = ratings1.union(ratings2)

    movie_ratings = all_ratings.map(lambda line: line.split(",")) \
                               .map(lambda x: (x[1], float(x[2])))

    rating_stats = movie_ratings.mapValues(lambda x: (x, 1)) \
                                .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    avg_ratings = rating_stats.mapValues(lambda x: (x[0] / x[1], x[1]))

    filtered_ratings = avg_ratings.filter(lambda x: x[1][1] >= 5)

    movies_rdd = movies.map(lambda line: line.split(",")) \
                       .map(lambda x: (x[0], x[1]))

    joined_data = filtered_ratings.join(movies_rdd)

    final_data = joined_data.map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1])))

    if not final_data.isEmpty():
        best_movie = final_data.max()

        avg_score = best_movie[0]
        title = best_movie[1][0]
        count = best_movie[1][1]

        print(f"{title} AverageRating: {avg_score:.2f} (TotalRatings: {count})")
        print(f"{title} is the highest rated movie with an average rating of {avg_score:.2f} among movies with at least 5 ratings.")

    sc.stop()

if __name__ == "__main__":
    main()