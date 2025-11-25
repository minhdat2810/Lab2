from pyspark import SparkContext, SparkConf

def main():
    conf = SparkConf().setAppName("Lab2_Bai3").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    ratings1 = sc.textFile("ratings_1.txt")
    ratings2 = sc.textFile("ratings_2.txt")
    users = sc.textFile("users.txt")
    movies = sc.textFile("movies.txt")

    users_rdd = users.map(lambda line: line.split(",")) \
                     .map(lambda x: (x[0], x[1]))

    all_ratings = ratings1.union(ratings2)
    ratings_parsed = all_ratings.map(lambda line: line.split(",")) \
                                .map(lambda x: (x[0], (x[1], float(x[2]))))

    user_ratings = ratings_parsed.join(users_rdd)

    movie_gender_rdd = user_ratings.map(lambda x: (x[1][0][0], (x[1][0][1], x[1][1])))

    def map_gender_stats(data):
        rating = data[0]
        gender = data[1]
        if gender == 'M':
            return (rating, 1, 0.0, 0)
        else:
            return (0.0, 0, rating, 1)

    mapped_stats = movie_gender_rdd.mapValues(map_gender_stats)

    reduced_stats = mapped_stats.reduceByKey(lambda x, y: (
        x[0] + y[0],
        x[1] + y[1],
        x[2] + y[2],
        x[3] + y[3]
    ))

    movies_names = movies.map(lambda line: line.split(",")) \
                         .map(lambda x: (x[0], x[1]))
    
    final_data = reduced_stats.join(movies_names)

    results = final_data.collect()

    for movie_id, (stats, title) in results:
        sum_m, cnt_m, sum_f, cnt_f = stats
        
        avg_m = f"{sum_m / cnt_m:.2f}" if cnt_m > 0 else "NA"
        avg_f = f"{sum_f / cnt_f:.2f}" if cnt_f > 0 else "NA"

        print(f"{title} - Male_Avg: {avg_m}, Female_Avg: {avg_f}")

    sc.stop()

if __name__ == "__main__":
    main()