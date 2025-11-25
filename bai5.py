from pyspark import SparkContext, SparkConf

def main():
    conf = SparkConf().setAppName("Lab2_Bai5").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    ratings1 = sc.textFile("ratings_1.txt")
    ratings2 = sc.textFile("ratings_2.txt")
    users = sc.textFile("users.txt")
    occupations = sc.textFile("occupation.txt")

    occ_dict = occupations.map(lambda line: line.split(",")) \
                          .map(lambda x: (x[0], x[1]))

    user_occ = users.map(lambda line: line.split(",")) \
                    .map(lambda x: (x[0], x[3]))

    all_ratings = ratings1.union(ratings2)
    user_ratings = all_ratings.map(lambda line: line.split(",")) \
                              .map(lambda x: (x[0], float(x[2])))

    joined_user_occ = user_ratings.join(user_occ)

    occ_ratings = joined_user_occ.map(lambda x: (x[1][1], x[1][0]))

    occ_stats = occ_ratings.mapValues(lambda x: (x, 1)) \
                           .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    occ_avg = occ_stats.mapValues(lambda x: (x[1], x[0] / x[1]))

    final_result = occ_avg.join(occ_dict)

    results = final_result.collect()

    for occ_id, (stats, name) in results:
        total_ratings = stats[0]
        avg_rating = stats[1]
        print(f"{name} - TotalRatings: {total_ratings}, AverageRating: {avg_rating:.2f}")

    sc.stop()

if __name__ == "__main__":
    main()