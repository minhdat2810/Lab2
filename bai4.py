from pyspark import SparkContext, SparkConf

def main():
    conf = SparkConf().setAppName("Lab2_Bai4").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    ratings1 = sc.textFile("ratings_1.txt")
    ratings2 = sc.textFile("ratings_2.txt")
    users = sc.textFile("users.txt")
    movies = sc.textFile("movies.txt")

    users_rdd = users.map(lambda line: line.split(",")) \
                     .map(lambda x: (x[0], int(x[2])))

    all_ratings = ratings1.union(ratings2)
    ratings_parsed = all_ratings.map(lambda line: line.split(",")) \
                                .map(lambda x: (x[0], (x[1], float(x[2]))))

    user_ratings = ratings_parsed.join(users_rdd)

    movie_age_rdd = user_ratings.map(lambda x: (x[1][0][0], (x[1][0][1], x[1][1])))

    def map_age_buckets(data):
        rating = data[0]
        age = data[1]
        
        s1, c1, s2, c2, s3, c3, s4, c4 = 0, 0, 0, 0, 0, 0, 0, 0

        if age < 18:
            s1, c1 = rating, 1
        elif 18 <= age < 35:
            s2, c2 = rating, 1
        elif 35 <= age < 50:
            s3, c3 = rating, 1
        else:
            s4, c4 = rating, 1
            
        return (s1, c1, s2, c2, s3, c3, s4, c4)

    mapped_buckets = movie_age_rdd.mapValues(map_age_buckets)

    reduced_stats = mapped_buckets.reduceByKey(lambda x, y: (
        x[0]+y[0], x[1]+y[1],
        x[2]+y[2], x[3]+y[3],
        x[4]+y[4], x[5]+y[5],
        x[6]+y[6], x[7]+y[7]
    ))

    movies_names = movies.map(lambda line: line.split(",")) \
                         .map(lambda x: (x[0], x[1]))
    
    final_data = reduced_stats.join(movies_names)

    results = final_data.collect()

    def format_avg(total_sum, count):
        return f"{total_sum/count:.2f}" if count > 0 else "NA"

    for movie_id, (stats, title) in results:
        s1, c1, s2, c2, s3, c3, s4, c4 = stats
        
        avg1 = format_avg(s1, c1)
        avg2 = format_avg(s2, c2)
        avg3 = format_avg(s3, c3)
        avg4 = format_avg(s4, c4)

        print(f"{title} - [0-18: {avg1}, 18-35: {avg2}, 35-50: {avg3}, 50+: {avg4}]")

    sc.stop()

if __name__ == "__main__":
    main()