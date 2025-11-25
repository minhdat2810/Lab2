from pyspark import SparkContext, SparkConf
from datetime import datetime

def main():
    conf = SparkConf().setAppName("Lab2_Bai6").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    ratings1 = sc.textFile("ratings_1.txt")
    ratings2 = sc.textFile("ratings_2.txt")
    
    all_ratings = ratings1.union(ratings2)

    def parse_year_rating(line):
        parts = line.split(",")
        rating = float(parts[2])
        timestamp = int(parts[3])
        
        dt_object = datetime.fromtimestamp(timestamp)
        year = dt_object.year
        
        return (year, rating)

    year_ratings = all_ratings.map(parse_year_rating)

    year_stats = year_ratings.mapValues(lambda x: (x, 1)) \
                             .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    year_avg = year_stats.mapValues(lambda x: (x[1], x[0] / x[1]))

    results = year_avg.sortByKey().collect()

    for year, (count, avg) in results:
        print(f"{year} - TotalRatings: {count}, AverageRating: {avg:.2f}")

    sc.stop()

if __name__ == "__main__":
    main()