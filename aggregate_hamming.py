import glob
import pickle

directory_list = glob.glob("*RESULTS.p")

total_results = {}

for file in directory_list:
    data = pickle.load(open(file, "rb"))
    id = int(file.split("_")[0])
    total_results[id] = data['total_avg']

pickle.dump(total_results, open("AGGREGATE_RESULTS.p", "wb"))
print("DONE")