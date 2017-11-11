from Descriptor import Descriptor
from Searcher import Searcher
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, help = "Link to csv file ")
ap.add_argument("-q", "--querry", required = True, help = "Link to the querry image")
ap.add_argument("-r", "--result-path", required = True, help = "Link to result folder")
args = vars(ap.parse_args())

descriptor = Descriptor((8,12,3))

querry = cv2.imread(args['querry'])
features = descriptor.describe(querry)

searcher = Searcher(args['index'])
results = searcher.search(features, 10)

output = open("found_return", "w")
for (score, resultID) in results:
	# load the result image and display it
    output.write("%s\n"%(resultID))
	# result = cv2.imread(args["result_path"] + "/" + resultID)
	# cv2.imshow("Result", result)
	# cv2.waitKey(0)