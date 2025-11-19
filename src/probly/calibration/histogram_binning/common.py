"""
def histogramMaths (predictions, binaryLabels) : 
    
    minPre = maxPre = predictions[0]

    minPre = min(predictions)
    maxPre = max(predictions)

    k = 10 #number of bins
    binWidth = (maxPre - minPre)/k

    binSize = [0]*k
    binPositives = [0]*k

    for prediction, label in zip(predictions, binaryLabels) : 
        binId = int((prediction - minPre)/binWidth)

        if binId == k : binId = k-1
        
        binSize[binId] += 1
        binPositives[binId] += label 

    binPro = [0]*k

    for i in range(k) : 
        if binSize[i] >= 1 : 
            binPro[i] = binPositives[i]/binSize[i]
        else : binPro[i] = 0.0

    calibratedPro = []

    for prediction in predictions : 
        binId = int((prediction - minPre)/binWidth)
        
        if binId == k : binId = k-1

        calibratedPro.append(binPro[binId])

    return calibratedPro

predictions  = [0.1, 0.2, 0.4, 0.45, 0.9, 0.95]
binaryLabels = [0,    0,   1,   1,    1,   0]

res = histogramMaths(predictions, binaryLabels)

print (res)
"""