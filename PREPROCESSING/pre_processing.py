import csv
f = open('UnApportionedIdentifiers.csv','r')

data = f.readlines()

f.close()

L = []

for i in data:
    L.append(i.split(','))

latitudes = []
longitudes = []
for i in range (1,len(L)):
    longitudes.append(L[i][-1])
    latitudes.append(L[i][-2])

longitudes = [x.replace('\n','') for x in longitudes]


f = open("Climate_Data.csv",'w+')
csvwriter = csv.writer(f, delimiter=',')


#Latitude, Longitude, Parameter, Year, Value
cols = ["Latitude","Longitude","Parameter","Year","Value"]
csvwriter.writerow(cols)


for i in range (0,len(latitudes)):
    file_name = "nasa_power_data"+latitudes[i]+"_"+longitudes[i]+".csv"
    g =  open(file_name,'r')
    data = g.readlines()[18:]
    for j in range (0,len(data)):
        line = data[j].split(',')
        L = []
        L.append(latitudes[i])
        L.append(longitudes[i])
        L.append(line[0])
        L.append(line[1])
        L.append(line[-1].replace('\n',''))
        csvwriter.writerow(L)
    g.close()
