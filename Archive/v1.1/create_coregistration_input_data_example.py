# This is an example script of creating input file for conregistration.py.
# The input file is a text file and contains comma-separated values.
# Each row has five fields -- index, longitude, latitude, time, and plate id.

# The script contains hardcoded file names.
# You should only use this script as an example and modify the example to prepare input data suitable to your research.

import numpy as np
import os, sys
import pandas as pd
from parameters import parameters
import shapefile
# from shapely import geometry
from shapely.geometry import shape
from shapely.geometry import Point
import Utils

# the age is a floating-point number. map the floating-point number to the nereast integer time in the range
def get_time_from_age(ages, start, end, step):
    ret = []
    times = range(start, end+1, step)
    
    for age in ages:
        age = float(age)
        if age <= start:
            ret.append(start)
        elif age >= end:
            ret.append(end)
        else:
            idx = int((age - start)//step)
            mod = (age - start)%step
            if not (mod < step/2.):
                idx = idx+1 
            ret.append(times[idx])
            
    return ret

def process_real_deposits(deposit_path, start_time, end_time, time_step): # path to the shapefile of mineral occurrences
    if not os.path.isfile(deposit_path):
        sys.exit('File not found!')
    reader = shapefile.Reader(deposit_path)
    recs = reader.records()
    min_occ_num = len(recs)
    # longitude
    lons = np.array(recs)[:, 3]
    # latitude
    lats = np.array(recs)[:, 4]
    # time
    times = get_time_from_age(np.array(recs)[:, 14], start_time, end_time, time_step) # get integer ages for andes data
    # plate id
    plate_ids = Utils.get_plate_id(lons, lats)
    # index, lon, lat, time, plate id
    data = []
    
    for i in range(min_occ_num):
        data.append([i, lons[i], lats[i], times[i], plate_ids[i]])
    data = np.array(data)
    
    return data

# def generate_random_deposits(data, start_time, end_time):
#     random_data = []
#     random_ages = np.random.randint(start_time+1, end_time, size=len(data)) # generate random ages for mineral occurrences
#     for i in range(len(data)):
#         random_data.append([i, data[i][1], data[i][2], random_ages[i], data[i][4]])
#     return random_data

def generate_non_deposits(polygon_path, start_time, end_time, num_features):
    polygon_obj = shapefile.Reader(polygon_path)
    feature = polygon_obj.shapeRecords()[0]
    first = feature.shape.__geo_interface__
    polygon = shape(first)
    
    times = np.random.randint(start_time, end_time+1, size=num_features*5).tolist()

    bounds = polygon.bounds
    
    rand_x = np.random.uniform(low=bounds[0], high=bounds[2], size=num_features*20)
    rand_y = np.random.uniform(low=bounds[1], high=bounds[3], size=num_features*20)
    non_deposit_x = []
    non_deposit_y = []
    
    for x, y in zip(rand_x, rand_y):
        if len(non_deposit_x) == num_features*5:
            print(f'{num_features*5} random samples generated successfully!')
            break
        p = Point((x, y))
        if polygon.contains(p):
            non_deposit_x.append(x)
            non_deposit_y.append(y)
    
    plate_ids = Utils.get_plate_id(non_deposit_x, non_deposit_y)

    # index, lon, lat, time, plate id
    data = []
    
    for i in range(num_features*5):
        data.append([i, non_deposit_x[i], non_deposit_y[i], times[i], plate_ids[i]])
    data = np.array(data)
    
    return data

def generate_samples(polygon_path, dist_x, dist_y, start_time, end_time, map_extent):
    polygon_obj = shapefile.Reader(polygon_path)
    feature = polygon_obj.shapeRecords()[0]
    first = feature.shape.__geo_interface__
    polygon = shape(first)
    
    x = np.arange(map_extent[0], map_extent[1], dist_x)
    y = np.arange(map_extent[2], map_extent[3], dist_y)
    nx = len(x)
    ny = len(y)
    xs, ys = np.meshgrid(x, y)

    sample_x = []
    sample_y = []
    sample_mask = []
    
    for xx, yy in zip(xs.flatten(), ys.flatten()):
        p = Point((xx, yy))
        if polygon.contains(p):
            sample_x.append(xx)
            sample_y.append(yy)
            sample_mask.append(True)
        else:
            sample_mask.append(False)
    
    mask_x = np.array([xs.flatten()]).T
    mask_y = np.array([ys.flatten()]).T
    sample_mask = np.array([sample_mask]).T
    mask_coords = np.hstack((mask_x, mask_y, sample_mask))
    
    plate_ids = Utils.get_plate_id(sample_x, sample_y)

    # index, lon, lat, time, plate id
    sample_data = []
    k = 0
    
    for i in range(start_time, end_time):
        for j in range(len(sample_x)):
            sample_data.append([k, sample_x[j], sample_y[j], i, plate_ids[j]])
            k += 1
    sample_data = np.array(sample_data)
    
    return sample_data, mask_coords, nx, ny
                         
def generate_trench_points(start_time, end_time, time_step):
    trench_data=[]
    trench_points = Utils.get_trench_points(0,-85,5,-70,-60) #subduction points in south america 
    i=0
    
    for t in range(start_time, end_time, time_step):
        for index, p in trench_points.iterrows():
            trench_data.append([i, p['trench_lon'], p['trench_lat'], t, p['trench_pid']]) 
            i+=1
            
    return trench_data    

def save_data(data,fn):
    # data are ready and write them to file
    with open(fn,'w+') as f:
        f.write('index,lon,lat,age,plate_id\n')
        for row in data:
            #print(row)
            if row:
                f.write('{0:d}, {1:.2f}, {2:.2f}, {3:d}, {4:d}'.format(
                    int(row[0]),float(row[1]),float(row[2]),int(row[3]),int(row[4])))
            f.write('\n')

    print(f'The data have been written into {fn} successfully!')                               

if __name__ == '__main__':
    start_time = parameters['time']['start']
    end_time = parameters['time']['end']
    time_step =  parameters['time']['step']
    deposit_path = parameters['deposit_path']
    polygon_path = parameters['region_of_interest_polygon']
    convergence_file = parameters['convergence_data_dir']+parameters['convergence_data_filename_prefix']+'_0.00.csv'
    num_features = pd.read_csv(convergence_file).shape[1]
    map_extent = parameters['map_extent']
    dist_x = 1
    dist_y = 1

    data = process_real_deposits(deposit_path, start_time, end_time, time_step)
    random_data = generate_non_deposits(polygon_path, start_time, end_time, num_features)
    sample_data = generate_samples(polygon_path, dist_x, dist_y, start_time, end_time, map_extent)
    trench_data = generate_trench_points(start_time, end_time, time_step)

    all_data = data+random_data+trench_data
    for i in range(len(all_data)):
        all_data[i][0] = i # assign correct indices

    save_data(all_data, 'coregistration_input_data_example.csv')
