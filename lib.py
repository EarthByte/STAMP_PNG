import cv2
import geopandas as gpd
import math
from netCDF4 import Dataset
import numpy as np
import os, sys
import pandas as pd
from ptt.subduction_convergence import subduction_convergence_over_time
import pygplates
import scipy.spatial
import shapefile
from shapely.geometry import LineString, Point
from tqdm.notebook import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from parameters_muller2019_v27 import parameters
# from parameters_muller2016_v27 import parameters
# ------------------------------------------------

# convergence kinematic features
def degree_to_straight_distance(degree):
    
    return math.sin(math.radians(degree)) / math.sin(math.radians(90 - degree/2.))

def query_raster(raster_name, lons, lats, search_radius, ball=False, verbose=True):    
    points=[pygplates.PointOnSphere((float(row[1]), float(row[0]))).to_xyz() for row in zip(lons, lats)]
    rasterfile = Dataset(raster_name, 'r')
            
    z = rasterfile.variables['z'][:] # masked array
    
    if verbose:
        print(raster_name)
    
    if len(z.shape) == 3:
        z = cv2.resize(z.transpose(1, 2, 0), dsize=(3601, 1801), interpolation=cv2.INTER_NEAREST)
    else:
        z = cv2.resize(z, dsize=(3601, 1801), interpolation=cv2.INTER_NEAREST)
    z = z.flatten()
    
    # query the tree 
    if not ball:
        global grid_points
        grid_points = np.asarray(grid_points)
        z_idx = ~np.isnan(z)
        z = z[z_idx]
        grid_tree = scipy.spatial.cKDTree(grid_points[z_idx])
        dists, indices = grid_tree.query(points, k=1, distance_upper_bound=degree_to_straight_distance(search_radius))
        z = np.append(z, [np.nan])
        
        return z[indices]
    else:
        # ball query the grid tree
        # construct the grid tree
        grid_x, grid_y = np.mgrid[-90:90:1801j, -180:180:3601j]
        grid_points = [pygplates.PointOnSphere((float(row[0]), float(row[1]))).to_xyz() for row in zip(grid_x.flatten(), grid_y.flatten())]
        full_grid_tree = scipy.spatial.cKDTree(grid_points)
        all_neighbors = full_grid_tree.query_ball_point(
            points, 
            degree_to_straight_distance(search_radius))
        ret = []
        for neighbors in all_neighbors:
            if len(neighbors)>0: # and (~np.isnan(z[neighbors])).any():
                ret.append(np.nanmean(z[neighbors]))
            else:
                ret.append(np.nan)
                
        return ret

def plate_temp(age, z, PLATE_THICKNESS):
    'Computes the temperature in a cooling plate for age = t\
    and at a depth = z.'

    KAPPA = 0.804E-6
    T_MANTLE = 1350.0
    T_SURFACE = 0.0
    SEC_PR_MY = 3.15576e13

    t = T_SURFACE

    sum = 0
    sine_arg = math.pi*z/PLATE_THICKNESS
    exp_arg = -KAPPA*math.pi*math.pi*age*SEC_PR_MY/(PLATE_THICKNESS*PLATE_THICKNESS)
    
    for k in range(1, 20):
        sum = sum + np.sin(k*sine_arg)*np.exp(k*k*exp_arg)/k

    if age <= 0.0:
        t = T_MANTLE*np.ones(z.shape)
    else:
        t = t+2.0*sum*(T_MANTLE-T_SURFACE)/math.pi+(T_MANTLE-T_SURFACE)*z/PLATE_THICKNESS
    
    return t

def plate_isotherm_depth(age, temp, *vartuple) :
    'Computes the depth to the temp - isotherm in a cooling plate mode.\
    Solution by iteration. By default the plate thickness is 125 km as\
    in Parsons/Sclater. Change given a 3rd parameter.'

    if len(vartuple) != 0:
        PLATE_THICKNESS_KM = vartuple[0]
    else :
        PLATE_THICKNESS_KM = 125

    PLATE_THICKNESS = PLATE_THICKNESS_KM * 1000
    
    # default depth is 0
    z = 0

    if age <= 0.0:
        z_try = 0
        done = 1
    else:
        z_too_small = 0.0
        z_too_big = PLATE_THICKNESS
        done = 0
        n_try = 0

    while done != 1 and n_try < 20:
        n_try += 1
        z_try = 0.5 * (z_too_small + z_too_big)
        t_try = plate_temp(age, z_try, PLATE_THICKNESS)
        t_wrong = temp - t_try
        if t_wrong < -0.001:
            z_too_big = z_try
        elif t_wrong > 0.001:
            z_too_small = z_try
        else:
            done = 1

        z = z_try
        
    return z

def trench_points_features(start_time, end_time, time_step, conv_dir, conv_prefix, conv_ext, plate_motion_model, random_state=1):
    time_steps = list(range(start_time, end_time+1, time_step))
    
    count = 0
    
    for file in os.listdir(conv_dir):
        if not file.startswith('features_target_extent'):
            count += 1
    
    if os.path.exists(conv_dir) and count == len(time_steps):
        print('The kinematic features have already been extracted!')
        print(f'Please check {conv_dir}')
        
        return
    else:
        print('Extracting trench points and their kinematic features ...')
    
        if not os.path.exists(conv_dir):
            os.makedirs(conv_dir)
        if plate_motion_model == 'muller2016':
            rotation_files = parameters['rotation_files']
            topology_files = parameters['topology_files']
        elif plate_motion_model == 'muller2019':
            rotation_files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(parameters['rotation_dir']) for f in filenames]
            topology_files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(parameters['topology_dir']) for f in filenames]
        
        kwargs = {
            'output_distance_to_nearest_edge_of_trench':True,
            'output_distance_to_start_edge_of_trench':True,
            'output_convergence_velocity_components':True,
            'output_trench_absolute_velocity_components':True,
            'output_subducting_absolute_velocity':True,
            'output_subducting_absolute_velocity_components':True,
            'output_trench_normal':True
        }
    
        subduction_convergence_over_time(
            conv_dir + '/' + conv_prefix,
            conv_ext,
            rotation_files,
            topology_files,
            math.radians(parameters['threshold_sampling_distance_degrees']),
            start_time,
            end_time,
            time_step,
            parameters['velocity_delta_time'],
            parameters['anchor_plate_id'],
            output_gpml_filename = None,
            **kwargs
        )
        
        time_steps = list(range(start_time, end_time+1, time_step))
        print('Extracting grid features ...')
        
        kinematic_features_lst = [
            'trench_lon',
            'trench_lat',
            'conv_rate_cm_yr',
            'conv_angle_deg',
            'trench_abs_rate_cm_yr',
            'trench_abs_angle_deg',
            'arc_len_deg',
            'trench_norm_deg',
            'subducting_pid',
            'trench_pid',
            'dist_nearest_edge_deg',
            'dist_from_start_deg',
            'conv_ortho_cm_yr',
            'conv_paral_cm_yr',
            'trench_abs_ortho_cm_yr',
            'trench_abs_paral_cm_yr',
            'subducting_abs_rate_cm_yr',
            'subducting_abs_angle_deg',
            'subducting_abs_ortho_cm_yr',
            'subducting_abs_paral_cm_yr',
            'trench_norm_x_cm_yr',
            'trench_norm_y_cm_yr',
            'trench_norm_z_cm_yr'
            ]
        
        for time in tqdm(time_steps):
            trench_points = pd.read_csv(f'{conv_dir}/{conv_prefix}_{time}.00.{conv_ext}', sep=' ', header=None, names=kinematic_features_lst)
            
            for grid in parameters['grid_files']:
                grid_name = grid[1]
                grid_data = query_raster(
                    grid[0].format(time=time),
                    trench_points.iloc[:, 0],
                    trench_points.iloc[:, 1],
                    search_radius = 3, # try to find the nearest valid data within the search radius
                    ball=True
                )
    
                trench_points[grid_name] = grid_data
    
                if grid_name == 'seafloor_age_ma':
                    thickness = [None]*len(grid_data)
                    T1 = 1150.
                    for i in range(len(grid_data)):
                        thickness[i] = plate_isotherm_depth(grid_data[i], T1)
    
                    # to convert arc_length from degrees on a sphere to m (using Earth's radius = 6371000 m)
                    arc_length_m = 2*math.pi*6371000*trench_points.arc_len_deg/360
                    # calculate subduction volume (in m^3 per year)
                    subduction_volume_m3y = trench_points.conv_ortho_cm_yr/100*thickness*arc_length_m
                    # calculate Subduciton Volume (slab flux) (km^3/yr)
                    subduction_volume_km3y = subduction_volume_m3y/1e9
                    subduction_volume_km3y[subduction_volume_km3y<0] = 0
                    trench_points['subduction_volume_km3_yr'] = subduction_volume_km3y
            
            iter_imputer = IterativeImputer(random_state=random_state)
            trench_points_imputed = pd.DataFrame(iter_imputer.fit_transform(trench_points), columns=trench_points.columns)
            trench_points_imputed.to_csv(f'{conv_dir}/{conv_prefix}_{time}.00.{conv_ext}', index=False, float_format='%.4f', na_rep='nan')
    
        print('Completed successfully!')
        print(f'The results have been saved in {conv_dir}')
        
        return
# -----------------------------------------------------

# sampling    
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

def get_plate_id(lons, lats, plate_motion_model):
    if plate_motion_model == 'muller2016':
        rotation_model = pygplates.RotationModel(parameters['rotation_files'])
    elif plate_motion_model == 'muller2019':
        rotation_model = pygplates.RotationModel([os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(parameters['rotation_dir']) for f in filenames])
    
    static_polygons = pygplates.FeatureCollection(parameters['static_polygons_file'])
    p_len = len(lons)
    
    assert p_len == len(lats), 'The lons and lats must have the same length.'
    point_features = []
    
    for i in range(p_len):
        point = pygplates.PointOnSphere(float(lats[i]), float(lons[i]))
        point_feature = pygplates.Feature()
        point_feature.set_geometry(point)
        point_feature.set_name(str(i))
        point_features.append(point_feature)

    plate_ids = [np.nan]*p_len
    # partition features
    points = pygplates.partition_into_plates(static_polygons, rotation_model, point_features)
    
    for p in points:
        plate_ids[int(p.get_name())] = p.get_reconstruction_plate_id()
        
    return plate_ids

def get_recon_ccords(lons, lats, plate_motion_model, time): # lons and lats must be list or scalar
    if np.isscalar(time):
        time = [time]

    if not np.isscalar(lons):
        time = time * len(lons)

    if np.isscalar(lons):
        lons = [lons]
        
    if np.isscalar(lons):
        lats = [lats]

    lons_lats_recon = []
    
    if all(t == 0 for t in time):
        for lon, lat in zip(lons, lats):
            lons_lats_recon.append((lat, lon))
        
        return lons_lats_recon
        
    if plate_motion_model == 'muller2016':
        rotation_model = pygplates.RotationModel(parameters['rotation_files'])
    elif plate_motion_model == 'muller2019':
        rotation_model = pygplates.RotationModel([os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(parameters['rotation_dir']) for f in filenames])
    
    plate_ids =  get_plate_id(lons, lats, plate_motion_model)

    for i, plate_id in enumerate(plate_ids):
        point_to_rotate = pygplates.PointOnSphere((float(lats[i]), float(lons[i]))) # lat, lon
        finite_rotation = rotation_model.get_rotation(int(time[i]), plate_id) # time, plate_id
        geom = finite_rotation * point_to_rotate
        lons_lats_recon.append(tuple(map(lambda x: round(x, 4), geom.to_lat_lon()))) # round tuple
        
    return lons_lats_recon

def process_real_deposits(deposit_path, start_time, end_time, time_step, plate_motion_model): # path to the shapefile of mineral occurrences
    if not os.path.isfile(deposit_path):
        sys.exit('File not found!')
    reader = shapefile.Reader(deposit_path)
    recs = reader.records()
    min_occ_num = len(recs)
    # longitude
    lons = np.array(recs)[:, 3].tolist()
    # latitude
    lats = np.array(recs)[:, 4].tolist()
    # time
    times = get_time_from_age(np.array(recs)[:, 14], start_time, end_time, time_step) # get integer ages
    # plate id
    plate_ids = get_plate_id(lons, lats, plate_motion_model)
    # reconstructed coords
    lons_lats_recon = get_recon_ccords(lons, lats, plate_motion_model, times)

    # index, lon, lat, time, plate id, recon lon, recon lat
    data = []
    
    for i in range(min_occ_num):
        data.append([i, lons[i], lats[i], times[i], plate_ids[i], lons_lats_recon[i][1], lons_lats_recon[i][0]])
    data = np.array(data)

    data = pd.DataFrame(data, columns=['index', 'lon', 'lat', 'age', 'plate_id', 'lon_recon', 'lat_recon'])
    data = data.astype({'index': int, 'lon': float, 'lat': float, 'plate_id': int, 'age': int, 'lon_recon': float, 'lat_recon': float})
    
    return data

def get_subduction_geometries(subduction_geoms, shared_boundary_sections):
    for shared_boundary_section in shared_boundary_sections:
        if shared_boundary_section.get_feature().get_feature_type() != pygplates.FeatureType.gpml_subduction_zone:
            continue
        for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
            subduction_polarity = shared_sub_segment.get_feature().get_enumeration(pygplates.PropertyName.gpml_subduction_polarity)
            if subduction_polarity == 'Left':
                subduction_geoms.append((shared_sub_segment.get_resolved_geometry(), -1))
            else:
                subduction_geoms.append((shared_sub_segment.get_resolved_geometry(), 1))

    return

# genrate buffer zones surrounding polylines (segment by segment) considering aspect to generate buffer zones
def generate_buffer_zones(subduction_geoms, width): # subduction_geoms generated by get_subduction_geometries
    lines_list = []
    buffer_zones_list = []
    buffer_zones_df_list = []

    # add an appropriate vertex to the lines on anti-meridian
    for geom, aspect in subduction_geoms:
        index_list = []
        vertex_list = []
        xy = geom.to_lat_lon_array()
        num_xy = xy.shape[0]
        for i in range(num_xy-1):
            if xy[i, 1] * xy[i+1, 1] < 0 and 180 - abs(xy[i, 1]) < abs(xy[i, 1]):
                index_list.append(i+1)
                m = (xy[i+1, 0] - xy[i, 0]) / (xy[i+1, 1] - xy[i, 1])
                b = xy[i, 0] - (m * xy[i, 1])
                pos_vertex = [m*180+b, 180]
                neg_vertex = [m*-180+b, -180]
                if xy[i, 1] > 0:
                    vertex_list.append([pos_vertex, neg_vertex])
                else:
                    vertex_list.append([neg_vertex, pos_vertex])

        if len(index_list) > 0:
            xy = np.insert(xy, index_list, np.array(vertex_list[0]), 0)

        # split the line where it passes over anti-meridian
        index_list = []
        num_xy = xy.shape[0]
        for i in range(num_xy-1):
            if xy[i, 1] * xy[i+1, 1] < 0 and 180 - abs(xy[i, 1]) < abs(xy[i, 1]):
                index_list.append(i+1)
        xy_split = np.split(xy, index_list)
        xy_split.append(aspect)
        lines_list.append(xy_split)

    for line in lines_list:
        aspect = line[-1]
        for i in range(len(line)-1):
            buffer_zones = []
            for j in range(line[i].shape[0]-1):
                buffer_zone = gpd.GeoSeries(LineString([(line[i][j, 1], line[i][j, 0]), (line[i][j+1, 1], line[i][j+1, 0])]), crs='EPSG:4326').buffer(-1*aspect*width, cap_style=2, single_sided=True)
                buffer_zones.append(buffer_zone)
            
            # convert the list of geoseries objects to a geodataframe
            buffer_zones_df = gpd.GeoDataFrame(gpd.GeoSeries(buffer_zones[0]))
            for k in range(1, len(buffer_zones)):
                buffer_zones_df.loc[k] = gpd.GeoSeries(buffer_zones[k])
            buffer_zones_df = buffer_zones_df.rename(columns={0: 'geometry'}).set_geometry('geometry')
            
            for m in range(buffer_zones_df.shape[0]):
                buffer_zone_dis_ch = buffer_zones_df.iloc[m:m+2].dissolve().convex_hull
                buffer_zones_list.append(buffer_zone_dis_ch)
            buffer_zones_df_list.append(buffer_zones_df)
    
    buffer_zones_list_df = gpd.GeoDataFrame(gpd.GeoSeries(buffer_zones_list[0]))
    for m in range(1, len(buffer_zones_list)):
        buffer_zones_list_df.loc[m] = gpd.GeoSeries(buffer_zones_list[m])
    buffer_zones_list_df = buffer_zones_list_df.rename(columns={0: 'geometry'}).set_geometry('geometry')
    buffer_zones_list_dis = buffer_zones_list_df.dissolve()

    return buffer_zones_df_list, buffer_zones_list_dis

# generate random samples inside buffer zones at a specific time step
def generate_random_samples(buffer_zones_lst, start_time, end_time, time_step, num_features, num_features_factor, rand_factor, plate_motion_model, random_state=1):
    '''
    rand_factor: a factor which is multiplied by the number of samples (num_samples) and
    determines the total number of samples to be generated from which random samples are selected.
    
    if the calculated number of samples per time step is less than one,
    the code uniformly distributes samples through the time period.
    '''
    
    time_steps = list(range(start_time, end_time+1, time_step))
    time_steps_random = time_steps.copy()
    num_rand_samples = num_features_factor * num_features
    num_time_steps = ((end_time - start_time) / time_step) + 1
    num_rand_samples_step = round(num_rand_samples / num_time_steps)
    
    if num_rand_samples_step < 1:
        num_rand_samples_step = 1
        np.random.seed(random_state)
        time_steps_random = np.random.random_integers(start_time, end_time, num_rand_samples).tolist()
    
    random_data_lst = []
    
    for time in time_steps_random:
        buffer_zone = buffer_zones_lst[time_steps.index(time)]
    
        bounds = buffer_zone.bounds
        x_min = bounds.loc[0]['minx']
        x_max = bounds.loc[0]['maxx']
        y_min = bounds.loc[0]['miny']
        y_max = bounds.loc[0]['maxy']
    
        if x_min < -180:
            x_min = -180
        if x_max > 180:
            x_max = 180
        if y_min < -90:
            y_min = -90
        if y_max > 90:
            y_max = 90
    
        rand_x_list = []
        rand_y_list = []
    
        for n in range(1, rand_factor):
            if len(rand_x_list) < num_rand_samples_step:
                rand_x = np.random.uniform(low=x_min, high=x_max, size=n*num_rand_samples_step)
                rand_y = np.random.uniform(low=y_min, high=y_max, size=n*num_rand_samples_step)
                for x, y in zip(rand_x, rand_y):
                    if len(rand_x_list) == num_rand_samples_step:
                        break
                    p = Point((x, y))
                    if p.within(buffer_zone.geometry[0]):
                        rand_x_list.append(x)
                        rand_y_list.append(y)
            else:
                break
    
        plate_ids = get_plate_id(rand_x_list, rand_y_list, plate_motion_model)
        # index, lon, lat, time, plate id
        data = []
        for i in range(num_rand_samples_step):
            data.append([rand_x_list[i], rand_y_list[i], time, plate_ids[i]])
            
        data = np.array(data)
        random_data_lst.append(data)
        
    # save the attributes of random samples
    random_data = np.vstack(random_data_lst)
    index_lst = list(range(random_data.shape[0]))
    index_lst = np.array(index_lst).reshape(-1, 1)
    random_data = np.hstack([index_lst, random_data])
    random_data = pd.DataFrame(random_data, columns=['index', 'lon','lat','age','plate_id'])
    random_data = random_data.astype({'index': int, 'lon': float, 'lat': float, 'plate_id': int, 'age': int})
    
    return time_steps_random, random_data

# generate sampling points inside buffer zones at a specific time step
def generate_samples(buffer_zone, dist_x, dist_y, time, plate_motion_model):
    bounds = buffer_zone.bounds
    x_min = bounds.loc[0]['minx']
    x_max = bounds.loc[0]['maxx']
    y_min = bounds.loc[0]['miny']
    y_max = bounds.loc[0]['maxy']
    
    if x_min < -180:
        x_min = -180
    if x_max > 180:
        x_max = 180
    if y_min < -90:
        y_min = -90
    if y_max > 90:
        y_max = 90
    
    x = np.arange(x_min, x_max, dist_x)
    y = np.arange(y_min, y_max, dist_y)
    nx = len(x)
    ny = len(y)
    xs, ys = np.meshgrid(x, y)

    sample_x = []
    sample_y = []
    sample_mask = []
    
    for xx, yy in zip(xs.flatten(), ys.flatten()):
        p = Point((xx, yy))
        if p.within(buffer_zone.geometry[0]):
            sample_x.append(xx)
            sample_y.append(yy)
            sample_mask.append(True)
        else:
            sample_mask.append(False)
    
    mask_x = np.array([xs.flatten()]).T
    mask_y = np.array([ys.flatten()]).T
    sample_mask = np.array([sample_mask]).T
    mask_coords = np.hstack((mask_x, mask_y, sample_mask))
    mask_coords = pd.DataFrame(mask_coords, columns=['lon','lat', 'include'])
    mask_coords = mask_coords.astype({'lon': float, 'lat': float, 'include': bool})
    
    plate_ids = get_plate_id(sample_x, sample_y, plate_motion_model)

    # index, lon, lat, time, plate id
    sample_data = []
    for i in range(len(sample_x)):
        sample_data.append([i, sample_x[i], sample_y[i], time, plate_ids[i]])
    sample_data = np.array(sample_data)
    sample_data = pd.DataFrame(sample_data, columns=['index', 'lon','lat','age','plate_id'])
    sample_data = sample_data.astype({'index': int, 'lon': float, 'lat': float, 'plate_id': int, 'age': int})

    
    
    return sample_data, mask_coords, nx, ny
# -----------------------------------------

# coregistration
def straight_distance_to_degree(dist):
    deg = 45.0  # initial guess for x
    epsilon = 1e-6  # desired precision of the solution

    while True:
        f = dist * math.sin(math.radians(90 - deg/2.)) - math.sin(math.radians(deg))
        f_prime = dist * (math.cos(math.radians(90 - deg/2.))/2.) - math.cos(math.radians(deg))
        deg -= f / f_prime

        if abs(f) < epsilon:
            
            return deg

def coregistration(coreg_input_dir, coreg_output_dir, coreg_input_files, conv_dir, conv_prefix, conv_ext, time_steps, search_radius):
    positive_data_file = coreg_output_dir + coreg_input_files[0]
    unlabelled_data_file = coreg_output_dir + coreg_input_files[1]
    target_points_out_files_lst = []
    
    for time in time_steps:
        target_points_out_files_lst.append(coreg_output_dir + coreg_input_files[2] + f'_{time}_Ma.csv')
    
    if os.path.isfile(positive_data_file) and os.path.isfile(unlabelled_data_file)\
    and all([os.path.isfile(file) for file in target_points_out_files_lst]):
        print('Data points have already been coregistered!')
    else:
        # run the coregistration script
        print('Coregistration in progress ...')
    
        if not os.path.exists(coreg_output_dir):
            os.makedirs(coreg_output_dir)
    
        coreg_input_files_lst = os.listdir(coreg_input_dir)
        sample_points_files_lst = []
        
        for file in coreg_input_files_lst:
            if file.startswith(tuple(coreg_input_files)):
                sample_points_files_lst.append(coreg_input_dir + file)
                
        trench_points_present_day = pd.read_csv(f'{conv_dir}/{conv_prefix}_0.00.{conv_ext}', index_col=False)
        trench_points_columns = trench_points_present_day.columns.tolist()
        trench_points_columns.append('distance_deg')
    
        for sample_points_file in tqdm(sample_points_files_lst):
            sample_points_file_tail = os.path.split(sample_points_file)[-1]
            coreg_output_file = coreg_output_dir + sample_points_file_tail
            sample_points = pd.read_csv(sample_points_file, index_col=False)
            df_empty = pd.DataFrame(np.empty((len(sample_points), len(trench_points_columns))), columns=trench_points_columns)
            sample_points_coreg = pd.concat([sample_points, df_empty], axis=1).reset_index(drop=True)
            ages = sorted(sample_points['age'].unique())
    
            for age in ages:
                trench_points = pd.read_csv(f'{conv_dir}/{conv_prefix}_{age}.00.{conv_ext}', index_col=False)
                trench_points_coords_3d = [pygplates.PointOnSphere((row[1], row[0])).to_xyz() for _, row in trench_points.iterrows()]
                trench_points_tree = scipy.spatial.cKDTree(trench_points_coords_3d)
                sample_points_age = sample_points.loc[sample_points['age'] == age]
    
                if os.path.split(sample_points_file)[-1] == 'mineral_occurrences.csv':
                    sample_points_age_coords_3d = [pygplates.PointOnSphere((row[6], row[5])).to_xyz() for _, row in sample_points_age.iterrows()]
                else:
                    sample_points_age_coords_3d = [pygplates.PointOnSphere((row[2], row[1])).to_xyz() for _, row in sample_points_age.iterrows()]
    
                dists, indices = trench_points_tree.query(sample_points_age_coords_3d, k=1, distance_upper_bound=degree_to_straight_distance(search_radius))
    
                for index_1, index_2, dist in zip(sample_points_age.index, indices, dists):
                    if dist == np.inf:
                        sample_points_coreg.drop(index=index_1, inplace=True)
                        if os.path.split(sample_points_file)[-1].startswith('target_points'):
                            mask_coords_file = f'{coreg_input_dir}/mask_{age}_Ma.csv'
                            mask_coords = pd.read_csv(mask_coords_file, index_col=False)
                            for i in range(mask_coords.shape[0]):
                                if sample_points_age['lon'][index_1] == mask_coords['lon'][i] and sample_points_age['lat'][index_1] == mask_coords['lat'][i]:
                                    mask_coords['include'][i] = False
                            mask_coords.to_csv(mask_coords_file, index=False)
                        continue
                    else:
                        trench_points_temp = trench_points.iloc[index_2]
                        trench_points_temp['distance_deg'] = round(straight_distance_to_degree(dist), 4)
                        sample_points_coreg.loc[sample_points_coreg['index'] == index_1, trench_points_columns] = trench_points_temp.tolist()
            
            if sample_points_coreg.isnull().values.any():
                print(f'Warning: {sample_points_file} contains NaN values!')
            
            sample_points_coreg.to_csv(coreg_output_file, index=False)
            print(f'Coregistration completed for {sample_points_file_tail}')
            
        print('\nCoregistration completed successfully!')
        print(f'The results have been saved in {coreg_output_dir}')
        
    return

def coregistration_point(points, conv_dir, conv_prefix, conv_ext, output_dir, file_prefix, time_steps, search_radius, plate_motion_model):
    points_files_lst = []
    
    for index in points['index']:
        points_files_lst.append(output_dir + file_prefix + f'_{index}.csv')
    
    if all([os.path.isfile(file) for file in points_files_lst]):
        print('Data points have already been coregistered!')
    else:
        # run the coregistration script
        print('Coregistration in progress ...')
    
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for file in points_files_lst:
            file_index = points_files_lst.index(file)
            
            if not os.path.exists(file):
                trench_points_present_day = pd.read_csv(f'{conv_dir}/{conv_prefix}_0.00.{conv_ext}', index_col=False)
                trench_points_columns = trench_points_present_day.columns.tolist()
                trench_points_columns.append('distance_deg')

                before_mineralisation = []
                
                for time in time_steps:
                    if points['age'][file_index] >= time:
                        before_mineralisation.append(False)
                    else:
                        before_mineralisation.append(True)
        
                point_lon = points['lon'][file_index]
                point_lat = points['lat'][file_index]
                
                point_recon_lats_lons = []
                
                for time in time_steps:
                    point_recon_lats_lons.append(get_recon_ccords([point_lon], [point_lat], plate_motion_model, time)[0])
                        
                points_coreg = pd.DataFrame()
                points_coreg['age'] = time_steps
                points_coreg['before_mineralisation'] = before_mineralisation
                points_coreg['lon'] = [coords[1] for coords in point_recon_lats_lons]
                points_coreg['lat'] = [coords[0] for coords in point_recon_lats_lons]
                points_coreg['valid'] = True
                
                df_empty = pd.DataFrame(np.empty((len(points_coreg), len(trench_points_columns))), columns=trench_points_columns)
                points_coreg = pd.concat([points_coreg, df_empty], axis=1).reset_index(drop=True)
            
                for time in time_steps:
                    trench_points = pd.read_csv(f'{conv_dir}/{conv_prefix}_{time}.00.{conv_ext}', index_col=False)
                    trench_points_coords_3d = [pygplates.PointOnSphere((row[1], row[0])).to_xyz() for _, row in trench_points.iterrows()]
                    trench_points_tree = scipy.spatial.cKDTree(trench_points_coords_3d)
                    point_age = points_coreg.loc[points_coreg['age'] == time]
                    point_age_coords_3d = [pygplates.PointOnSphere((float(point_age['lat']), float(point_age['lon']))).to_xyz()]
                    dists, indices = trench_points_tree.query(point_age_coords_3d, k=1, distance_upper_bound=degree_to_straight_distance(search_radius))
        
                    for index, dist in zip(indices, dists):
                        if dist == np.inf:
                            point_age['valid'] = False
                            point_age[trench_points_columns] = np.nan
                        else:
                            trench_points_temp = trench_points.iloc[index]
                            trench_points_temp['distance_deg'] = round(straight_distance_to_degree(dist), 4)
                            point_age[trench_points_columns] = trench_points_temp.tolist()
                    
                    points_coreg.loc[points_coreg['age'] == time] = point_age
                        
                points_coreg.to_csv(f'{file}', index=False)
                print(f'Coregistration completed for index {file_index} out of {len(points)-1}')               
            else:
                print(f'Coregistred file for index {file_index} already exists!')
            
        print('Coregistration completed successfully!')
        print(f'The results have been saved in {output_dir}')
        
    return
# --------

# visualisation
def get_subduction_teeth(lons, lats, tesselation_degrees=2, triangle_base_length=1, triangle_aspect=-1):
    polyline = pygplates.PolylineOnSphere(zip(lats, lons))
    tessellated_polyline = polyline.to_tessellated(math.radians(0.5))
    points = tessellated_polyline.to_lat_lon_list()
    lats, lons = zip(*points)
    distance = tesselation_degrees 
    teeth = []
    PA = np.array([lons[0], lats[0]])
    
    for lon, lat in zip(lons[1:], lats[1:]):
        PB = np.array([lon, lat])
        AB_dist = np.sqrt((PB[0]-PA[0])**2 + (PB[1]-PA[1])**2)
        distance += AB_dist
        if distance > tesselation_degrees:
            distance = 0
            AB_norm = (PB - PA)/AB_dist
            AB_perpendicular = np.array([AB_norm[1], -AB_norm[0]]) # perpendicular to line A->B
            B0 = PA + triangle_base_length*AB_norm # new B
            C0 = PA + 0.5*triangle_base_length*AB_norm # middle point between A and B
            # project point along normal vector
            C = C0 + triangle_base_length*triangle_aspect*AB_perpendicular
            teeth.append([PA, B0, C]) # three vertices of the triagle

        PA = PB
        
    return teeth

def subduction_teeth(triangle_base_length, time, plate_motion_model):
    if plate_motion_model == 'muller2016':
        rotation_files = parameters['rotation_files']
        topology_files = parameters['topology_files']
    elif plate_motion_model == 'muller2019':
        rotation_files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(parameters['rotation_dir']) for f in filenames]
        topology_files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(parameters['topology_dir']) for f in filenames]
    
    # use pygplates to resolve the topologies
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(topology_files, rotation_files, resolved_topologies, time, shared_boundary_sections)
    # subduction zones
    subduction_geoms = []
    get_subduction_geometries(subduction_geoms, shared_boundary_sections)
    
    for geom, aspect in subduction_geoms:
        lat, lon = zip(*(geom.to_lat_lon_list()))
        teeth = get_subduction_teeth(lon, lat, triangle_base_length=triangle_base_length, triangle_aspect=aspect)
        
    return teeth
