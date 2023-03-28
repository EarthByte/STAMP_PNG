import cv2
import errno, glob, json, math, os, pprint, requests
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from parameters import parameters
import pygplates
import scipy.spatial

# def load_config(fn):
#     with open(fn) as json_file:
#         data = json.load(json_file)
        
#         for key in data:
#             parameters[key] = data[key]

def print_parameters():
    print('Below are the parameters in use. Check the parameters carefully to ensure a sucussful run.')
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(parameters)

def get_parameter(name):
    return parameters[name]

# expand *.rot, *.gpml
def get_files(names):
    files = []
    
    for f in names:
        files += glob.glob(f)
        
    return files

# input: degrees between two points on sphere
# output: straight distance between the two points (assume the earth radius is 1)
# to get the kilometers, use the return value to multiply by the real earth radius
def degree_to_straight_distance(degree):
    return math.sin(math.radians(degree)) / math.sin(math.radians(90 - degree/2.))

# construct the grid tree
grid_x, grid_y = np.mgrid[-90:91, -180:181]
grid_points = [pygplates.PointOnSphere((float(row[0]), float(row[1]))).to_xyz() for row in zip(grid_x.flatten(), grid_y.flatten())]
full_grid_tree = scipy.spatial.cKDTree(grid_points)

def query_raster(raster_name, lons, lats, region=10, ball=False):
    points=[pygplates.PointOnSphere((float(row[1]), float(row[0]))).to_xyz() for row in zip(lons, lats)]
    rasterfile = Dataset(raster_name, 'r')
    z = rasterfile.variables['z'][:] # masked array
    zz = cv2.resize(z, dsize=(361, 181), interpolation=cv2.INTER_CUBIC)
    lon_actual_range = rasterfile.variables['lon'].actual_range
    
    if int(lon_actual_range[0]) == 0: # roll 180 if the grid ranges from 0 to 360
        zz = np.roll(zz, 180)
    
    z = np.ma.asarray(zz.flatten())
    
    # query the tree 
    if not ball:
        global grid_points
        grid_points = np.asarray(grid_points)
        z_idx = ~np.isnan(z)
        z = z[z_idx]
        grid_tree = scipy.spatial.cKDTree(grid_points[z_idx])
        dists, indices = grid_tree.query(points, k=1, distance_upper_bound=degree_to_straight_distance(region))
        z = np.append(z, [np.nan])
        return z[indices]
    else:
        # ball query the grid tree
        all_neighbors = full_grid_tree.query_ball_point(
                points, 
                degree_to_straight_distance(region))
        ret=[]
        for neighbors in all_neighbors: 
            if len(neighbors)>0 and (~np.isnan(z[neighbors])).any():
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

    if len(vartuple) != 0 :
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

def get_convergence_dir():
    return parameters['convergence_data_dir'] + '/'

def get_trench_points(age, top_left_lon=None, top_left_lat=None,
                      bottom_right_lon=None, bottom_right_lat=None):
    conv_dir = get_convergence_dir()
    conv_prefix = parameters['convergence_data_filename_prefix']
    conv_ext = parameters['convergence_data_filename_ext']
    
    fn = conv_dir + conv_prefix + f'_{age:.2f}.' + conv_ext
    
    if not os.path.isfile(fn):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fn)

    data = pd.read_csv(fn)
    
    if [top_left_lon, top_left_lat, bottom_right_lon, bottom_right_lat].count(None) == 0:
        data = data[data['trench_lon']>top_left_lon]
        data = data[data['trench_lon']<bottom_right_lon]
        data = data[data['trench_lat']>bottom_right_lat]
        data = data[data['trench_lat']<top_left_lat]
        
    return data

region_of_interest_polygon = None
def get_region_of_interest_polygon():
    global region_of_interest_polygon
    
    if region_of_interest_polygon is None:
        print(f"reading {parameters['region_of_interest_vertices']}")
        region_of_interest_polygon = pd.read_csv(parameters['region_of_interest_vertices'])
        
    return region_of_interest_polygon

def check_points_within_polygon(lons, lats, polygon_points):
    polygon = pygplates.PolygonOnSphere(zip(polygon_points[1::2], polygon_points[::2]))
    ret = []
    
    for lon, lat in zip(lons, lats):
        # print(lon,lat)
        try:
            if polygon.is_point_in_polygon((lat, lon)):
                ret.append(True)
            else:
                ret.append(False)
        except:
            ret.append(False)
            
    return ret

def get_mesh_points(poloygon_points):
    script_dir = os.path.dirname(__file__)
    data = pd.read_csv(script_dir+'/../GIS/global_mesh.csv', sep=' ', names=['lon', 'lat'])
    points_mask = check_points_within_polygon(data['lon'], data['lat'], poloygon_points)
    
    return data[points_mask]

def get_region_of_interest_extent():
    if len(parameters['map_extent']) == 4:
        return parameters['map_extent']
    
    region_of_interest_polygon = get_region_of_interest_polygon()
    left_lon = max(region_of_interest_polygon['lon'].min()-5, -180)
    right_lon = min(region_of_interest_polygon['lon'].max()+5, 180)
    bottom_lat = max(region_of_interest_polygon['lat'].min()-5, -90)
    top_lat = min(region_of_interest_polygon['lat'].max()+5, 90)
    
    return [left_lon, right_lon, bottom_lat, top_lat]

def download_agegrid(time):
    if not os.path.isdir('./AgeGrids'):
        os.system('mkdir AgeGrids')

    # download the age grid if necessary
    url_temp = get_parameter('agegrid_url')
    
    if not url_temp:
        print('Age grid url has not been set. Not downloading age grid file.')
        return None
    file_temp='./AgeGrids/'+os.path.basename(url_temp)
    agegrid_file = file_temp.format(time)
    if not os.path.isfile(agegrid_file):
        print(f'Downloading age grids to {agegrid_file}')
        myfile = requests.get(url_temp.format(time))
        open(agegrid_file, 'wb').write(myfile.content)
        
    return agegrid_file

def get_age_grid_color_map_from_cpt(cpt_file):
    values = []
    colors = []
    
    with open(cpt_file,'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line[0] in ['#', 'B', 'F', 'N']: continue
            vals = line.split()
            if len(vals) !=8: continue
            values.append(float(vals[0]))
            values.append(float(vals[4]))
            colors.append([float(vals[1]), float(vals[2]), float(vals[3])])
            colors.append([float(vals[5]), float(vals[6]), float(vals[7])])

    colour_list = []
    for i in range(len(values)):
        colour_list.append((values[i]/(values[-1]-values[0]),
                            [x/255.0 for x in colors[i]]))
        
    return LinearSegmentedColormap.from_list('agegrid_cmap', colour_list)

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

# function to make a velocity mesh nodes at an arbitrary set of points defined in Lat
# Long and Lat are assumed to be 1d arrays 
def make_GPML_velocity_feature(Long, Lat):
    # Add points to a multipoint geometry
    multi_point = pygplates.MultiPointOnSphere([(float(lat), float(lon)) for lat, lon in zip(Lat, Long)])

    # Create a feature containing the multipoint feature, and defined as MeshNode type
    meshnode_feature = pygplates.Feature(pygplates.FeatureType.create_from_qualified_string('gpml:MeshNode'))
    meshnode_feature.set_geometry(multi_point)
    meshnode_feature.set_name('Velocity Mesh Nodes from pygplates')

    output_feature_collection = pygplates.FeatureCollection(meshnode_feature)
    
    # NB: at this point, the feature could be written to a file using
    # output_feature_collection.write('myfilename.gpmlz')
    
    # for use within the notebook, the velocity domain feature is returned from the function
    
    return output_feature_collection

def Get_Plate_Velocities(velocity_domain_features, topology_features, rotation_model, time, delta_time, rep='vector_comp'):
    # All domain points and associated (magnitude, azimuth, inclination) velocities for the current time
    all_domain_points = []
    all_velocities = []

    # Partition our velocity domain features into our topological plate polygons at the current 'time'
    plate_partitioner = pygplates.PlatePartitioner(topology_features, rotation_model, time)

    for velocity_domain_feature in velocity_domain_features:
        # A velocity domain feature usually has a single geometry but we'll assume it can be any number
        # Iterate over them all.
        for velocity_domain_geometry in velocity_domain_feature.get_geometries():
            for velocity_domain_point in velocity_domain_geometry.get_points():
                all_domain_points.append(velocity_domain_point)
                partitioning_plate = plate_partitioner.partition_point(velocity_domain_point)
                if partitioning_plate:
                    # We need the newly assigned plate ID to get the equivalent stage rotation of that tectonic plate
                    partitioning_plate_id = partitioning_plate.get_feature().get_reconstruction_plate_id()

                    # Get the stage rotation of partitioning plate from 'time + delta_time' to 'time'
                    equivalent_stage_rotation = rotation_model.get_rotation(time, partitioning_plate_id, time + delta_time)

                    # Calculate velocity at the velocity domain point
                    # This is from 'time + delta_time' to 'time' on the partitioning plate
                    velocity_vectors = pygplates.calculate_velocities(
                        [velocity_domain_point],
                        equivalent_stage_rotation,
                        delta_time)
                    
                    if rep=='mag_azim':
                        # Convert global 3D velocity vectors to local (magnitude, azimuth, inclination) tuples (one tuple per point).
                        velocities = pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                            [velocity_domain_point],
                            velocity_vectors)
                        all_velocities.append(velocities[0])
                    elif rep=='vector_comp':
                        # Convert global 3D velocity vectors to local (magnitude, azimuth, inclination) tuples (one tuple per point).
                        velocities = pygplates.LocalCartesian.convert_from_geocentric_to_north_east_down(
                            [velocity_domain_point],
                            velocity_vectors)
                        all_velocities.append(velocities[0])
                else:
                    all_velocities.append((0, 0, 0))

    return all_velocities

def get_velocity_x_y_u_v(time, rotation_model, topology_filenames):
    delta_time = 5.
    Xnodes = np.arange(-180, 180, 10)
    Ynodes = np.arange(-90, 90, 10)
    Xg, Yg = np.meshgrid(Xnodes, Ynodes)
    Xg = Xg.flatten()
    Yg = Yg.flatten()
    velocity_domain_features = make_GPML_velocity_feature(Xg, Yg)

    # Load the topological plate polygon features
    topology_features = []
    
    for fname in topology_filenames:
        for f in pygplates.FeatureCollection(fname):
            topology_features.append(f)

    # Call the function we created above to get the velocities
    all_velocities = Get_Plate_Velocities(velocity_domain_features,
                                          topology_features,
                                          rotation_model,
                                          time,
                                          delta_time,
                                          'vector_comp')

    uu = []
    vv = []
    
    for vel in all_velocities:
        if not hasattr(vel, 'get_y'): 
            uu.append(vel[1])
            vv.append(vel[0])
        else:
            uu.append(vel.get_y())
            vv.append(vel.get_x())
    u = np.asarray([uu]).reshape((Ynodes.shape[0], Xnodes.shape[0]))
    v = np.asarray([vv]).reshape((Ynodes.shape[0], Xnodes.shape[0]))

    return Xnodes, Ynodes, u, v
    # compute native x, y coordinates of grid
    # x, y = m(Xg, Yg)

    # uproj, vproj, xx, yy = m.transform_vector(u, v, Xnodes, Ynodes, 15, 15, returnxy=True, masked=True)
    # plot
    # Q = m.quiver(xx, yy, uproj, vproj, scale=1000, color='grey')
    # make quiver key
    # qk = plt.quiverkey(Q, 0.95, 1.05, 50, '50 mm/yr', labelpos='W')

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
            teeth.append([PA,B0,C]) # three vertices of the triagle

        PA = PB
        
    return teeth

def get_anim_dir():
    return './animation/'

def get_plate_id(lons, lats):
    static_polygons = pygplates.FeatureCollection(parameters['static_polygons_file'])
    rotation_model = pygplates.RotationModel(parameters['rotation_file'])
    p_len = len(lons)
    
    assert (p_len == len(lats)), 'The lons and lats must have the same length.'
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

def get_coreg_input_dir():
    return parameters['coreg_input_dir'] + '/'

def get_deposit_candidates():
    polygon_points = get_region_of_interest_polygon().values.flatten()
    mesh_points = get_mesh_points(polygon_points)

    # find plate id for the mesh points
    mesh_plate_ids = get_plate_id(mesh_points.lon.tolist(), mesh_points.lat.tolist())
    mesh_points['plate_id'] =  mesh_plate_ids
    deposit_candidates = []
    start_time = parameters['time']['start']
    end_time = parameters['time']['end']
    time_step =  parameters['time']['step']
    
    for t in range(start_time, end_time+1, time_step):
        for index, p in mesh_points.iterrows():
            deposit_candidates.append([p['lon'], p['lat'], t, p['plate_id']]) 
    deposit_candidates = pd.DataFrame(deposit_candidates, columns=['lon','lat','age','plate_id'])
    deposit_candidates = deposit_candidates.astype({'plate_id': int, 'age': int})
    
    return deposit_candidates

def select_points_in_region(candidate_lons, candidate_lats, trench_lons, trech_lats, region=5):
    # build the tree
    points_3d = [pygplates.PointOnSphere((lat, lon)).to_xyz() for lon, lat in zip(trench_lons, trech_lats)]
    points_tree = scipy.spatial.cKDTree(points_3d)

    candidates = [pygplates.PointOnSphere((lat, lon)).to_xyz() for lon, lat in zip(candidate_lons, candidate_lats)]
    
    dists, indices = points_tree.query(candidates, k=1,
                                       distance_upper_bound=degree_to_straight_distance(region))

    return indices < len(points_3d)

def get_coreg_output_dir():
    return parameters['coreg_output_dir'] + '/'

def print_columns():
    columns = [
        'reconstructed mineral deposits longitude',
        'reconstructed mineral deposits latitude',
        'distance to the nearest trench point',
        'the index of trench point',
        'trench point longitude',
        'trench point latitude',
        'subducting convergence (relative to trench) velocity magnitude (in cm/yr)',
        'subducting convergence velocity obliquity angle (angle between trench normal vector and convergence velocity vector)',
        'trench absolute (relative to anchor plate) velocity magnitude (in cm/yr)',
        'trench absolute velocity obliquity angle (angle between trench normal vector and trench absolute velocity vector)',
        'length of arc segment (in degrees) that current point is on',
        'trench normal azimuth angle (clockwise starting at North, ie, 0 to 360 degrees) at current point',
        'subducting plate ID',
        'trench plate ID',
        'distance (in degrees) along the trench line to the nearest trench edge',
        'the distance (in degrees) along the trench line from the start edge of the trench',
        'convergence velocity orthogonal (in cm/yr)',
        'convergence velocity parallel (in cm/yr)',
        'the trench plate absolute velocity orthogonal (in cm/yr)',
        'the trench plate absolute velocity parallel (in cm/yr)',
        'the subducting plate absolute velocity magnitude (in cm/yr)',
        'the subducting plate absolute velocity obliquity angle (in degrees)',
        'the subducting plate absolute velocity orthogonal',
        'the subducting plate absolute velocity parallel',
        'sea floor age',
        'subduction volume(km3y)',
        'decompacted sediment thickness',
        'sediment thickness',
        'ocean crust carb percent'
        ]
    for i in range(len(columns)):
        print('*', i, columns[i])

def get_ml_input_dir():
    return parameters['ml_input_dir'] + '/'

def get_ml_output_dir():
    return parameters['ml_output_dir'] + '/'

def download_grid(time, url_temp):
    if not os.path.isdir('./grids'):
        os.system('mkdir grids')

    # download the grid 
    if not url_temp:
        print('Grid url has not been set. Not downloading grid file.')
        return None
    file_temp='./grids/'+os.path.basename(url_temp)
    filename = file_temp.format(time)
    if not os.path.isfile(filename):
        print(f'Downloading grids to {filename}')
        myfile = requests.get(url_temp.format(time))
        open(filename, 'wb').write(myfile.content)
    return filename

'''
def create_dirs():
    casename = parameters['case_name']
    dirs = [casename,
            casename+'/'+parameters['coreg_input_dir'],
            casename+'/'+parameters['coreg_output_dir'],
            casename+'/'+parameters['ml_output_dir'],
            casename+'/'+parameters['ml_input_dir'],
            casename+'/'+parameters['convergence_data_dir'],
            casename+'/animation/']
    print(dirs)
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)
'''

def set_parameter(name, value):
    parameters[name] = value
    return

def get_agegrid_dir():
    return parameters['agegrid_dir'] + '/'
