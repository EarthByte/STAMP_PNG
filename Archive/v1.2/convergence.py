import math, os, sys
import pandas as pd
import Utils
sys.path.append(Utils.get_parameter('plate_tectonic_tools_path'))
from subduction_convergence import subduction_convergence_over_time

# basically the subduction_convergence.py does most of the work.
# see https://github.com/EarthByte/PlateTectonicTools/blob/master/ptt/subduction_convergence.py

# The columns in the output file
# * 0 lon
# * 1 lat
# * 2 subducting convergence (relative to trench) velocity magnitude (in cm/yr)
# * 3 subducting convergence velocity obliquity angle (angle between trench normal vector and convergence velocity vector)
# * 4 trench absolute (relative to anchor plate) velocity magnitude (in cm/yr)
# * 5 trench absolute velocity obliquity angle (angle between trench normal vector and trench absolute velocity vector)
# * 6 length of arc segment (in degrees) that current point is on
# * 7 trench normal azimuth angle (clockwise starting at North, ie, 0 to 360 degrees) at current point
# * 8 subducting plate ID
# * 9 trench plate ID
# * 10 distance (in degrees) along the trench line to the nearest trench edge
# * 11 the distance (in degrees) along the trench line from the start edge of the trench
# * 12 convergence velocity orthogonal (in cm/yr)
# * 13 convergence velocity parallel  (in cm/yr) 
# * 14 the trench plate absolute velocity orthogonal (in cm/yr)
# * 15 the trench plate absolute velocity parallel (in cm/yr)
# * 16 the subducting plate absolute velocity magnitude (in cm/yr)
# * 17 the subducting plate absolute velocity obliquity angle (in degrees)
# * 18 the subducting plate absolute velocity orthogonal       
# * 19 the subducting plate absolute velocity parallel

def run_it():
    print('Generating convergence data ...')
    time_period = Utils.get_parameter('time')
    start_time = time_period['start'] 
    end_time = time_period['end']
    time_step = time_period['step']

    conv_dir = Utils.get_parameter('convergence_data_dir')
    conv_prefix = Utils.get_parameter('convergence_data_filename_prefix')
    conv_ext = Utils.get_parameter('convergence_data_filename_ext')

    if not os.path.exists(conv_dir):
        os.makedirs(conv_dir)
    else:
        if (os.path.isfile(f'{conv_dir}/{conv_prefix}_0.00.{conv_ext}') and  
            not Utils.get_parameter('overwrite_existing_convergence_data')):
            print(f'Convergence data already exist in {conv_dir}.')
            
            return
    
    rotation_files = Utils.get_files(Utils.get_parameter('rotation_file'))
    topology_files = Utils.get_files(Utils.get_parameter('topology_file'))
   
    kwargs = {
        'output_distance_to_nearest_edge_of_trench':True,
        'output_distance_to_start_edge_of_trench':True,
        'output_convergence_velocity_components':True,
        'output_trench_absolute_velocity_components':True,
        'output_subducting_absolute_velocity':True,
        'output_subducting_absolute_velocity_components':True}
    
    return_code = subduction_convergence_over_time(
            conv_dir + '/' + conv_prefix,
            conv_ext,
            rotation_files,
            topology_files,
            math.radians(Utils.get_parameter('threshold_sampling_distance_degrees')),
            start_time,
            end_time,
            time_step,
            Utils.get_parameter('velocity_delta_time'),
            Utils.get_parameter('anchor_plate_id'),
            output_gpml_filename = None,
            **kwargs)

    # There are some more data acquired from various grids.
    # Append the additional data to the subduction convergence kinematics statistics.
    for age in reversed(range(start_time, end_time+1, time_step)):
        # print(age, end=' ')
        
        column_list = ['trench_lon', 'trench_lat', 'conv_rate', 'conv_angle', 'trench_abs_rate', 'trench_abs_angle',
            'arc_len', 'trench_norm', 'subducting_pid', 'trench_pid', 'dist_nearest_edge', 'dist_from_start',
            'conv_ortho', 'conv_paral', 'trench_abs_ortho', 'trench_abs_paral', 'subducting_abs_rate',
            'subducting_abs_angle', 'subducting_abs_ortho', 'subducting_abs_paral']
        
        trench_file = conv_dir + conv_prefix + f'_{age:.2f}.' + conv_ext
        trench_data= pd.read_csv(trench_file, sep=' ', header=None, names=column_list)
        
        for grid in Utils.get_parameter('grid_files'):
            print(f'Querying {grid[0].format(time=age)}, {grid[1]}')
            grid_name = grid[1]
            grid_data = Utils.query_raster(
                grid[0].format(time=age),
                trench_data.iloc[:, 0],
                trench_data.iloc[:, 1],
                10) # region of interest (degrees), try to find the nearest valid data within 10 degrees
            
            trench_data[grid_name] = grid_data

            if grid_name == 'seafloor_age':
                thickness = [None]*len(grid_data)
                T1 = 1150.
                for i in range(len(grid_data)):
                    thickness[i] = Utils.plate_isotherm_depth(grid_data[i], T1)

                # To convert arc_length from degrees on a sphere to m (using earth's radius = 6371000 m)
                arc_length_m = 2*math.pi*6371000*trench_data.arc_len/360

                # Calculate Subduction Volume (in m^3 per year)
                subduction_volume_m3y = trench_data.conv_ortho/100 * thickness * arc_length_m

                # Calculate Subduciton Volume (slab flux) (in km^3 per year)
                subduction_volume_km3y = subduction_volume_m3y/1e9
                subduction_volume_km3y[subduction_volume_km3y<0] = 0
                
                trench_data['subduction_volume_km3y'] = subduction_volume_km3y
                
        trench_data.to_csv(conv_dir + conv_prefix + f'_{age}.00.csv', index=False, float_format='%.2f', na_rep='nan')

    print('')
    print('Convergence completed successfully!')
    print('The result data has been saved in {}!'.format(conv_dir))

# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print('Usage: python3 convergence.py config.json 2> convergence.log')
#         sys.exit()
#     else:
#         Utils.load_config(sys.argv[1])
#         run_it()
