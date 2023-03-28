# These default parameters can be overridden in config.json
# import os
# script_dir = os.path.dirname(__file__)

parameters = {
    # IMPORTANT:
    # the convergence.py depends on the PlateTectonicTools,
    # you need to tell the script where to find it
    # The PlateTectonicTools repository https://github.com/EarthByte/PlateTectonicTools.git is a submodule now
    # run the following command after you have cloned this spatio-temporal exploration repository.
    # git submodule update --init --recursive

    'agegrid_dir': '../Muller_2016/Muller_etal_2016_AREPS_v1.17_netCDF/',
    'agegrid_url' : 'https://www.earthbyte.org/webdav/ftp/Data_Collections/Muller_etal_2016_AREPS/Muller_etal_2016_AREPS_Agegrids/Muller_etal_2016_AREPS_Agegrids_v1.17/Muller_etal_2016_AREPS_v1.17_netCDF/Muller_etal_2016_AREPS_v1.17_AgeGrid-{}.nc',
    
    'coastlines_file' : '../Muller_2016/Global_EarthByte_230-0Ma_GK07_AREPS_Coastlines.gpmlz',
    'rotation_file' : ['../Muller_2016/Global_EarthByte_230-0Ma_GK07_AREPS.rot'],
    'static_polygons_file' : '../Muller_2016/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons_2015_v1.shp',
    'topology_file' : ['../Muller_2016/Global_EarthByte_230-0Ma_GK07_AREPS_PlateBoundaries.gpml.gz',
                        '../Muller_2016/Global_EarthByte_230-0Ma_GK07_AREPS_Topology_BuildingBlocks.gpml.gz'],

    'anchor_plate_id' : 0, # see https://www.gplates.org/user-manual/MoreReconstructions.html
    
    'case_name' : 'default-test-case-name',
    
    'convergence_data_dir' : './convergence_kinematic_features/',
    'convergence_data_filename_prefix' : 'features',
    'convergence_data_filename_ext' : 'csv',

    # folder contains the coregistration input files.
    'coreg_input_dir' : './coreg_input/',

    # the file which contains the seed points.
    # coregistration input file
    'coreg_input_files' : ['mineral_occurrences.csv', 'random_samples.csv', 'target_points'],
    # folder contains the coregistration output files.
    'coreg_output_dir' : './coreg_output/',

    # a list of grid files from which the coregistration scripts query data
    'grid_files' : [
        ['../Muller_2016/Muller_etal_2016_AREPS_v1.17_netCDF/Muller_etal_2016_AREPS_v1.17_AgeGrid-{time:d}.nc', 'seafloor_age'],
        ['../Muller_2016/carbonate_sediment_thickness_grids/compacted_sediment_thickness_0.5_{time:d}.nc', 'carbonate_sediment_thickness'],
        ['../Muller_2016/sediment_thickness_grids/sed_thick_0.2d_{time:d}.nc', 'total_sediment_thick'],
        ['../Muller_2016/oceanic_crustal_CO2_grids/co2_percent_{time:d}.nc', 'ocean_crust_carb_percent']
    ],

    'target_extent_file' : '../GIS/target_extent.shp',
    
    # 'min_occ_file' : '../GIS/min_occ_holm_porphyry.shp',
    'min_occ_file' : '../GIS/min_occ_holm.shp',

    # folder contains the machine learning input files.
    'ml_input_dir' : './ml_input/',

    # folder contains the machine learning output files.
    'ml_output_dir' : './ml_output/',

    'overwrite_existing_convergence_data' : False, # if True, always generate the new convergence data

    'plate_tectonic_tools_path' : '../PlateTectonicTools/ptt/',

    # the region of interest parameters are used in coregistration.py
    # given a seed point, the coregistration code looks for the nearest geomery within region[0] first
    # if not found, continue to search in region[1], region[2], etc
    # if still not found after having tried all regions, give up
    # the distance is in degrees.
    'region_of_interest_vertices' : '../GIS/PNG_Papua_Dissolved_CH_Points.csv',
    'region_of_interest_polygon' : '../GIS/PNG_Papua_Dissolved.shp',
    'regions' : [5, 10, 15, 20],

    'selected_features' : ['distance', 'conv_rate', 'conv_angle', 'trench_abs_rate',
                       'trench_abs_angle', 'arc_len', 'trench_norm', 'dist_nearest_edge',
                       'dist_from_start', 'conv_ortho', 'conv_paral', 'trench_abs_ortho',
                       'trench_abs_paral', 'subducting_abs_rate', 'subducting_abs_angle',
                       'subducting_abs_ortho', 'subducting_abs_paral', 'seafloor_age',
                       'subduction_volume_km3y', 'carbonate_sediment_thickness',
                       'total_sediment_thick', 'ocean_crust_carb_percent'],

    'terranes' : '',

    # the following two parameters are used by subduction_convergence
    # see https://github.com/EarthByte/PlateTectonicTools/blob/master/ptt/subduction_convergence.py
    'threshold_sampling_distance_degrees' : 0.2,
    'velocity_delta_time' : 1,

    'time' : {
        'start' : 0,
        'end'   : 230,
        'step'  : 1
    },

    'topo_grid_file' : '../GIS/topo15_3600x1800.nc',
    
    # a list of file paths from which the coregistration scripts query data
    'vector_files' : [
        '{conv_dir}features_{time:.2f}.csv', # can be generated by convergence.py
        # './convergence_data/features_{time:.2f}.csv',
        ]
    }