parameters = {
    'plate_motion_model' : 'muller2016',
    
    'rotation_files' : '../Muller_2016/muller2016/Global_EarthByte_230-0Ma_GK07_AREPS.rot',
    'topology_files' : ['../Muller_2016/muller2016/Global_EarthByte_230-0Ma_GK07_AREPS_PlateBoundaries.gpml',
                        '../Muller_2016/muller2016/Global_EarthByte_230-0Ma_GK07_AREPS_Topology_BuildingBlocks.gpml'],
    'coastlines_file' : '../Muller_2016/muller2016/Global_EarthByte_230-0Ma_GK07_AREPS_Coastlines.gpml',
    'cob_file' : '../Muller_2016/muller2016/Global_EarthByte_230-0Ma_GK07_AREPS_IsoCOB.gpml',
    'static_polygons_file' : '../Muller_2016/muller2016/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons_2015_v1.gpml',
    'terranes_file' : '../Muller_2016/muller2016/Global_EarthByte_230-0Ma_GK07_AREPS_COB_Terranes_polygon.gpml',

    # a list of grid files from which the coregistration scripts query data
    'grid_files' : [
        ['../Muller_2016/carbonatethick/compacted_sediment_thickness_0.1_{time:d}.nc', 'carbonate_sediment_thickness_m'],
        ['../Muller_2016/carbonper/co2_percent_{time:d}.nc', 'ocean_crust_carb_percent'],
        ['../Muller_2016/agegrid/Muller_etal_2016_AREPS_v1.17_AgeGrid-{time:d}.nc', 'seafloor_age_ma'],
        ['../Muller_2016/spreadrate_old/rategrid_{time:d}.nc', 'seafloor_spread_rate_mm_yr'],
        ['../Muller_2016/sedthick/sed_thick_0.1d_{time:d}.nc', 'total_sediment_thick_m']
    ],

    'agegrid_dir': '../Muller_2016/agegrid/',
    
    'convergence_data_dir' : './kinematic_features_muller2016/',
    'convergence_data_filename_prefix' : 'features',
    'convergence_data_filename_ext' : 'csv',

    # folder contains the coregistration input files
    'coreg_input_dir' : './coreg_input_muller2016/',
    # the file which contains the seed points
    # coregistration input file
    'coreg_input_files' : ['mineral_occurrences.csv', 'random_samples.csv', 'target_points'],
    # folder contains the coregistration output files
    'coreg_output_dir' : './coreg_output_muller2016/',

    # folder contains the machine learning input files
    'ml_input_dir' : './ml_input_muller2016/',
    
    'augmentation_dir' : './augmentation/',

    # folder contains the machine learning output files
    'ml_output_dir' : './ml_output_muller2016/',
    
    'min_occ_prob_dir' : './min_occ_prob_muller2016/',
    
    'selected_features' : [
        'arc_len_deg',
        'conv_angle_deg',
        'conv_ortho_cm_yr',
        'conv_paral_cm_yr',
        'conv_rate_cm_yr',
        'dist_nearest_edge_deg',
        'distance_deg',
        'trench_abs_angle_deg',
        'trench_abs_ortho_cm_yr',
        'trench_abs_paral_cm_yr',
        'trench_abs_rate_cm_yr',
        'subducting_abs_angle_deg',
        'subducting_abs_ortho_cm_yr',
        'subducting_abs_paral_cm_yr',
        'subducting_abs_rate_cm_yr',
        'carbonate_sediment_thickness_m',
        'ocean_crust_carb_percent',
        'seafloor_age_ma',
        'seafloor_spread_rate_mm_yr',
        'subduction_volume_km3_yr',
        'total_sediment_thick_m',
        ],
    
    'selected_features_names' : [
        'Length of the arc segment (deg)',
        'Relative motion vector (obliquity angle, deg)',
        'Relative motion vector (orthogonal, cm/yr)',
        'Relative motion vector (parallel, cm/yr)',
        'Relative motion vector (magnitude, cm/yr)',
        'Distance to the nearest trench edge (deg)',
        'Distance to the nearest trench point (deg)',
        'Overriding absolute plate velocity (obliquity angle, deg)',
        'Overriding absolute plate velocity (orthogonal, cm/yr)',
        'Overriding absolute plate velocity (parallel, cm/yr)',
        'Overriding absolute plate velocity (magnitude, cm/yr)',
        'Downgoing absolute plate velocity (obliquity angle, deg)',
        'Downgoing absolute plate velocity (orthogonal, cm/yr)',
        'Downgoing absolute plate velocity (parallel, cm/yr)',
        'Downgoing absolute plate velocity (magnitude, cm/yr)',
        'Deep-sea carbonate sediment thickness (m)',
        'Upper ocean crust carbonate concentration (%)',
        'Seafloor age (Ma)',
        'Seafloor spreading rate (mm/yr)',
        'Subducting plate volume (km^3/yr)',
        'Total deep-sea sediment thickness (m)'
        ],
    
    'selected_features_names_nounit' : [
        'Length of the arc segment',
        'Relative motion vector (obliquity angle)',
        'Relative motion vector (orthogonal)',
        'Relative motion vector (parallel)',
        'Relative motion vector (magnitude)',
        'Distance to the nearest trench edge',
        'Distance to the nearest trench point',
        'Overriding absolute plate velocity (obliquity angle)',
        'Overriding absolute plate velocity (orthogonal)',
        'Overriding absolute plate velocity (parallel)',
        'Overriding absolute plate velocity (magnitude)',
        'Downgoing absolute plate velocity (obliquity angle)',
        'Downgoing absolute plate velocity (orthogonal)',
        'Downgoing absolute plate velocity (parallel)',
        'Downgoing absolute plate velocity (magnitude)',
        'Deep-sea carbonate sediment thickness',
        'Upper ocean crust carbonate concentration',
        'Seafloor age',
        'Seafloor spreading rate',
        'Subducting plate volume',
        'Total deep-sea sediment thickness'
        ],

    # the following parameters are used by subduction_convergence
    # see https://github.com/EarthByte/PlateTectonicTools/blob/master/ptt/subduction_convergence.py
    'threshold_sampling_distance_degrees' : 0.1,
    'anchor_plate_id' : 0, # see https://www.gplates.org/user-manual/MoreReconstructions.html
    'velocity_delta_time' : 1,

    'time' : {
        'start' : 0,
        'end'   : 50,
        'step'  : 1
        },
    
    'min_occ_file' : '../GIS/min_occ_holm.shp',
    
    'target_extent_file' : '../GIS/target_extent.shp'
    }
