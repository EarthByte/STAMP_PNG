parameters = {
    'plate_motion_model' : 'muller2019',
    
    'rotation_dir' : '../Muller_2019/rotation',
    'topology_dir' : '../Muller_2019/topology',
    'coastlines_file' : '../Muller_2019/coastlines/Global_coastlines_2019_v1_low_res.shp',
    'cob_file' : '../Muller_2019/Muller_etal_2019_PlateMotionModel_v2.0_Tectonics/StaticGeometries/COBLineSegments/Global_EarthByte_GeeK07_COBLineSegments_2019_v1.shp',
    'static_polygons_file' : '../Muller_2019/static_polygons/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons_2019_v1.shp',
    'terranes_file' : '../Muller_2019/Global_EarthByte_GeeK07_Terranes_ContinentsOnly.gpmlz',

    # a list of grid files from which the coregistration scripts query data
    'grid_files' : [
        ['../Muller_2019/carbonatethick/compacted_sediment_thickness_0.5_{time:d}.nc', 'carbonate_sediment_thickness_m'],
        ['../Muller_2019/carbonper/ocean_crust_carb_percent_{time:d}.nc', 'ocean_crust_carb_percent'],
        ['../Muller_2019/agegrid/Muller_etal_2019_Tectonics_v2.0_AgeGrid-{time:d}.nc', 'seafloor_age_ma'],
        ['../Muller_2019/spreadrate/rategrid_final_mask_{time:d}.nc', 'seafloor_spread_rate_mm_yr'],
        ['../Muller_2019/sedthick/sed_thick_0.1d_{time:d}.nc', 'total_sediment_thick_m']
    ],

    'agegrid_dir': '../Muller_2019/agegrid/',
    
    'convergence_data_dir' : './kinematic_features_muller2019/',
    'convergence_data_filename_prefix' : 'features',
    'convergence_data_filename_ext' : 'csv',

    # folder contains the coregistration input files
    'coreg_input_dir' : './coreg_input_muller2019/',
    # the file which contains the seed points
    # coregistration input file
    'coreg_input_files' : ['mineral_occurrences.csv', 'random_samples.csv', 'target_points'],
    # folder contains the coregistration output files
    'coreg_output_dir' : './coreg_output_muller2019/',

    # folder contains the machine learning input files
    'ml_input_dir' : './ml_input_muller2019/',
    
    'augmentation_dir' : './augmentation/',

    # folder contains the machine learning output files
    'ml_output_dir' : './ml_output_muller2019/',
    
    'min_occ_prob_dir' : './min_occ_prob_muller2019/',
    
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
        'Relative motion vector (Obliquity angle, deg)',
        'Relative motion vector (Orthogonal, cm/yr)',
        'Relative motion vector (Parallel, cm/yr)',
        'Relative motion vector (Magnitude, cm/yr)',
        'Distance to the nearest trench edge (deg)',
        'Distance to the nearest trench point (deg)',
        'Overriding absolute plate velocity (Obliquity angle, deg)',
        'Overriding absolute plate velocity (Orthogonal, cm/yr)',
        'Overriding absolute plate velocity (Parallel, cm/yr)',
        'Overriding absolute plate velocity (Magnitude, cm/yr)',
        'Downgoing absolute plate velocity (Obliquity angle, deg)',
        'Downgoing absolute plate velocity (Orthogonal, cm/yr)',
        'Downgoing absolute plate velocity (Parallel, cm/yr)',
        'Downgoing absolute plate velocity (Magnitude, cm/yr)',
        'Deep sea carbonate sediment thickness (m)',
        'Upper ocean crust carbonate concentration (%)',
        'Seafloor age (Ma)',
        'Seafloor spreading rate (mm/yr)',
        'Subducting plate volume (km^3/yr)',
        'Total deep sea sediment thickness (m)'
        ],
    
    'selected_features_names_nounit' : [
        'Length of the arc segment',
        'Relative motion vector (Obliquity angle)',
        'Relative motion vector (Orthogonal)',
        'Relative motion vector (Parallel)',
        'Relative motion vector (Magnitude)',
        'Distance to the nearest trench edge',
        'Distance to the nearest trench point',
        'Overriding absolute plate velocity (Obliquity angle)',
        'Overriding absolute plate velocity (Orthogonal)',
        'Overriding absolute plate velocity (Parallel)',
        'Overriding absolute plate velocity (Magnitude)',
        'Downgoing absolute plate velocity (Obliquity angle)',
        'Downgoing absolute plate velocity (Orthogonal)',
        'Downgoing absolute plate velocity (Parallel)',
        'Downgoing absolute plate velocity (Magnitude)',
        'Deep sea carbonate sediment thickness',
        'Upper ocean crust carbonate concentration',
        'Seafloor age',
        'Seafloor spreading rate',
        'Subducting plate volume',
        'Total deep sea sediment thickness'
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
    
    # 'min_occ_file' : '../GIS/min_occ_holm_porphyry.shp',
    'min_occ_file' : '../GIS/min_occ_holm.shp',
    
    'target_extent_file' : '../GIS/target_extent.shp'
    }
