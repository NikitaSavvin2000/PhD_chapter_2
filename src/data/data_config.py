datasets_csv_dict = {
    "Australia_Bundoora": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTl3ZMKUEqYeXJe1b8A4IbfYIKjWlm0lR61glDoXOEfHxsmDUv1ZZ2IK2GpjkH2fZ6fvX3NaCOryqzW/pub?gid=751874949&single=true&output=csv",
        "col_time": "Datetime",
        "col_target": "consumption",
    },
    "Australia Albury-Wodonga": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQmRJCXCBp-qsY4LQrf8x_zJax_5FAnZDl6-sv1zje9m0pCM7hore-cjS3zlzJezgHIm6h81KY1hsEz/pub?gid=1184660391&single=true&output=csv",
        "col_time": None,
        "col_target": None,
    },
    "Australia Bendigo": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQgXCJsm0V7ylsqvzRzK_LHZzky0lABeXvRuiqRWzDumN1Y8i8xul-Ih1ERIU1v-C46AKISnOOzBmtb/pub?gid=1902219272&single=true&output=csv",
        "col_time": None,
        "col_target": None,
    },
    "morocco_zone_1": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
        "col_time": "Datetime",
        "col_target": "consumption",
    },
    "Morocco Zone 2": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQT1DfqAB5Yec8MIQ_E5A8w-SXNcRmTwbXsv2W-ZT1ZcXN_G83BHlb6QBgnWkO-MpH3oVgfLoE0SnLx/pub?gid=1952392108&single=true&output=csv",
        "col_time": None,
        "col_target": None,
    },
    "Morocco Zone 3": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSHw5k7n3_RM6ksGbvdQJsa1i9-zF-18CFLCFnXFkCxQwqLcQ4Wu2_8EF2H1lF02ih2NLL9BDecFzQ/pub?gid=1952392108&single=true&output=csv",
        "col_time": None,
        "col_target": None,
    },
    "italy_temp": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRFF6SXvGbQgQG1bh0hwbXgVWpUU_UG8OQAVhHcNAcAFT5x-XoIYxMAeF-goym6_wNhJEwQd3iGHp9b/pub?gid=791211028&single=true&output=csv",
        "col_time": None,
        "col_target": None,
    },
    "DEKENERG_ZONE2_S_PAMURENE": {
        "csv_link": "/Users/nikitasavvin/Desktop/Business/repo/collection_rus_energy_data/result_end/DEKENERG_ZONE2_S_PAMURENE.csv",
        "col_time": None,
        "col_target": None,
    },
    "russia_amur_region": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnke5YReb1RRwN6ZynJSxoUcXnr2buEXP45mOtnv7X3neOQhT-_vMPnIB4lUBpeu0nQaqWq6awDvtG/pub?gid=935054795&single=true&output=csv",
        "col_time": "datetime",
        "col_target": "VC_факт",
    },
    "russia_elista": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTK8aQxQ15u_BZDSe1Qz5HW1gm-C7pLCEoJ8OrBskXMFgOzgUblKdJzGMUNFCnHe8JjPOBJzMcp1W8V/pub?gid=442094039&single=true&output=csv",
        "col_time": "datetime",
        "col_target": "value",
    },
    "Istanbul_Traffic_Index": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vT6LSgQb7b1u-c_zBRhvkEA_CmNd6Cby2tlMLBwoX9GlSsUHjJvdh9Xz9zyQx5HbxQAUH2GEVrn95U7/pub?gid=2111659113&single=true&output=csv",
        "col_time": "datetime",
        "col_target": "maximum_traffic_index",
    },
    "Air_Quality_India": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQTTyEI6cJP1q0SIUlXa46vpBFGdCe6BuI4Y9RxvJ8ZCkSarq85wFAlox6qbcVvsZ3fdHdErmugSJ2l/pub?gid=1652276659&single=true&output=csv",
        "col_time": "Timestamp",
        "col_target": "PM2.5",
    },
    "Daily_Climate": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vT6-YtMLu5kDfAfr4ISYnMOulJf4IU3RNAe6vRo7adY_A4Hot6K3vWtO0HKsrAZjNng0cMQkxIxvQbp/pub?gid=81811996&single=true&output=csv",
        "col_time": "date",
        "col_target": "meantemp",
    },
    "NYC_Taxi_Traffic": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQgq9c-tCaL3ZFDLoFvCaB-dvW1SPwO2u8RHYMGpNAzg4O0Bl-kjRaKatD2i-V2-uafFcNOKYcMytnZ/pub?gid=797724504&single=true&output=csv",
        "col_time": "timestamp",
        "col_target": "value",
    },
    "Weather": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbZQWfPBOCnG6OQI5uzi886u0ZjOYeNdCcyJCx3up2bjjPdm7BX20akl8oSM2rjpiEoAtQmREK2s_7/pub?gid=2018623599&single=true&output=csv",
        "col_time": "date",
        "col_target": "T",
    },
    "Weather_water_temp": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbZQWfPBOCnG6OQI5uzi886u0ZjOYeNdCcyJCx3up2bjjPdm7BX20akl8oSM2rjpiEoAtQmREK2s_7/pub?gid=2018623599&single=true&output=csv",
        "col_time": None,
        "col_target": None,
    },
    "Metro_Traffic": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSeR7Yjg4gVAUkdivDrivB1IIkOuBMyrjyydu0MF5-_TrOCroOndmY-__kQd0H_prwmfp08EDtEVdme/pub?gid=1260692357&single=true&output=csv",
        "col_time": "date_time",
        "col_target": "traffic_volume",
    },
    "Air_Pollution": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTms7XrNlLNAHq1r9CMhpr8CpcC3wpLDkqJN4lb376f1jq31up9FPVvDVNEDYoAGY32iDmNNGhL_K1n/pub?gid=1982192545&single=true&output=csv",
        "col_time": None,
        "col_target": None,
    },
    "Temperature_in_Celsius": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSVuumAit9fVNgMukSYmD8Xf6fdjc0DMsdwS5a1ObXvgWSEHsedi-E36TZ0RlV9Y-9dJp6x1G6ta2A3/pub?gid=327522163&single=true&output=csv",
        "col_time": "Datetime",
        "col_target": "Value",
    },
    "Italian_init_PL": {
        "csv_link": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRKJzXJlQSNew0dxrQ8mFQMhBv_4owvfsF2If1b-rmxMZkR5gabHC4OiaSwt8Ul1Omc8taR27UohSeg/pub?gid=429486365&single=true&output=csv",
        "col_time": "time",
        "col_target": "P_l",
    }
}

